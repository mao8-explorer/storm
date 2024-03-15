""" Example spawning a robot in gym 
只关心 运动规划问题 mpc with TrackIK_Guild
无碰撞
无SDF参与
无MultiModal

"""
from FrankaEnvBase import FrankaEnvBase
from utils import LimitedQueue , IKProc
import torch
import numpy as np
from storm_kit.gym.core import Gym
from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml
from storm_kit.mpc.task.reacher_task import ReacherTask
import rospy
import queue
import time
import csv


class IKSolve:
    def __init__(self):
        self.num_proc = 1
        self.maxsize = 5
        self.output_queue = LimitedQueue(self.maxsize)
        self.ik_procs = []
        for _ in range(self.num_proc):
            self.ik_procs.append(
                IKProc(
                    self.output_queue,
                    input_queue_maxsize=self.maxsize,
                )
            )
            self.ik_procs[-1].daemon = True #守护进程 主进程结束 IKProc进程随之结束
            self.ik_procs[-1].start()       


class MPCRobotController(FrankaEnvBase):
    def __init__(self, gym_instance , ik_mSolve):
        super().__init__(gym_instance = gym_instance)
        self.mpc_control = ReacherTask( self.mpc_config, self.world_description, self.tensor_args )
        self._environment_init()
        x,z,y = 0.45 , 0.45 , 0.45
        self.goal_list = [
             [x,y, z],
             [-x,y, z],
             [-x,y, -z],
             [x,y, -z],
             ]
        # self.goal_list = [
        #      [0.20,0.30,-0.65],
        #      [0.20,0.30,0.65],]
        self.goal_state = self.goal_list[-1]
        self.update_goal_state()
        self.rollout_fn = self.mpc_control.controller.rollout_fn
        self.goal_ee_transform = np.eye(4)
        # 暂行多进程方案是通过传参的方式 引导ik_proc句柄 保证ik_proc在主进程启动 避免无法共享内存的问题
        self.ik_mSolve = ik_mSolve

        self.fieldnames = ['whole_time', 'opt_step_count', 'collison_count', 'crash_rate', 
                      'ee_path_length', 'joints_path_length', 
                      'Avg.Speed', 'Max.Speed','Mean_weight',
                      'oneLoop','oneOpt','Note'] 
        
    
        self.sim_dt = self.mpc_control.exp_params['control_dt']
        self.lap_count = 8
        self.thresh = 0.02 # goal next thresh in Cart


    def run(self):
        self.goal_flagi = -1 # 调控目标点
        t_step = gym_instance.get_sim_time()
        self.jnq_des = np.zeros(7)
        last = time.time()
        opt_step_count = 0 
        self.curr_collision = 0
        opt_time_sum = 0
        self.crash_rate = 0.0
        self.collision_hanppend = False
        while not rospy.is_shutdown() and \
                self.goal_flagi / len(self.goal_list) != self.lap_count:
            try:
                self.gym_instance.step()
                self.gym_instance.clear_lines()
                # monitor ee_pose_gym and update goal_param_mpc
                self.monitorMPCGoalupdate()
                # seed goal to MPC_Policy _ get Command
                t_step += self.sim_dt
                self.current_robot_state = self.robot_sim.get_state(self.env_ptr, self.robot_ptr) # "dict: pos | vel | acc"
                qinit = self.current_robot_state['position'] # shape is (7,)
                self.goal_ee_transform[:3,3] = self.rollout_fn.goal_ee_pos.cpu().numpy()
                self.goal_ee_transform[:3,:3] = self.rollout_fn.goal_ee_rot.cpu().numpy()
                # 逆解获取请求发布 input_queue
                self.ik_mSolve.ik_procs[-1].ik(self.goal_ee_transform , qinit , ind = t_step)
                opt_time_last = time.time()
                command = self.mpc_control.get_command(self.current_robot_state)
                opt_time_sum += time.time() - opt_time_last
                # get position command:
                self.command = command
                q_des ,qd_des ,qdd_des = command['position'] ,command['velocity'] , command['acceleration']
                self.curr_state_tensor = torch.as_tensor(np.hstack((q_des,qd_des,qdd_des)), **self.tensor_args).unsqueeze(0) # "1 x 3*n_dof"
                # trans ee_pose in robot_coordinate to world coordinate
                self.updateGymVisual_GymGoalUpdate()
                self.visual_top_trajs_ingym()
                # Command_Robot_State include keyboard control : SPACE For Pause | ESCAPE For Exit 
                successed = self.robot_sim.command_robot_state(q_des, qd_des, self.env_ptr, self.robot_ptr)
                if not successed : break 
                if self.goal_flagi > -1 :
                    self.traj_append()
                    opt_step_count += 1
                    self.traj_log['collision'].append(0.0)
                # 逆解获取查询 output_queue
                try :
                    output = self.ik_mSolve.output_queue.get()
                    if output[1] is not None: # 无解
                        self.rollout_fn.goal_jnq = torch.as_tensor(output[1], **self.tensor_args).unsqueeze(0) # 1 x n_dof
                        self.jnq_des = output[1]
                    else : 
                        self.rollout_fn.goal_jnq = None
                        self.jnq_des = np.zeros(7)
                        print("warning: no iksolve")
                except queue.Empty:
                    "针对 output_queue队列为空的问题 会出现queue.Empty的情况发生"
                    continue
            except KeyboardInterrupt:
                print('Closing')

        avgvel, maxvel, ee_traj_length, joints_path_length = self.ee_vel_evaluate()

        row = {
            'whole_time': round(time.time() - last, 3),
            'opt_step_count': opt_step_count, 
            'collison_count':self.curr_collision, 
            'crash_rate': round(self.crash_rate / (self.lap_count*len(self.goal_list)) * 100, 3),  
            'ee_path_length': round(ee_traj_length, 3), 
            'joints_path_length': round(joints_path_length, 3), 
            'Avg.Speed': round(avgvel, 3), 
            'Max.Speed': round(maxvel, 3),
            'oneLoop':(time.time() - last) / opt_step_count * 1000, 
            'oneOpt':opt_time_sum / opt_step_count * 1000,
               }
        # 将字典的内容转换为字符串并打印到终端
        log_message = "\n".join(["{}: {}".format(key, value) for key, value in row.items()])
        rospy.loginfo(log_message)

        # self.mpc_control.close()
        self.coll_robot_pub.unregister() 
        self.pub_env_pc.unregister()
        self.pub_robot_link_pc.unregister()
        with open('./SDFcost_Franka/SDFcost_CompareForFranka.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not f.tell():
                writer.writeheader()
            writer.writerow(row)
        self.plot_traj(root_path = './SDFcost_Franka/' , img_name = 'PPV.png')
        print("mpc_close...")



if __name__ == '__main__':


    ik_mSolve = IKSolve() # 多进程的问题 （应该是没有正确的解决 含有糊弄的成分 主要就像要让 IKProc在主进程启动 同时 在spawn之前启动）
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rospy.init_node('pointcloud_publisher_node')
    sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
    sim_params['headless'] = False
    gym_instance = Gym(**sim_params)

    controller = MPCRobotController(gym_instance , ik_mSolve)
    
    controller.run()
