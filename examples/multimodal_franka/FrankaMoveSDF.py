""" Example spawning a robot in gym 

SDF collision design
"""

from FrankaEnvBase import FrankaEnvBase
from FilterPointCloud import FilterPointCloud
from utils import LimitedQueue , IKProc
import torch
import numpy as np
from storm_kit.gym.core import Gym
from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml
import rospy
import queue
import time

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
                ))
            self.ik_procs[-1].daemon = True #守护进程 主进程结束 IKProc进程随之结束
            self.ik_procs[-1].start()       

        # self.goal_list = [
        #         [0.10,0.30,-0.65],
        #         [0.10,0.30,0.65],
        #         [-x,y,z],
        #         [0.10,0.30,0.65],
        #         [0.10,0.30,-0.65],
        #         [-x,y,-z],]
        # self.goal_list = [
        #      [x,y,-z],
        #      [x,y,z],]

class MPCRobotController(FrankaEnvBase):
    def __init__(self, gym_instance , ik_mSolve):
        super().__init__(gym_instance = gym_instance)
        self._environment_init()
        self.envpc_filter = FilterPointCloud(self.robot_sim.camObsHandle.cam_pose) #sceneCollisionNet 句柄 现在只是用来获取点云
        self.coll_dt_scale = 0.015 # left and right
        self.coll_movebound_leftright = [-0.30,0.30] # 左右实验的位置边界
        # self.coll_dt_scale = 0.01 # up and down
        # self.coll_movebound_updown = [0.48,0.85] # 上下实验的位置边界
        self.uporient = -1.0
        self.thresh = 0.05 # goal next thresh in Cart
        self.goal_list = [ # 两个目标点位置
             [0.20,0.30,-0.65],
             [0.20,0.30,0.65]]
        self.goal_state = self.goal_list[0]
        self.update_goal_state()
        self.update_collision_state([0.46 ,0.50,0.0])
        self.rollout_fn = self.mpc_control.controller.rollout_fn
        self.goal_ee_transform = np.eye(4)
        # 暂行多进程方案是通过传参的方式 引导ik_proc句柄 保证ik_proc在主进程启动 避免无法共享内存的问题
        self.ik_mSolve = ik_mSolve

    def run(self):
        self.goal_flagi = 0 # 调控目标点
        sim_dt = self.mpc_control.exp_params['control_dt']
        t_step = gym_instance.get_sim_time()
        obs = {}
        lap_count = 8 # 跑5轮次
        self.jnq_des = np.zeros(7)
        last = time.time()
        # 指标性元素
        opt_step_count = 0 
        opt_time_sum = 0
        self.curr_collision = 0
        while not rospy.is_shutdown() and \
            self.goal_flagi / len(self.goal_list) != lap_count:
            try:
                opt_step_count += 1
                t_step += sim_dt
                self.gym_instance.step()
                self.gym_instance.clear_lines()
                # generate pointcloud 6ms
                self.robot_sim.updateCamImage()
                obs.update(self.robot_sim.ImageToPointCloud()) #耗时大！
                self.envpc_filter._update_state(obs) 
                # compute pointcloud to sdf_map 4.5ms
                self.collision_grid = self.mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll. \
                                     _opt_compute_dynamic_voxeltosdf(self.envpc_filter.cur_scene_pc, visual = True)
                # monitor ee_pose_gym and update goal_param_mpc
                self.monitorMPCGoalupdate()
                # seed goal to MPC_Policy _ get Command
                self.current_robot_state = self.robot_sim.get_state(self.env_ptr, self.robot_ptr) # "dict: pos | vel | acc"
                # 逆解获取请求发布 input_queue
                qinit = self.current_robot_state['position'] # shape is (7,)
                self.goal_ee_transform[:3,3] = self.rollout_fn.goal_ee_pos.cpu().numpy()
                self.goal_ee_transform[:3,:3] = self.rollout_fn.goal_ee_rot.cpu().numpy()
                self.ik_mSolve.ik_procs[-1].ik(self.goal_ee_transform , qinit , ind = t_step)
                opt_time_last = time.time()
                command = self.mpc_control.get_command(t_step, self.current_robot_state, control_dt=sim_dt, WAIT=True)
                opt_time_sum += time.time() - opt_time_last
                # get position command:
                self.command = command
                q_des ,qd_des ,qdd_des = command['position'] ,command['velocity'] , command['acceleration']
                self.curr_state_tensor = torch.as_tensor(np.hstack((q_des,qd_des,qdd_des)), **self.tensor_args).unsqueeze(0) # "1 x 3*n_dof"
                # trans ee_pose in robot_coordinate to world coordinate
                self.updateGymVisual_GymGoalUpdate()
                self.updateRosMsg(visual_gradient=False)
                # Command_Robot_State include keyboard control : SPACE For Pause | ESCAPE For Exit 
                successed = self.robot_sim.command_robot_state(q_des, qd_des, self.env_ptr, self.robot_ptr)
                if not successed : break 

                curr_coll = self.mpc_control.controller.rollout_fn.primitive_collision_cost.current_state_collision
                if (curr_coll > 0.95).any() : 
                    self.curr_collision += 1
                    collision_info = "Collision Count: {}, Collisions: {}".format(self.curr_collision, torch.nonzero(curr_coll > 0.95).flatten().cpu().numpy())
                    rospy.logwarn(collision_info)
                # self._dynamic_object_moveDesign_updown()
                self._dynamic_object_moveDesign_leftright()
                self.traj_append()
                # 逆解获取查询 output_queue
                try :
                    output = self.ik_mSolve.output_queue.get()
                    if output[1] is not None: # 无解
                        self.rollout_fn.goal_jnq = torch.as_tensor(output[1], **self.tensor_args).unsqueeze(0) # 1 x n_dof
                        self.jnq_des = output[1]
                    else : 
                        self.rollout_fn.goal_jnq = None
                        self.jnq_des = np.zeros(7)
                        rospy.logwarn("warning: no iksolve")
                except queue.Empty:
                    "针对 output_queue队列为空的问题 会出现queue.Empty的情况发生"
                    continue

            except KeyboardInterrupt:
                rospy.logwarn('Closing')
        rospy.loginfo("whole_time: {}, opt_step_count: {}, collison_count: {}, "
                      "oneLoop: {}, oneOpt: {}".format(time.time() - last, opt_step_count, self.curr_collision, 
                                                      (time.time() - last) / opt_step_count * 1000, 
                                                       opt_time_sum / opt_step_count * 1000))
        # self.mpc_control.close()
        self.coll_robot_pub.unregister() 
        self.pub_env_pc.unregister()
        self.pub_robot_link_pc.unregister()
        self.plot_traj()
        rospy.logwarn("mpc_close...")
        
if __name__ == '__main__':


    ik_mSolve = IKSolve() # 多进程的问题 （应该是没有正确的解决 含有糊弄的成分 主要就像要让 IKProc在主进程启动 同时 在spawn之前启动）
    torch.multiprocessing.set_start_method('spawn', force=True)
    # torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rospy.init_node('pointcloud_publisher_node')
    sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
    sim_params['headless'] = False
    gym_instance = Gym(**sim_params)
    controller = MPCRobotController(gym_instance , ik_mSolve)    
    controller.run()
