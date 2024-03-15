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
from storm_kit.mpc.task.reacher_task import ReacherTask
import rospy
import queue
import time
import pickle
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
        # Initialize the MPC control
        self.mpc_control = ReacherTask( self.mpc_config, self.world_description, self.tensor_args )
        self._environment_init()
        self.envpc_filter = FilterPointCloud(self.robot_sim.camObsHandle.cam_pose) #sceneCollisionNet 句柄 现在只是用来获取点云
        self.task_leftright = False
        if self.task_leftright:
            self.coll_dt_scale = 0.015 # left and right 0.02测试一次
            self.coll_movebound_leftright = [-0.40,0.40] # 左右实验的位置边界 [-0.4,0.4]测试一次
            self.goal_list = [ # 两个目标点位置
                [0.25,0.40,  0.65],
                [0.20,0.40, -0.65]]
        else:
            self.coll_dt_scale = 0.015 # up and down
            self.coll_movebound_updown = [0.40,0.80] # 上下实验的位置边界
            self.goal_list = [ # 两个目标点位置
                [0.20,0.35,  0.65],
                [0.20,0.35, -0.65]]
        self.uporient = -1.0
        self.init_coll_pos = [0.40,0.60,-0.20]
        self.goal_state = self.goal_list[-1]
        self.update_goal_state()
        self.update_collision_state(self.init_coll_pos)
        self.rollout_fn = self.mpc_control.controller.rollout_fn
        self.goal_ee_transform = np.eye(4)
        # 暂行多进程方案是通过传参的方式 引导ik_proc句柄 保证ik_proc在主进程启动 避免无法共享内存的问题
        self.ik_mSolve = ik_mSolve
        #  visual 控件
        self.gradient_visual_rviz = False
        self.pointcloud_visual_rviz = False
        self.fieldnames = ['whole_time', 'opt_step_count', 'collision_count', 'crash_rate', 
                      'ee_path_length', 'joints_path_length', 
                      'Avg.Speed', 'Max.Speed',
                      'goal_w', 'collision_w',
                      'oneLoop','oneOpt'] 

        self.sim_dt = self.mpc_control.exp_params['control_dt']
        self.lap_count = 10
        self.thresh = 0.05 # goal next thresh in Cart

    def reinit(self, bounds):
        if self.task_leftright:
            self.coll_movebound_leftright = bounds
        self.init_coll_pos = [0.40,0.60,-0.20]
        self.goal_state = self.goal_list[-1]
        self.update_goal_state()
        self.update_collision_state(self.init_coll_pos)
        self.traj_log = {'position':[], 'velocity':[], 'acc':[] , 'des':[] , 
                         'weights':[], 'collision': [], 'ee_pos':[],'run_message':[],
                         'cart_goal_pos':[], 'thresh_index':[]}
        

    def run(self):
        self.goal_flagi = -1 # 调控目标点
        t_step = gym_instance.get_sim_time()
        obs = {}
        self.jnq_des = np.zeros(7)
        last = time.time()
        # 指标性元素
        opt_step_count = 0 
        opt_time_sum = 0
        self.curr_collision = 0
        self.crash_rate = 0.0
        self.collision_hanppend = False
        while not rospy.is_shutdown() and \
            self.goal_flagi / len(self.goal_list) != self.lap_count and \
            opt_step_count < 3000 :
            try:
                t_step += self.sim_dt
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
                command = self.mpc_control.get_command(self.current_robot_state)
                opt_time_sum += time.time() - opt_time_last
                # get position command:
                self.command = command
                q_des ,qd_des ,qdd_des = command['position'] ,command['velocity'] , command['acceleration']
                self.curr_state_tensor = torch.as_tensor(np.hstack((q_des,qd_des,qdd_des)), **self.tensor_args).unsqueeze(0) # "1 x 3*n_dof"
                # trans ee_pose in robot_coordinate to world coordinate
                self.updateGymVisual_GymGoalUpdate()
                # self.visual_top_trajs_ingym()
                # self.updateRosMsg(visual_gradient = self.gradient_visual_rviz)
                # Command_Robot_State include keyboard control : SPACE For Pause | ESCAPE For Exit 
                successed = self.robot_sim.command_robot_state(q_des, qd_des, self.env_ptr, self.robot_ptr)
                if not successed : break 

                # curr_coll max
                curr_coll = self.mpc_control.controller.rollout_fn.primitive_collision_cost.current_state_collision
                if (curr_coll > 0.90).any() : 
                    self.curr_collision += 1
                    self.collision_hanppend = True
                    collision_info = "Collision Count: {}, Collisions: {}".format(self.curr_collision, torch.nonzero(curr_coll > 0.90).flatten().cpu().numpy())
                    rospy.logwarn(collision_info)
                if self.task_leftright:
                    self._dynamic_object_moveDesign_leftright()
                else :
                    self._dynamic_object_moveDesign_updown()

                if self.goal_flagi > -1 :
                    self.traj_append()
                    opt_step_count += 1
                    self.traj_log['collision'].append(curr_coll.cpu().max())

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

        avgvel, maxvel, ee_traj_length, joints_path_length = self.ee_vel_evaluate()

        row = {
            'whole_time': round(time.time() - last, 3),
            'opt_step_count': opt_step_count, 
            'collision_count':self.curr_collision, 
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

        return row


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
    FrankaController = MPCRobotController(gym_instance , ik_mSolve)    

    def run_experiment(bounds, csv_filename):
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FrankaController.fieldnames)
            if not f.tell():
                writer.writeheader()

            for g_w in np.arange(10, 80.1, 10):
                for coll_w in np.arange(80, 250, 20):
                    # if g_w not in [10, 20, 50, 80] or coll_w not in [80, 100, 150, 200, 250]:

                    FrankaController.rollout_fn.dist_cost.weight = torch.tensor(g_w, **FrankaController.tensor_args)
                    FrankaController.rollout_fn.primitive_collision_cost.weight = torch.tensor(coll_w, **FrankaController.tensor_args)

                    results = []
                    for i in range(4):
                        FrankaController.reinit(bounds=bounds)
                        row  = FrankaController.run()
                        
                        row['goal_w'] = g_w 
                        row['collision_w'] = coll_w

                        writer.writerow(row)
                        f.flush()  # 刷新缓冲
                        results.append(row)
                        if row['opt_step_count'] == 3000:  break
                        img_name='P_'+str(g_w)+'_'+ str(coll_w)+'_'+str(i)+'_leftright_'
                        FrankaController.plot_traj(root_path='./SDFcost_Franka/temp/', img_name = img_name, plot_traj_multimodal=False)
                        # with open(img_name+'.pkl', 'wb') as fpkl:
                        #      pickle.dump(FrankaController.traj_log, fpkl)

                    if len(results) > 1: 
                        params = {key: [] for key in row}
                        for result in results:
                            for key in result:
                                params[key].append(result[key])
                        averages = {key: round(np.mean(values), 3) for key, values in params.items()}
                        for key in params:
                            q1 = np.percentile(params[key], 25)
                            q3 = np.percentile(params[key], 75)
                            iqr = q3 - q1
                            outlier_range = 1.5 * iqr
                            params[key] = [value for value in params[key] if (q1 - outlier_range) <= value <= (q3 + outlier_range)]

                        averages_no_outliers = {key: round(np.mean(values), 3) if values else 'N/A' for key, values in params.items()}
                        writer.writerow(averages) # 均值保存
                        writer.writerow(averages_no_outliers) # 箱线图数据
                        f.flush()  # 刷新缓冲


    # bounds = [-0.30, 0.30]
    # run_experiment(bounds, './SDFcost_Franka/SDFcost_FrankaScatter_bound0.03_vel0.02.csv')


    bounds = [-0.40, 0.40]
    run_experiment(bounds, './SDFcost_Franka/SDFcost_FrankaScatter_bound0.04.csv')
