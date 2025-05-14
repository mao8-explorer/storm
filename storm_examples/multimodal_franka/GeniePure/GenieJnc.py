""" Example spawning a robot in gym 
"""
from GenieEnvBase import GenieEnvBase
from storm_examples.multimodal_franka.FilterPointCloud import FilterPointCloud
from storm_examples.multimodal_franka.utils import LimitedQueue , IKProc
import torch
import numpy as np
np.int = int
np.float = float
np.bool = bool


from storm_kit.gym.core import Gym
from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml
from storm_kit.mpc.task.reacher_task import ReacherTask
import queue
import time


class IKSolve:
    def __init__(self):
        self.num_proc = 3
        self.maxsize = 5
        self.output_queue = LimitedQueue(self.maxsize)
        self.ik_procs = []
        for _ in range(self.num_proc):
            self.ik_procs.append(
                IKProc(
                    self.output_queue,
                    input_queue_maxsize=self.maxsize,
                    urdf_path='content/assets/urdf/genie_description/A2DWithFixBase.urdf',
                    base_link='base_link',
                    end_link='ee_link'
                )
            )
            self.ik_procs[-1].daemon = True #守护进程 主进程结束 IKProc进程随之结束
            self.ik_procs[-1].start()       


class MPCRobotController(GenieEnvBase):
    def __init__(self, gym_instance , ik_mSolve):
        super().__init__(gym_instance = gym_instance)
        self.mpc_control = ReacherTask( self.mpc_config, self.world_description, self.tensor_args )
        self._environment_init()
        self.envpc_filter = FilterPointCloud(self.robot_sim.camObsHandle.cam_pose) #sceneCollisionNet 句柄 现在只是用来获取点云
        # 实验一: 半椭圆跟踪 （验证动态性能）
        self.trac_target_velscale = 0.6
        self.base_height_y =0.18
        self.z_radius = 0.50
        self.y_radius = 0.18
        self.x = 0.40
        z = self.z_radius * np.cos(0.0)
        y = self.y_radius * np.sin(0.0) + self.base_height_y
    
        self.goal_state =  [self.x,y,z]

        # 实验二：动态障碍物往复运动
        self.task_leftright = False
        self.coll_dt_scale = 0.01 # up and down
        self.goal_list = [ # 两个目标点位置
            [0.45, 0.190,  0.42],
            [0.45, 0.190,  -0.42]]

        self.coll_movebound_leftright = [-0.40,0.40] # 左右实验的位置边界 [-0.4,0.4]测试一次
        self.coll_movebound_updown = [0.10,0.40] # 上下实验的位置边界

        self.uporient = -1.0
        self.init_coll_pos = [-6.30,0.25,0.0]
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
        self.lap_count = 20
        self.thresh = 0.05 # goal next thresh in Cart


    def run(self):
        self.goal_flagi = -1 # 调控目标点
        t_step = gym_instance.get_sim_time()
        obs = {}
        self.jnq_des = np.zeros(7)
        last = time.time()
        env_time_sum = 0
        opt_step_count = 0 
        simVisual_time_sum = 0
        self.curr_collision = 0
        opt_time_sum = 0

        self.crash_rate = 0.0
        self.collision_hanppend = False

        try:
            while self.goal_flagi < self.lap_count * len(self.goal_list):
                # 正常循环主体
                self.gym_instance.step()
                self.gym_instance.clear_lines()

                ### ---  感知模块  ---###
                ##----- collision with environment generate pointcloud 6ms -----##
                # step 1. 仿真中获取当前时刻深度图 -（带有 environment & robot 语义的深度图）| numpy cpu
                env_time_last = time.time()
                self.robot_sim.updateCamImage()
                # step 2. 基于深度图 ->仿射变换 -> 点云数据 | numpy cpu
                obs.update(self.robot_sim.ImageToPointCloud())
                # step 3. 滤除robot点云（语义label） | numpy cpu --> tensor cuda 
                self.envpc_filter._update_state(obs) 
                # step 4. compute pointcloud to sdf_map 
                self.collision_grid = self.mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll. \
                                        _opt_compute_dynamic_voxeltosdf(self.envpc_filter.cur_scene_pc, visual = False)
                env_time_sum += time.time() - env_time_last

                ### --- 规划模块 ---###
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
                opt_step_count += 1
                command = self.mpc_control.get_command(self.current_robot_state)
                opt_time_sum += time.time() - opt_time_last
                # get position command:
                self.command = command

                ### 仿真可视化处理模块 ###
                # command = self.current_robot_state
                simVisual_time_last = time.time()
                q_des ,qd_des ,qdd_des = command['position'] ,command['velocity'] , command['acceleration']
                self.curr_state_tensor = torch.as_tensor(np.hstack((q_des,qd_des,qdd_des)), **self.tensor_args).unsqueeze(0) # "1 x 3*n_dof"
                # trans ee_pose in robot_coordinate to world coordinate
                self.updateGymVisual_GymGoalUpdate()
                self._dynamic_goal_track(t_step)

                self.visual_top_trajs_ingym()
                # Command_Robot_State include keyboard control : SPACE For Pause | ESCAPE For Exit 
                successed = self.robot_sim.command_robot_state(q_des, qd_des, self.env_ptr, self.robot_ptr)
                if not successed : break 

                # curr_coll max
                curr_coll = self.mpc_control.controller.rollout_fn.primitive_collision_cost.current_state_collision
                if (curr_coll > 0.90).any() : 
                    self.curr_collision += 1
                    self.collision_hanppend = True
                    collision_info = "Collision Count: {}, Collisions: {}".format(self.curr_collision, torch.nonzero(curr_coll > 0.90).flatten().cpu().numpy())
                    print(collision_info)

                if self.task_leftright:
                    self._dynamic_object_moveDesign_leftright()
                else :
                    self._dynamic_object_moveDesign_updown()

                if self.goal_flagi > -1 :
                    self.traj_append()
                    # self.traj_log['collision'].append(curr_coll.cpu().max())
                simVisual_time_sum += time.time() - simVisual_time_last


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
            print("KeyboardInterrupt detected. Exiting cleanly...")


        # avgvel, maxvel, ee_traj_length, joints_path_length = self.ee_vel_evaluate()

        row = {
            'opt_step_count': opt_step_count, 
            # 'collison_count':self.curr_collision, 
            # 'crash_rate': round(self.crash_rate / (self.lap_count*len(self.goal_list)) * 100, 3),  
            # 'ee_path_length': round(ee_traj_length, 3), 
            # 'joints_path_length': round(joints_path_length, 3), 
            # 'Avg.Speed': round(avgvel, 3), 
            # 'Max.Speed': round(maxvel, 3),
            'oneLoop':(time.time() - last) / opt_step_count * 1000, 
            'OptimizeTime':opt_time_sum / opt_step_count * 1000,
            'EnvTime':env_time_sum / opt_step_count * 1000,
            'SimVisualTime':simVisual_time_sum / opt_step_count * 1000,
               }
        # 将字典的内容转换为字符串并打印到终端
        log_message = "\n".join(["{}: {}".format(key, value) for key, value in row.items()])
        print("[INFO]", log_message)

        # self.mpc_control.close()
        # self.plot_traj(root_path = './SDFcost_Franka/' , img_name = 'PPV.png')
        print("mpc_close...")



if __name__ == '__main__':


    ik_mSolve = IKSolve() # 多进程的问题 （应该是没有正确的解决 含有糊弄的成分 主要就像要让 IKProc在主进程启动 同时 在spawn之前启动）
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
    sim_params['headless'] = False
    gym_instance = Gym(**sim_params)

    controller = MPCRobotController(gym_instance , ik_mSolve)
    
    controller.run()
