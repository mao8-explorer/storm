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

from typing import List, Optional

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
                    urdf_path='content/assets/urdf/genie_description/A2DWithFixBase.urdf',
                    base_link='base_link',
                    end_link='ee_link'
                )
            )
            self.ik_procs[-1].daemon = True #守护进程 主进程结束 IKProc进程随之结束
            self.ik_procs[-1].start()       



class JointNameMapper:
    """
    将一个来源的关节顺序（src_names）映射到统一的内部顺序（dst_names），
    支持对关节值的快速重排、缺失补全与 fallback 替代。
    """

    def __init__(self, src_names: List[str], dst_names: List[str]):
        """
        Args:
            src_names: 来源系统的关节名顺序（如 Gym、TracIK 等）
            dst_names: 内部标准顺序（如 URDF 控制器所用）
        """
        self.src_names = src_names
        self.dst_names = dst_names
        self.num_dof = len(dst_names)

        # 名称 → 索引映射（src 侧）
        self._src_name_to_idx = {name: i for i, name in enumerate(src_names)}
        # 每个 dst_name 对应的 src_q 中索引，若缺失则为 -1
        self._reorder_indices = [self._src_name_to_idx.get(name, -1) for name in dst_names]

    def forward(
        self,
        src_q: np.ndarray,
        fallback: Optional[np.ndarray] = None,
        default_value: float = 0.0
    ) -> np.ndarray:
        """
        将 src_q 重排为 dst_names 顺序，缺失值自动填补。

        Args:
            src_q: 原始关节值数组，对应 src_names 顺序
            fallback: 缺失关节名时使用的默认值数组（如已有状态）
            default_value: 若未提供 fallback，则使用该默认值补全

        Returns:
            np.ndarray: 重排并补全后的关节数组，对应 dst_names 顺序
        """
        if fallback is not None:
            dst_q = fallback.copy()
        else:
            dst_q = np.full(self.num_dof, default_value, dtype=src_q.dtype)

        for dst_idx, src_idx in enumerate(self._reorder_indices):
            if src_idx != -1:
                dst_q[dst_idx] = src_q[src_idx]
        return dst_q

    def debug_mapping(self):
        """
        打印 src_names → dst_names 的映射关系，用于调试。
        """
        print(f"{'dst_idx':>7s} | {'dst_name':<20s} | {'src_idx':>7s} | {'src_name':<20s}")
        print("-" * 60)
        for dst_idx, (dst_name, src_idx) in enumerate(zip(self.dst_names, self._reorder_indices)):
            src_name = self.src_names[src_idx] if src_idx != -1 else '---'
            src_idx_str = f"{src_idx}" if src_idx != -1 else '  -'
            print(f"{dst_idx:7d} | {dst_name:<20s} | {src_idx_str:7s} | {src_name:<20s}")



class MPCRobotController(GenieEnvBase):
    def __init__(self, gym_instance , ik_mSolve):
        super().__init__(gym_instance = gym_instance)
        self.mpc_control = ReacherTask( self.mpc_config, self.world_description, self.tensor_args )
        self._environment_init()
        self.envpc_filter = FilterPointCloud(self.robot_sim.camObsHandle.cam_pose) #sceneCollisionNet 句柄 现在只是用来获取点云

        # 实验一: 半椭圆跟踪 （验证动态性能）
        self.trac_target_velscale = 0.5
        self.base_height_y = -0.50
        self.base_height_z = 0.70
        self.z_radius = 0.4
        self.y_radius = 0.4
        self.x = 0.60
        z = self.z_radius * np.cos(0.0) + self.base_height_z
        y = self.y_radius * np.sin(0.0) + self.base_height_y
        self.goal_state =  [self.x,z,y]
        

        # 实验二：动态障碍物往复运动
        self.task_leftright = False
        self.coll_dt_scale = 0.01 # up and down
        # self.goal_list = [ # 两个目标点位置
        #     [0.45, 0.190,  0.42],
        #     [0.45, 0.190,  -0.42]]
        self.goal_list = [ # 两个目标点位置
            [0.60, 0.80,   -0.56],
            [0.45, 0.19,  -0.52],
            [0.707,0.50, 0.05]]
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

        # === 统一提取 joint name 列表 ===
        # 来自 URDF 模型内部控制顺序（标准顺序）
        internal_joint_names = self.rollout_fn.dynamics_model.internal_joint_names
        # 来自 Gym 仿真接口的关节顺序
        gym_joint_names = self.robot_sim.joint_names
        # 来自 IK 解算器（可能是部分子集）
        ik_joint_names = list(self.ik_mSolve.ik_procs[0].ik_solver.joint_names)

        self.gym_mapper = JointNameMapper(gym_joint_names, internal_joint_names)
        self.ik_mapper = JointNameMapper(ik_joint_names, internal_joint_names)
        self.interToik_mapper = JointNameMapper(internal_joint_names, ik_joint_names)
        self.interTogym_mapper = JointNameMapper(internal_joint_names, gym_joint_names)


        # update goal_joint_space:
        init_joint_state = self.robot_sim.init_robot_state
        franka_bl_state = np.concatenate([self.gym_mapper.forward(init_joint_state['pos']), self.gym_mapper.forward(init_joint_state['vel'])], axis=0)
        self.mpc_control.update_params(goal_state=franka_bl_state)
        self.g_pos = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        self.g_q = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())

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
            # while self.goal_flagi < self.lap_count * len(self.goal_list):
            while True:
                # 正常循环主体
                self.gym_instance.step()
                self.gym_instance.clear_lines()

                ### ---  感知模块  ---###
                ##----- collision with environment generate pointcloud 6ms -----##
                # step 1. 仿真中获取当前时刻深度图 -（带有 environment & robot 语义的深度图）| numpy cpu
                    # env_time_last = time.time()
                    # self.robot_sim.updateCamImage()
                    # # step 2. 基于深度图 ->仿射变换 -> 点云数据 | numpy cpu
                    # obs.update(self.robot_sim.ImageToPointCloud())
                    # # step 3. 滤除robot点云（语义label） | numpy cpu --> tensor cuda 
                    # self.envpc_filter._update_state(obs) 
                    # # step 4. compute pointcloud to sdf_map 
                    # self.collision_grid = self.mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll. \
                    #                         _opt_compute_dynamic_voxeltosdf(self.envpc_filter.cur_scene_pc, visual = False)
                    # env_time_sum += time.time() - env_time_last

                ### --- 规划模块 ---###
                # monitor ee_pose_gym and update goal_param_mpc
                self.monitorMPCGoalupdate()
                # seed goal to MPC_Policy _ get Command
                t_step += self.sim_dt
                self.current_robot_state = self.robot_sim.get_state(self.env_ptr, self.robot_ptr) # "dict: pos | vel | acc"
                self.current_robot_state['position'] = self.gym_mapper.forward(self.current_robot_state['position'])
                self.current_robot_state['velocity'] = self.gym_mapper.forward(self.current_robot_state['velocity'])                     

                self.goal_ee_transform[:3,3] = self.rollout_fn.goal_ee_pos.cpu().numpy()
                self.goal_ee_transform[:3,:3] = self.rollout_fn.goal_ee_rot.cpu().numpy()
                # 逆解获取请求发布 input_queue
                qinit = self.interToik_mapper.forward(self.current_robot_state['position']) # shape is (7,)
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
                # self.dual_updateGymVisual_GymGoalUpdate()
                self._dynamic_goal_track(t_step)

                self.visual_top_trajs_ingym()
                # Command_Robot_State include keyboard control : SPACE For Pause | ESCAPE For Exit 
                successed = self.robot_sim.command_robot_state(self.interTogym_mapper.forward(q_des), self.interTogym_mapper.forward(qd_des), self.env_ptr, self.robot_ptr)
                if not successed : break 

                # # curr_coll max
                # curr_coll = self.mpc_control.controller.rollout_fn.primitive_collision_cost.current_state_collision
                # if (curr_coll > 0.90).any() : 
                #     self.curr_collision += 1
                #     self.collision_hanppend = True
                #     collision_info = "Collision Count: {}, Collisions: {}".format(self.curr_collision, torch.nonzero(curr_coll > 0.90).flatten().cpu().numpy())
                #     print(collision_info)

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
                        self.rollout_fn.goal_jnq = torch.as_tensor(self.ik_mapper.forward(output[1], q_des), **self.tensor_args).unsqueeze(0) # 1 x n_dof
                        self.jnq_des = output[1]
                        print("------------iksolve")
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


def print_ik_joint_limits(ik_solver):
    
    print(f"IK Solver: {ik_solver.__class__.__name__}")
    names = ik_solver.joint_names
    lower, upper = ik_solver.joint_limits

    header = f"{'Idx':>3s} | {'Joint Name':<20s} | {'Limit Low':>10s} | {'Limit Up':>10s}"
    separator = "-" * len(header)
    print(header)
    print(separator)
    for i, (name, l, u) in enumerate(zip(names, lower, upper)):
        print(f"{i:3d} | {name:<20s} | {l:10.3f} | {u:10.3f}")


if __name__ == '__main__':


    ik_mSolve = IKSolve() # 多进程的问题 （应该是没有正确的解决 含有糊弄的成分 主要就像要让 IKProc在主进程启动 同时 在spawn之前启动）
    ik_single_solver = ik_mSolve.ik_procs[0].ik_solver
    print_ik_joint_limits(ik_single_solver)

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
