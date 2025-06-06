""" Example spawning a robot in gym 
"""
from GenieEnvBase import GenieEnvBase
from storm_examples.multimodal_franka.FilterPointCloud import FilterPointCloud
from storm_examples.multimodal_franka.utils import LimitedQueue , IKProc
from tracikpy import TracIKSolver
import torch
import numpy as np
np.int = int
np.float = float
np.bool = bool


from storm_kit.gym.core import Gym
from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.mpc.task.reacher_task import ReacherTask
import queue
import time

from logger_utils import create_loggers, log_collision_info, log_performance_stats
from joint_name_mapper import JointNameMapper

# 创建所有需要的日志记录器
loggers = create_loggers()

class IKSolve:
    def __init__(self, rollout_fn, internal_joint_names):
        """
        Initialize the IKSolve class, which manages dual TRAC-IK solvers for left and right end-effectors.
        """
        self.maxsize = 5
        self.num_procs = 2  # Number of processes to handle IK computations, addressing output queue empty issues
        self.ik_procs_l = []
        self.ik_procs_r = []

        # ── (1) Construct two TRAC-IK solvers ─────────────────────────
        urdf_path = join_path(get_assets_path(), 'urdf/genie_description/A2DWithFixBase.urdf')
        base_link = "base_link"
        left_ee = "ee_link_l"
        right_ee = "ee_link_r"

        self.l_output_queue = LimitedQueue(self.maxsize)
        for _ in range(self.num_procs):
            # Left IK solver
            self.ik_procs_l.append(
                IKProc(
                    self.l_output_queue,
                    input_queue_maxsize=self.maxsize,
                    urdf_path=urdf_path,
                    base_link=base_link,
                    end_link=left_ee
                )
            )
        self.r_output_queue = LimitedQueue(self.maxsize)
        for _ in range(self.num_procs):
            # Right IK solver
            self.ik_procs_r.append(
                IKProc(
                    self.r_output_queue,
                    input_queue_maxsize=self.maxsize,
                    urdf_path=urdf_path,
                    base_link=base_link,
                    end_link=right_ee
                )
            )
            # # Add solvers to the process list and start them
            # self.ik_procs.extend([self.ik_solver_l, self.ik_solver_r])
        self.rollout_fn = rollout_fn
        self.internal_joint_names = internal_joint_names
        
        self.goal_jnq = None
        self.jnq_mask = None  # shape: (n_dof,)

       # ==== 统计字段 ====
        self.ik_fail_count = 0
        self.l_arm_fail_count = 0
        self.r_arm_fail_count = 0
        self.ik_total_calls = 0

        # 给每个 proc 索引生成一个固定 seed 的 RNG
        self.noise_std = 0.02
        self.rng_left = [np.random.RandomState(seed=100 + i) for i in range(self.num_procs)]
        self.rng_right = [np.random.RandomState(seed=200 + i) for i in range(self.num_procs)]


    def start_solvers(self):
        """
        真正启动 IK 多进程，同时初始化映射器。
        """
        temp_solver_l = None
        for ik_proc in self.ik_procs_l:
            # ─── (A) 用与子进程完全相同的参数，在父进程里新建一个 TracIKSolver 只为读取信息 ───
            if temp_solver_l is None:
                try:
                    temp_solver_l = TracIKSolver(
                        ik_proc.urdf_path,
                        ik_proc.base_link,
                        ik_proc.end_link
                    )
                    # 读取并打印 joint_names 与 joint_limits
                    ik_proc.joint_names = temp_solver_l.joint_names
                    ik_proc.joint_limits = temp_solver_l.joint_limits

                except Exception as e:
                    print(f"[WARNING] 无法在父进程里临时构造 IK solver 来读取关节信息：{e}")

            # ─── (B) 真正启动子进程，子进程会在自己的 run() 中创建 ik_solver ───
            ik_proc.daemon = True
            ik_proc.start()


        temp_solver_r = None
        for ik_proc in self.ik_procs_r:
            # ─── (A) 用与子进程完全相同的参数，在父进程里新建一个 TracIKSolver 只为读取信息 ───
            if temp_solver_r is None:
                try:
                    temp_solver_r = TracIKSolver(
                        ik_proc.urdf_path,
                        ik_proc.base_link,
                        ik_proc.end_link
                    )
                    # 读取并打印 joint_names 与 joint_limits
                    ik_proc.joint_names = temp_solver_r.joint_names
                    ik_proc.joint_limits = temp_solver_r.joint_limits

                except Exception as e:
                    print(f"[WARNING] 无法在父进程里临时构造 IK solver 来读取关节信息：{e}")

            # ─── (B) 真正启动子进程，子进程会在自己的 run() 中创建 ik_solver ───
            ik_proc.daemon = True
            ik_proc.start()

        # ik initialize
        self.l_goal_ee_transform = np.eye(4)
        self.r_goal_ee_transform = np.eye(4)

        # 来自 IK 解算器（可能是部分子集）
        # ik_joint_names = list(self.ik_mSolve.ik_procs[0].ik_solver.joint_names)
        self.ik_joint_names_l = list(self.ik_procs_l[0].joint_names)
        self.ik_joint_names_r = list(self.ik_procs_r[0].joint_names)

        self.ik_mapper_l = JointNameMapper(self.ik_joint_names_l, self.internal_joint_names)
        self.ik_mapper_r = JointNameMapper(self.ik_joint_names_r, self.internal_joint_names)
        self.interToik_mapper_l = JointNameMapper(self.internal_joint_names, self.ik_joint_names_l)
        self.interToik_mapper_r = JointNameMapper(self.internal_joint_names, self.ik_joint_names_r)

        self.mask_indices_l = [i for i in self.ik_mapper_l._src_to_dst_index if i != -1]
        self.mask_indices_r = [i for i in self.ik_mapper_r._src_to_dst_index if i != -1]


    def print_ik_joint_limits(self, ik_solver):
        
        print(f"IK Solver: {ik_solver.__class__.__name__}")
        names = ik_solver.joint_names
        lower, upper = ik_solver.joint_limits

        header = f"{'Idx':>3s} | {'Joint Name':<20s} | {'Limit Low':>10s} | {'Limit Up':>10s}"
        separator = "-" * len(header)
        print(header)
        print(separator)
        for i, (name, l, u) in enumerate(zip(names, lower, upper)):
            print(f"{i:3d} | {name:<20s} | {l:10.3f} | {u:10.3f}")


    def query_dual_ik(self, q_internal, t_step):
        # --- Target poses ---
        self.l_goal_ee_transform[:3, 3] = self.rollout_fn.l_goal_ee_pos.cpu().numpy()
        self.l_goal_ee_transform[:3, :3] = self.rollout_fn.l_goal_ee_rot.cpu().numpy()

        self.r_goal_ee_transform[:3, 3] = self.rollout_fn.r_goal_ee_pos.cpu().numpy()
        self.r_goal_ee_transform[:3, :3] = self.rollout_fn.r_goal_ee_rot.cpu().numpy()

        base_seed_left = self.interToik_mapper_l.forward(q_internal)
        base_seed_right = self.interToik_mapper_r.forward(q_internal)

        # 仅使用一个进程（轮转选择）
        proc_idx = t_step % self.num_procs

        noise_left = self.rng_left[proc_idx].normal(
            loc=0.0, scale=self.noise_std, size=base_seed_left.shape
        )
        noise_right = self.rng_right[proc_idx].normal(
            loc=0.0, scale=self.noise_std, size=base_seed_right.shape
        )

        seed_l = base_seed_left + noise_left
        seed_r = base_seed_right + noise_right

        self.ik_procs_l[proc_idx].ik(self.l_goal_ee_transform, seed_l, ind=t_step)
        self.ik_procs_r[proc_idx].ik(self.r_goal_ee_transform, seed_r, ind=t_step)

    def merge_dual_ik(self, q_des):
        self.ik_total_calls += 1
        try:
            sol_l = self.l_output_queue.get()[1]
            sol_r = self.r_output_queue.get()[1]

            q_merged = q_des.copy()
            mask = np.zeros_like(q_des, dtype=bool)

            if sol_l is not None:
                q_merged = self.ik_mapper_l.forward(sol_l, fallback=q_merged)
                mask[self.mask_indices_l] = True
            else:
                self.l_arm_fail_count += 1

            if sol_r is not None:
                q_merged = self.ik_mapper_r.forward(sol_r, fallback=q_merged)
                mask[self.mask_indices_r] = True
            else:
                self.r_arm_fail_count += 1

            self.goal_jnq = q_merged
            self.jnq_mask = mask

            # if solved:
            #     logger.info(f"IK solved: {', '.join(solved)} arm(s).")

            # else:
            #     logger.warning("IK failed: both arms. Fallback to q_des.")

            # logger.info(f"IK solved arms: {', '.join(solved) if solved else 'None'}")
            # logger.info(f"L mask indices: {self.mask_indices_l}")
            # logger.info(f"R mask indices: {self.mask_indices_r}")
            # loggers['ik'].info(f"IK mask: {mask.astype(int).tolist()}")
            return q_merged

        except queue.Empty:
            self.ik_fail_count += 1
            # loggers['ik'].warning("IK output queue empty. Using fallback q_des")
            self.goal_jnq = q_des
            self.jnq_mask = np.zeros_like(q_des, dtype=bool)
            return q_des



    def get_ik_stats(self):
        stats = {
            "total_merge_calls": self.ik_total_calls,
            # "queue_empty_count": self.ik_fail_count,
            # "l_arm_fail_count": self.l_arm_fail_count,
            # "r_arm_fail_count": self.r_arm_fail_count,
        }
        success_count = self.ik_total_calls - self.ik_fail_count
        stats["ik_success_rate"] = round(success_count / (self.ik_total_calls + 1e-5), 3)
        stats["left_arm_success_rate"] = round(
            (success_count - self.l_arm_fail_count) / (self.ik_total_calls + 1e-5), 3)
        stats["right_arm_success_rate"] = round(
            (success_count - self.r_arm_fail_count) / (self.ik_total_calls + 1e-5), 3)
        return stats



class MPCRobotController(GenieEnvBase):
    def __init__(self, gym_instance):
        super().__init__(gym_instance = gym_instance)
        self.mpc_control = ReacherTask(self.mpc_config, self.world_description, self.tensor_args)
        self._environment_init()
        self.envpc_filter = FilterPointCloud(self.robot_sim.camObsHandle.cam_pose) #sceneCollisionNet 句柄 现在只是用来获取点云

        # 实验一: 半椭圆跟踪 （验证动态性能）
        self.trac_target_velscale = 0.5
        self.base_height_y = -0.30
        self.base_height_z = 0.70
        self.z_radius = 0.40
        self.y_radius = 0.40
        self.x = 0.70
        z = self.z_radius * np.cos(0.0) + self.base_height_z
        y = self.y_radius * np.sin(0.0) + self.base_height_y
        # self.goal_state =  [self.x,z,y]
        self.goal_state_l = [self.x, z, y]
        self.goal_state_r = [self.x, z, -y]


        self.init_coll_pos = [-6.30,0.25,0.0]
        self.dual_update_goal_state()
        self.update_collision_state(self.init_coll_pos)
        self.rollout_fn = self.mpc_control.controller.rollout_fn
        self.l_goal_ee_transform = np.eye(4)
        self.r_goal_ee_transform = np.eye(4)
        # 暂行多进程方案是通过传参的方式 引导ik_proc句柄 保证ik_proc在主进程启动 避免无法共享内存的问题

        # === 统一提取 joint name 列表 ===
        # 来自 URDF 模型内部控制顺序（标准顺序）
        internal_joint_names = self.rollout_fn.dynamics_model.internal_joint_names

        # ik 引导 唤起
        self.ik_mSolve = IKSolve(self.rollout_fn, internal_joint_names)

        # 来自 Gym 仿真接口的关节顺序
        gym_joint_names = self.robot_sim.joint_names
        self.gym_mapper = JointNameMapper(gym_joint_names, internal_joint_names)
        self.interTogym_mapper = JointNameMapper(internal_joint_names, gym_joint_names)
        # self.ik_mapper = JointNameMapper(ik_joint_names, internal_joint_names)
        # self.interToik_mapper = JointNameMapper(internal_joint_names, ik_joint_names)


        self.num_dof = len(internal_joint_names)
        dst_q = np.full(self.num_dof, 0.0)
        jnq_mask = np.zeros_like(dst_q, dtype=bool)
        self.rollout_fn.goal_jnq = torch.tensor(dst_q, **self.tensor_args).unsqueeze(0)
        self.rollout_fn.jnq_mask = torch.tensor(jnq_mask, dtype=torch.bool, device=self.tensor_args['device']).unsqueeze(0)  # shape: (1, n_dof)

        # update goal_joint_space:
        init_joint_state = self.robot_sim.init_robot_state
        franka_bl_state = np.concatenate([self.gym_mapper.forward(init_joint_state['pos']), self.gym_mapper.forward(init_joint_state['vel'])], axis=0)
        self.mpc_control.dual_update_params(goal_state=franka_bl_state)
        self.g_l_pos = np.ravel(self.mpc_control.controller.rollout_fn.l_goal_ee_pos.cpu().numpy())
        self.g_l_quat = np.ravel(self.mpc_control.controller.rollout_fn.l_goal_ee_quat.cpu().numpy())
        self.g_r_pos = np.ravel(self.mpc_control.controller.rollout_fn.r_goal_ee_pos.cpu().numpy())
        self.g_r_quat = np.ravel(self.mpc_control.controller.rollout_fn.r_goal_ee_quat.cpu().numpy())

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
        # self.jnq_des = np.zeros(7)
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
                self.dual_monitorMPCGoalupdate()
                # seed goal to MPC_Policy _ get Command
                t_step += self.sim_dt
                self.current_robot_state = self.robot_sim.get_state(self.env_ptr, self.robot_ptr) # "dict: pos | vel | acc"
                self.current_robot_state['position'] = self.gym_mapper.forward(self.current_robot_state['position'])
                self.current_robot_state['velocity'] = self.gym_mapper.forward(self.current_robot_state['velocity'])                     

                # self.l_goal_ee_transform[:3,3] = self.rollout_fn.l_goal_ee_pos.cpu().numpy()
                # self.l_goal_ee_transform[:3,:3] = self.rollout_fn.l_goal_ee_rot.cpu().numpy()

                # self.r_goal_ee_transform[:3,3] = self.rollout_fn.r_goal_ee_pos.cpu().numpy()
                # self.r_goal_ee_transform[:3,:3] = self.rollout_fn.r_goal_ee_rot.cpu().numpy()
                # 逆解获取请求发布 input_queue
                # qinit = self.interToik_mapper.forward(self.current_robot_state['position']) # shape is (7,)
                # self.ik_mSolve.ik_procs[-1].ik(self.l_goal_ee_transform , qinit , ind = t_step)
                self.ik_mSolve.query_dual_ik(self.current_robot_state['position'], opt_step_count)

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

                # Command_Robot_State include keyboard control : SPACE For Pause | ESCAPE For Exit 
                successed = self.robot_sim.command_robot_state(self.interTogym_mapper.forward(q_des), self.interTogym_mapper.forward(qd_des), self.env_ptr, self.robot_ptr)
                if not successed : break 

                self.dual_dynamic_goal_track(t_step)
                self.visual_top_trajs_ingym()
                # curr_coll max
                # curr_coll = self.mpc_control.controller.rollout_fn.primitive_collision_cost.current_state_collision
                # if (curr_coll > 0.90).any() : 
                #     self.curr_collision += 1
                #     self.collision_hanppend = True
                #     collision_info = "Collision Count: {}, Collisions: {}".format(self.curr_collision, torch.nonzero(curr_coll > 0.90).flatten().cpu().numpy())
                #     print(collision_info)

                if self.rollout_fn.exp_params['cost']['robot_self_collision']['weight'] != 0:
                    curr_coll = self.mpc_control.controller.rollout_fn.robot_self_collision_cost.current_state_collision
                    curr_coll_val = curr_coll.detach().item()

                    log_collision_info(loggers['collision'], curr_coll_val)

                if self.goal_flagi > -1 :
                    self.traj_append()
                    # self.traj_log['collision'].append(curr_coll.cpu().max())
                simVisual_time_sum += time.time() - simVisual_time_last


                self.ik_mSolve.merge_dual_ik(q_des)
                self.rollout_fn.goal_jnq = torch.tensor(self.ik_mSolve.goal_jnq, **self.tensor_args).unsqueeze(0)
                self.rollout_fn.jnq_mask = torch.tensor(self.ik_mSolve.jnq_mask, dtype=torch.bool, device=self.tensor_args['device']).unsqueeze(0)  # shape: (1, n_dof)

        except KeyboardInterrupt:
            loggers['control'].info("KeyboardInterrupt detected. Exiting cleanly...")

        # 最终性能统计
        final_stats = {
            'opt_step_count': opt_step_count,
            'oneLoop': (time.time() - last) / opt_step_count * 1000,
            'OptimizeTime': opt_time_sum / opt_step_count * 1000,
            'EnvTime': env_time_sum / opt_step_count * 1000,
            'SimVisualTime': simVisual_time_sum / opt_step_count * 1000,
        }

        log_performance_stats(loggers['performance'], final_stats)
        # 输出 IK 解算器表现
        ik_stats = controller.ik_mSolve.get_ik_stats()
        log_performance_stats(loggers['performance'], ik_stats)



if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_num_threads(16)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
    sim_params['headless'] = False
    gym_instance = Gym(**sim_params)

    controller = MPCRobotController(gym_instance)
    controller.ik_mSolve.start_solvers()  # 💥在显式时机启动多进程
    controller.run()
