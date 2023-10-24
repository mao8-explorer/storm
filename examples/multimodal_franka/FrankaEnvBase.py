
import copy
from isaacgym import gymapi
import torch
import trimesh.transformations as tra
import yaml
import time
import numpy as np
from storm_kit.gym.core import  World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_gym_configs_path, join_path, get_assets_path
from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
import matplotlib.pyplot as plt
import rospy
from ros_visual import RosVisualBase

np.set_printoptions(precision=2)

class FrankaEnvBase(RosVisualBase):
    def __init__(self, gym_instance):
        super().__init__()
        self.mpc_config = 'franka_reacher.yml'
        self.world_description = 'collision_primitives_3d.yml'
        self.gym_instance = gym_instance
        self.device = torch.device('cuda', 0)
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}
        self.gym = gym_instance.gym
        self.sim = gym_instance.sim
        self.env_ptr = gym_instance.env_list[0]
        self.viewer = gym_instance.viewer
        self.collision_grid = None
        self.curr_state_tensor = None
        self.traj_log = {'position':[], 'velocity':[], 'acc':[] , 'des':[] , 'weights':[]}

    def _environment_init(self):
        self._initialize_robot_simulation() # robot_sim 
        self._initialize_world_and_camera() # world_instance
        self._initialize_mpc_control() # mpc_control 
        self._initialize_env_objects() # 设置 gym 可操作物 handle
        self._init_point_transform() # use for trans trajs_pos in robotCoordinate to world coordinate
    
    def _initialize_robot_simulation(self):
        """
        contains a generic robot class
            that can load a robot asset into sim and 
            gives access to robot's state and receive command_of_policy.
        """
        # Initialize the robot simulation
        robot_yml = join_path(get_gym_configs_path(), 'franka.yml')
        with open(robot_yml) as file:
            robot_params = yaml.load(file, Loader=yaml.FullLoader)
        sim_params = robot_params['sim_params']  # get from -->'/home/zm/MotionPolicyNetworks/storm_ws/src/storm/content/configs/gym/franka.yml'
        sim_params['asset_root'] = get_assets_path()
        sim_params['collision_model']=None
        robot_pose = sim_params['robot_pose']  # robot_pose: [0, 0.0, 0, -0.707107, 0.0, 0.0, 0.707107]

        # create robot simulation: contains a generic robot class that can load a robot asset into sim and gives access to robot's state and control.
        self.robot_sim = RobotSim(
            gym_instance=self.gym_instance.gym, 
            sim_instance=self.gym_instance.sim,
            env_instance = self.gym_instance.env_list[0],
            viewer = self.gym_instance.viewer,
            **sim_params,
            device=torch.device('cuda', 0) )
        # create gym environment: 
        self.robot_ptr = self.robot_sim.spawn_robot(self.gym_instance.env_list[0], robot_pose, coll_id=2)
        # ensure world_robot transform
        self.w_T_r = self.robot_sim.spawn_robot_pose


    def _initialize_world_and_camera(self):
        """
        Initialize the world instance and camera_pose
        加载静态模型 包括桌面 球体 方块        
        """
        # spawn camera:
        external_transform = tra.euler_matrix(0, 0, np.pi / 9).dot(
                tra.euler_matrix(0, np.pi / 2, 0)
        )
        external_transform[0, 3] = 2.0
        external_transform[1, 3] = 1.0
        external_transform[2, 3] = 0
        self.robot_sim.spawn_camera(self.gym_instance.env_list[0], 60, 640, 480, external_transform)

        # world initialization
        world_yml = join_path(get_gym_configs_path(), self.world_description)
        with open(world_yml) as file:
            world_params = yaml.load(file, Loader=yaml.FullLoader)  # world_mode
        self.world_instance = World(
            self.gym_instance.gym,
            self.gym_instance.sim,
            self.gym_instance.env_list[0],
            world_params,
            w_T_r=self.w_T_r)

        self.gym_instance.build_sphere_geom()

    def _initialize_mpc_control(self):
        # update goal_joint_space:
        franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.mpc_control.update_params(goal_state=franka_bl_state)
        self.g_pos = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        self.g_q = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())

    def _initialize_env_objects(self):
             
        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        object_pose.r = gymapi.Quat(0.801, 0.598, 0.0, 0)
        obj_asset_root = get_assets_path()

        target_asset_file = "urdf/mug/movable_mug.urdf"
        collision_move_asset_file = "urdf/mug/movable_collision_test.urdf"
        current_asset_file = "urdf/mug/mug.urdf"
  
        target_object = self.world_instance.spawn_object(target_asset_file, obj_asset_root, object_pose,
                                                    name='ee_target_object')
        collision_obj = self.world_instance.spawn_collision_obj(collision_move_asset_file, obj_asset_root, object_pose,
                                                    name='collision_move_test')    
        current_ee_obj = self.world_instance.spawn_object(current_asset_file, obj_asset_root, object_pose, 
                                                name='ee_current_as_mug')    

        self.collision_base_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, collision_obj, 0)
        self.collision_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, collision_obj, 6)

        self.target_base_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, target_object, 0)
        self.target_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, target_object, 6)

        self.ee_base_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, current_ee_obj, 0)
      
        # set different color for three type objects RGB       
        self.gym.set_rigid_body_color(self.env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.8, 0.1, 0.1))
        self.gym.set_rigid_body_color(self.env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.8, 0.1, 0.1))

        self.gym.set_rigid_body_color(self.env_ptr, collision_obj, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.80, 0.42, 0.13))
        self.gym.set_rigid_body_color(self.env_ptr, collision_obj, 6, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.80, 0.42, 0.13))
                
        self.gym.set_rigid_body_color(self.env_ptr, current_ee_obj, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.0, 0.8, 0.0))

        # reset initial_position of target_object and collision_move_object
        object_pose.p = gymapi.Vec3(0.280,0.469,0.118)
        object_pose.r = gymapi.Quat(0.392,0.608,-0.535,0.436)
        self.gym.set_rigid_transform(self.env_ptr, self.target_base_handle, object_pose)

        object_pose.p = gymapi.Vec3(0.700 , 0.16,  0.704)
        object_pose.r = gymapi.Quat(0.278,0.668,-0.604,0.334)
        self.gym.set_rigid_transform(self.env_ptr, self.collision_base_handle, object_pose)


    def _init_point_transform(self):
        w_T_robot = torch.eye(4)
        quat = torch.tensor([self.w_T_r.r.w, self.w_T_r.r.x, self.w_T_r.r.y, self.w_T_r.r.z]).unsqueeze(0)
        rot = quaternion_to_matrix(quat)
        w_T_robot[0,3] = self.w_T_r.p.x
        w_T_robot[1,3] = self.w_T_r.p.y
        w_T_robot[2,3] = self.w_T_r.p.z
        w_T_robot[:3,:3] = rot[0]
        self.w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3,3].unsqueeze(0),
                                            rot=w_T_robot[0:3,0:3].unsqueeze(0),
                                            tensor_args=self.tensor_args)    
 

    def monitorMPCGoalupdate(self):
        """
        谁控制了target_body_handle 谁控制了MPC_Policy_Goal 不管是通过Gym还是通过代码的方式 都可以
        检测 gym目标变化情况, 一旦变化， 更新MPC 目标
        """
        pose_w = copy.deepcopy(self.world_instance.get_pose(self.target_body_handle))
        # self.gym.set_rigid_transform(self.env_ptr, self.target_base_handle, pose_w)
        pose = self.w_T_r.inverse() * pose_w #将world坐标系下的目标点转到robot坐标系下
        if (np.linalg.norm(self.g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (
                np.linalg.norm(self.g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z])) > 0.0)):
            self.g_pos[0] = pose.p.x
            self.g_pos[1] = pose.p.y
            self.g_pos[2] = pose.p.z
            self.g_q[1] = pose.r.x   
            self.g_q[2] = pose.r.y
            self.g_q[3] = pose.r.z
            self.g_q[0] = pose.r.w
            self.mpc_control.update_params(goal_ee_pos=self.g_pos,goal_ee_quat=self.g_q)

    def update_goal_state(self):
        # target_base 与 body的讨论见草稿 10.07
        goal_state = self.goal_state
        world_T_body_des = copy.deepcopy(self.world_instance.get_pose(self.target_body_handle))
        body_T_world = copy.deepcopy(world_T_body_des).inverse()
        world_T_base = copy.deepcopy(self.world_instance.get_pose(self.target_base_handle))
        world_T_body_des.p = gymapi.Vec3(goal_state[0],goal_state[1],goal_state[2])
        set_world_T_base = world_T_body_des * body_T_world * world_T_base
        self.gym.set_rigid_transform(self.env_ptr, self.target_base_handle, set_world_T_base)

    def updateGymVisual_GymGoalUpdate(self , end_trajvisual = False):
               
        # trans ee_pose in robot_coordinate to world coordinate
        ee_pose = gymapi.Transform()
        pose_state = self.mpc_control.controller.rollout_fn.get_ee_pose(self.curr_state_tensor)
        cur_e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
        cur_e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
        ee_pose.p = gymapi.Vec3(cur_e_pos[0], cur_e_pos[1], cur_e_pos[2])
        ee_pose.r = gymapi.Quat(cur_e_quat[1], cur_e_quat[2], cur_e_quat[3], cur_e_quat[0])
        ee_pose = self.w_T_r * ee_pose
        self.gym.set_rigid_transform(self.env_ptr, self.ee_base_handle, ee_pose)

        # if current_ee_pose in goal_pose thresh ,update to next goal_pose
        if (np.linalg.norm(np.array(self.g_pos - cur_e_pos)) < self.thresh):
            self.goal_flagi += 1
            self.goal_state = self.goal_list[(self.goal_flagi+1) % len(self.goal_list)]
            self.update_goal_state()
            log_message = "next goal: {}, lap_count: {}, collision_count: {}".format(self.goal_flagi, self.goal_flagi / len(self.goal_list), self.curr_collision)
            rospy.loginfo(log_message)
            if self.goal_flagi %  ( 2*len(self.goal_list) )== 1 : 
                self.traj_log = {'position':[], 'velocity':[], 'acc':[] , 'des':[] , 'weights':[]}
                print("置零")

        
        if end_trajvisual :
            self.visual_top_trajs_ingym_multimodal()
            # self.visual_top_trajs_ingym()
    
    def visual_top_trajs_ingym(self):

        top_trajs = self.mpc_control.top_trajs  # .numpy()
        n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
        w_pts = self.w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)

        top_trajs = w_pts.cpu().numpy()
        color = np.array([0.0, 1.0, 0.0])
        for k in range(top_trajs.shape[0]):
            pts = top_trajs[k, :, :]
            color[0] = float(k) / float(top_trajs.shape[0])
            color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
            self.gym_instance.draw_lines(pts, color=color)


    def visual_top_trajs_ingym_multimodal(self):
        # gym_instance.clear_lines() 放在while初始，在订阅点云前清屏
        # 0 -1 mean_action 
        # 1  0 sensi_best_action
        # 2  1 greedy_best_action
        # 3  2 sensi_mean
        # 4  3 greedy_mean
        greedy_top_trajs = self.mpc_control.controller.greedy_top_trajs
        sensi_top_trajs = self.mpc_control.controller.sensi_top_trajs
        n_p, n_t = greedy_top_trajs.shape[0], greedy_top_trajs.shape[1]
        greedy_w_pts = self.w_robot_coord.transform_point(greedy_top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)
        sensi_w_pts = self.w_robot_coord.transform_point(sensi_top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)
        self.draw_lines(greedy_w_pts,color = np.array([1.0, 0.0, 0.0]) )
        self.draw_lines(sensi_w_pts,color = np.array([0.0, 1.0, 0.0]) )
        visual_trajectory = self.mpc_control.controller.rollout_fn.visual_trajectory  # .numpy() # 5 * 30 *3
        n_p, n_t = visual_trajectory.shape[0], visual_trajectory.shape[1]
        w_pts = self.w_robot_coord.transform_point(visual_trajectory.view(n_p * n_t, 3)).view(n_p, n_t, 3)       
        top_trajs = w_pts.cpu().numpy()
        self.gym_instance.draw_spheres(top_trajs)

        """
        暂停设计: 启动pause查看静止的轨迹分布
        gym可视化内容 : 
            红色的greedy策略规划的末端轨迹
            绿色的sensi 策略规划的末端轨迹
            使用定制的蓝球 可视化 mean策略规划的末端轨迹
        rviz可视化内容 :
            整个robot在不同侧策略下的joint_trajectory
        """
        while not self.robot_sim.playing:
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "PAUSE" and evt.value > 0:
                    self.robot_sim.playing = not self.robot_sim.playing
            time.sleep(0.05) #
            self.gym_instance.step()
            # publish trajectory?
            mean_joint_visual_trajectory = self.mpc_control.controller.rollout_fn.mean_joint_visual_trajectory.cpu().numpy() # 30 * 7
            sensi_joint_visual_trajectory = self.mpc_control.controller.rollout_fn.sensi_joint_visual_trajectory.cpu().numpy() 
            greedy_joint_visual_trajectory = self.mpc_control.controller.rollout_fn.greedy_joint_visual_trajectory.cpu().numpy() 
            self.pub_multi_joint_trajectory(mean_joint_visual_trajectory , sensi_joint_visual_trajectory, greedy_joint_visual_trajectory)


    def draw_lines(self,top_trajs,color=None):
        top_trajs = top_trajs.cpu().numpy()
        
        for k in range(top_trajs.shape[0]):
            pts = top_trajs[k, :, :]
            # color[0] = float(k) / float(top_trajs.shape[0])
            # color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
            self.gym_instance.draw_lines(pts, color=color)

    """
    下面三个是 三个独立的测试动态特征的实验 
    _dynamic_object_moveDesign_updown : 障碍物上下移动
    _dynamic_object_moveDesign_leftright : 障碍物左右移动

    _dynamic_goal_track : 目标点做半圆周运动 测试跟踪性能
    _dynamic_goal_track + _dynamic_object_moveDesign_leftright_track : 目标点半圆周运动 , 障碍物左右移动 , 测试轨迹跟踪过程中的动态避障能力
    """
     
    def _dynamic_object_moveDesign_updown(self):
        # Update velocity vector based on move bounds and current pose
        collision_T = copy.deepcopy(self.world_instance.get_pose(self.collision_body_handle))
        if collision_T.p.y < self.coll_movebound_updown[0] or \
           collision_T.p.y > self.coll_movebound_updown[1] :
            self.uporient *= -1
        # Move the object based on the velocity vector
        collision_T.p.y += self.uporient * self.coll_dt_scale
        self.gym.set_rigid_transform(
            self.env_ptr, self.collision_base_handle, collision_T)
        
    def _dynamic_object_moveDesign_leftright(self):
        # Update velocity vector based on move bounds and current pose
        collision_T = copy.deepcopy(self.world_instance.get_pose(self.collision_body_handle))
        if collision_T.p.z < self.coll_movebound_leftright[0] or \
           collision_T.p.z > self.coll_movebound_leftright[1] :
            self.uporient *= -1
        # Move the object based on the velocity vector
        collision_T.p.z += self.uporient * self.coll_dt_scale
        self.gym.set_rigid_transform(
            self.env_ptr, self.collision_base_handle, collision_T)

    def _dynamic_goal_track(self, t_step):
        """
        轨迹跟踪任务 定制化的函数
        功能 : 目标点做半圆周运动
        """
        # trans ee_pose in robot_coordinate to world coordinate
        ee_pose = gymapi.Transform()
        pose_state = self.mpc_control.controller.rollout_fn.get_ee_pose(self.curr_state_tensor)
        cur_e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
        cur_e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
        ee_pose.p = gymapi.Vec3(cur_e_pos[0], cur_e_pos[1], cur_e_pos[2])
        ee_pose.r = gymapi.Quat(cur_e_quat[1], cur_e_quat[2], cur_e_quat[3], cur_e_quat[0])
        ee_pose = self.w_T_r * ee_pose
        self.gym.set_rigid_transform(self.env_ptr, self.ee_base_handle, ee_pose)
        t_step = t_step * self.trac_target_velscale
        z = self.z_radius * np.cos(t_step)
        y = self.y_radius * np.abs(np.sin(t_step)) + self.base_height_y
        self.goal_state =  [self.x,y,z]
        self.update_goal_state()
        # self.visual_top_trajs_ingym()
        
    def _dynamic_object_moveDesign_leftright_track(self):
        # Update velocity vector based on move bounds and current pose
        collision_T = copy.deepcopy(self.world_instance.get_pose(self.collision_body_handle))
        if collision_T.p.z < self.coll_movebound_leftright[0] or \
           collision_T.p.z > self.coll_movebound_leftright[1] :
            self.uporient *= -1
        # Move the object based on the velocity vector
        collision_T.p.z += self.uporient * self.coll_dt_scale
        self.gym.set_rigid_transform(
            self.env_ptr, self.collision_base_handle, collision_T)

    def update_collision_state(self,pose):
        # target_base 与 body的讨论见草稿 10.07
        collision_T = copy.deepcopy(self.world_instance.get_pose(self.collision_body_handle))
        collision_T.p = gymapi.Vec3(pose[0],pose[1],pose[2])
        self.gym.set_rigid_transform(self.env_ptr, self.collision_base_handle, collision_T)


    def traj_append(self):
        self.traj_log['position'].append(self.command['position'])
        self.traj_log['velocity'].append(self.command['velocity'])
        self.traj_log['acc'].append(self.command['acceleration'])
        self.traj_log['des'].append(self.jnq_des)

    def traj_append_multimodal(self):
        self.traj_log['weights'].append(self.mpc_control.controller.weights_divide.cpu().numpy())

    def plot_traj_multimodal(self):
        weights = np.matrix(self.traj_log['weights'])
        plt.figure()
        axs = [plt.subplot(2,1,i+1) for i in range(2)]
        axs[0].set_title('weight assignment')
        axs[0].plot(weights[:,0], 'r', label='greedy')
        axs[0].legend() 
        axs[1].plot(weights[:,1], 'g', label='sensi')
        axs[1].legend() 
        plt.savefig('weight_assignment.png')

    def plot_traj(self):
        plt.figure()
        position = np.matrix(self.traj_log['position'])
        vel = np.matrix(self.traj_log['velocity'])
        acc = np.matrix(self.traj_log['acc'])
        des = np.matrix(self.traj_log['des'])
        axs = [plt.subplot(3,1,i+1) for i in range(3)]
        if(len(axs) >= 3):
            axs[0].set_title('Position')
            axs[1].set_title('Velocity')
            axs[2].set_title('Acceleration')
            axs[0].plot(position[:,0], 'r', label='joint1')
            axs[0].plot(position[:,2], 'g',label='joint3')
            axs[0].plot(des[:,0], 'r-.', label='joint1_des')
            axs[0].plot(des[:,2],'g-.', label='joint3_des')
            axs[0].legend()
            axs[1].plot(vel[:,0], 'r',label='joint1')
            axs[1].plot(vel[:,2], 'g', label='joint3')
            axs[1].legend()
            axs[2].plot(acc[:,0], 'r',label='joint1')
            axs[2].plot(acc[:,2], 'g', label='joint3')
            axs[2].legend()
        plt.savefig('trajectory.png')
        plt.show()
