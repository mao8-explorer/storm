
from isaacgym import gymapi
import torch
import trimesh.transformations as tra
import yaml
import numpy as np
from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_gym_configs_path, join_path, get_assets_path
from storm_kit.mpc.task.reacher_task import ReacherTask
from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
np.set_printoptions(precision=2)
torch.multiprocessing.set_start_method('spawn', force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class FrankaEnvBase(object):
    def __init__(self, gym_instance):
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

    def _environment_init(self):
        self._initialize_robot_simulation() # robot_sim 
        self._initialize_world_and_camera() # world_instance
        self._initialize_mpc_control() # mpc_control 
        self._initialize_env_objects() # 设置 gym 可操作物 handle
        self._initialize_rospy()
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
            w_T_r=self.w_T_r 
        )

    def _initialize_mpc_control(self):
        # Initialize the MPC control
        self.mpc_control = ReacherTask(
            self.mpc_config,
            self.world_description, 
            self.tensor_args)
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

        self.collision_obj_base_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, collision_obj, 0)
        self.collision_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, collision_obj, 6)

        self.target_base_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, target_object, 0)
        self.target_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, target_object, 6)

        self.ee_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, current_ee_obj, 0)
      
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
        self.gym.set_rigid_transform(self.env_ptr, self.collision_obj_base_handle, object_pose)


    def _initialize_rospy(self):
        #  all ros_related
        self.msg = PointCloud2()
        self.msg.header.frame_id = "world"
        self.msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        self.msg.is_bigendian = False
        self.msg.point_step = 12
        self.msg.is_dense = False
        self.coll_msg = Float32()
        self.coll_robot_pub = rospy.Publisher('robot_collision', Float32, queue_size=10)
        self.pub_env_pc = rospy.Publisher('env_pc', PointCloud2, queue_size=5)
        self.pub_robot_link_pc = rospy.Publisher('robot_link_pc', PointCloud2, queue_size=5)
        

    def _init_point_transform(self):
        w_T_robot = torch.eye(4)
        quat = torch.tensor([self.w_T_r.r.w, self.w_T_r.r.x, self.w_T_r.r.y, self.w_T_r.r.z]).unsqueeze(0)
        rot = quaternion_to_matrix(quat)
        w_T_robot[0,3] = self.w_T_r.p.x
        w_T_robot[1,3] = self.w_T_r.p.y
        w_T_robot[2,3] = self.w_T_r.p.z
        w_T_robot[:3,:3] = rot[0]
        self.w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3,3].unsqueeze(0),
                                            rot=w_T_robot[0:3,0:3].unsqueeze(0))    
 
    def update_goal_state(self):
        goal_state = self.goal_state
        object_pose = self.world_instance.get_pose(self.target_body_handle)
        object_pose.p = gymapi.Vec3(goal_state[0],goal_state[1],goal_state[2])
        self.gym.set_rigid_transform(self.env_ptr, self.target_base_handle, object_pose)

    def updateGymVisual_GoalUpdate(self):
               
        # trans ee_pose in robot_coordinate to world coordinate
        ee_pose = gymapi.Transform()
        pose_state = self.mpc_control.controller.rollout_fn.get_ee_pose(self.curr_state_tensor)
        e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
        e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
        ee_pose.p = gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2])
        ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
        ee_pose = self.w_T_r * ee_pose
        self.gym.set_rigid_transform(self.env_ptr, self.ee_body_handle, ee_pose)

        # if current_ee_pose in goal_pose thresh ,update to next goal_pose
        thresh = 0.005
        if (np.linalg.norm(np.array(self.goal_state) - np.ravel([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z])) < thresh):
            self.goal_state = self.goal_list[(self.goal_flagi+1) % len(self.goal_list)]
            self.update_goal_state()
            self.goal_flagi += 1
            print("next goal",self.goal_flagi)

        # gym_instance.clear_lines() 放在while初始，在订阅点云前清屏
        top_trajs = self.mpc_control.top_trajs.cpu().float()  # .numpy()
        n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
        w_pts = self.w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)

        top_trajs = w_pts.cpu().numpy()
        color = np.array([0.0, 1.0, 0.0])
        for k in range(top_trajs.shape[0]):
            pts = top_trajs[k, :, :]
            color[0] = float(k) / float(top_trajs.shape[0])
            color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
            self.gym_instance.draw_lines(pts, color=color)

    def monitorGoalupdate(self):
        """
        谁控制了target_body_handle 谁控制了MPC_Policy_Goal 不管是通过Gym还是通过代码的方式 都可以
        """
        pose = self.world_instance.get_pose(self.target_body_handle)
        pose = self.w_T_r.inverse() * pose #将world坐标系下的目标点转到robot坐标系下
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


       
    def pub_pointcloud(self,pc,pub_handle):

        self.msg.header.stamp = rospy.Time().now()
        if len(pc.shape) == 3:
            self.msg.height = pc.shape[1]
            self.msg.width = pc.shape[0]
        else:
            self.msg.height = 1
            self.msg.width = len(pc)

        self.msg.row_step = self.msg.point_step * pc.shape[0]
        self.msg.data = np.asarray(pc, np.float32).tobytes()

        pub_handle.publish(self.msg)   


    def updateRosMsg(self):
        # ROS Publish
        robot_collision_cost = self.mpc_control.controller.rollout_fn \
                                    .robot_self_collision_cost(self.curr_state_tensor.unsqueeze(0)[:,:,:7]) \
                                    .squeeze().cpu().numpy()
        self.coll_msg.data = robot_collision_cost
        self.coll_robot_pub.publish(self.coll_msg)
        # pub env_pointcloud and robot_link_spheres
        w_batch_link_spheres = self.mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.robot_coll.w_batch_link_spheres 
        spheres = [s[0][:, :3].cpu().numpy() for s in w_batch_link_spheres]
        # 将所有球体位置信息合并为一个NumPy数组
        robotsphere_positions = np.concatenate(spheres, axis=0)
        self.pub_pointcloud(robotsphere_positions, self.pub_robot_link_pc)
        collision_grid_pc = self.collision_grid.cpu().numpy() 
        self.pub_pointcloud(collision_grid_pc, self.pub_env_pc)

     
    def _dynamic_object_moveDesign(self):
        # Update velocity vector based on move bounds and current pose
        if np.abs(self.move_pose.p.x) >= self.move_bounds:
            self.velocity_vector *= -1
        # Move the object based on the velocity vector
        dt_scale = 0.01
        self.move_pose.p.x += self.velocity_vector[0][0] * dt_scale
        self.move_pose.p.y += self.velocity_vector[0][1] * dt_scale
        self.move_pose.p.z += self.velocity_vector[0][2] * dt_scale
        w_move = self.w_T_r * self.move_pose
        self.gym.set_rigid_transform(
            self.env_ptr, self.collision_obj_base_handle, w_move)

