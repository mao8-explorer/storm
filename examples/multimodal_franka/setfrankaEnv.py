
from isaacgym import gymapi
import torch
import trimesh.transformations as tra
import yaml
import numpy as np
from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.mpc.task.reacher_task import ReacherTask
import matplotlib.pyplot as plt
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

class setfrankaEnv(object):
    def __init__(self,args, gym_instance):

        # yml配置
        self.robot_coll_description = 'franka.yml' #没啥用 通过franka_reacher.yml 中的"collision_spheres"指定
        self.mpc_config = args.robot + '_reacher.yml'
        self.world_description = 'collision_primitives_3d.yml'

        self.gym_instance = gym_instance
        self.args = args
        self.device = torch.device('cuda', 0)
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}

        self.gym = gym_instance.gym
        self.sim = gym_instance.sim
        self.env_ptr = gym_instance.env_list[0]
        self.viewer = gym_instance.viewer
        self._initialize_robot_simulation() # robot_sim 
        self._initialize_world_and_camera() # world_instance
        self._initialize_mpc_control() # mpc_control 
        self._initialize_env_objects() # 设置 gym 可操作物

        self._initialize_rospy()

    def _initialize_robot_simulation(self):
        # Initialize the robot simulation
  
        robot_yml = join_path(get_gym_configs_path(), self.args.robot + '.yml')
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


    def _initialize_world_and_camera(self):
        # Initialize the world instance and camera_pose

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
            w_T_r=self.robot_sim.spawn_robot_pose
        )

    def _initialize_mpc_control(self):
        # Initialize the MPC control
        self.mpc_control = ReacherTask(
            self.mpc_config, 
            self.robot_coll_description, 
            self.world_description, 
            self.tensor_args)
        # update goal_joint_space:
        franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.mpc_control.update_params(goal_state=franka_bl_state)

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

        object_pose.p = gymapi.Vec3(0.580,0.626, -0.274)
        object_pose.r = gymapi.Quat(0.278,0.668,-0.604,0.334)
        self.gym.set_rigid_transform(self.env_ptr, self.collision_obj_base_handle, object_pose)
     

    def _dynamic_object_moveDesign(self):
        # Update velocity vector based on move bounds and current pose
        if self.move_pose.p.x <= self.move_bounds[0][0] or self.move_pose.p.x >= self.move_bounds[1][0]:
            self.velocity_vector *= -1

        # Move the object based on the velocity vector
        dt_scale = 0.01
        self.move_pose.p.x += self.velocity_vector[0][0] * dt_scale
        self.move_pose.p.y += self.velocity_vector[0][1] * dt_scale
        self.move_pose.p.z += self.velocity_vector[0][2] * dt_scale
        w_move = self.w_T_r * self.move_pose
        self.gym.set_rigid_transform(
            self.env_ptr, self.collision_obj_base_handle, w_move
        )

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