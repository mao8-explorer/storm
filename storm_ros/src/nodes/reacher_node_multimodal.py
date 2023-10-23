#!/usr/bin/env python

#General imports
import os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import torch
import rospy
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState
import rospkg
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField
from storm_kit.mpc.task.reacher_task import ReacherTask
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
import yaml
import tf2_ros
import tf
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True




def transform_to_matrix(transform):
    translation = [transform.transform.translation.x,
                   transform.transform.translation.y,
                   transform.transform.translation.z]

    rotation = [transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w]

    rotation_matrix = tf.transformations.quaternion_matrix(rotation)

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation

    transform_matrix = np.dot(translation_matrix, rotation_matrix)

    return transform_matrix

def get_world_T_cam():
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    transform = tf_buffer.lookup_transform("world", "rgb_camera_link", rospy.Time(0), rospy.Duration(1.0))
    return transform_to_matrix(transform)
    

class MPCReacherNode():
    def __init__(self) -> None:

        self.world_T_cam = get_world_T_cam()
        print("world_T_cam is :",self.world_T_cam)


        rospack = rospkg.RosPack()
        # self.pkg_path = rospack.get_path('storm_ros')
        self.pkg_path = "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_ros"
        self.storm_path = os.path.dirname(self.pkg_path)
        rospy.loginfo(self.storm_path)

        world_file = 'collision_primitives_3d.yml'
        world_yml = join_path(get_gym_configs_path(), world_file)

        with open(world_yml) as file:
            self.world_params = yaml.load(file, Loader=yaml.FullLoader)  # world_model

        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.env_pc_topic = rospy.get_param('~env_pc_topic', '/points2_filter')

        self.marker_pub = rospy.Publisher('trajectory_pub', Marker, queue_size=10)
        self.coll_marker_pub = rospy.Publisher('collision_pub', Marker, queue_size=10)

        self.world_description = os.path.join(self.storm_path, rospy.get_param('~world_description', 'content/configs/gym/collision_primitives_3d.yml'))
        self.robot_coll_description = os.path.join(self.storm_path, rospy.get_param('~robot_coll_description', 'content/configs/robot/franka_multipoint.yml'))
        self.mpc_config = os.path.join(self.storm_path, rospy.get_param('~mpc_config', 'content/configs/mpc/franka_real_robot_reacher.yml'))
        self.control_dt = rospy.get_param('~control_dt', 0.05)

        self.joint_names = rospy.get_param('~robot_joint_names', ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'])
        
        self.device = torch.device('cuda', 0)
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}

        #STORM Initialization
        self.policy = ReacherTask(self.mpc_config, self.robot_coll_description, self.world_description, self.tensor_args)

        #buffers for different messages
        self.mpc_command = JointState()
        self.mpc_command.name = self.joint_names
        self.mpc_command.effort = np.zeros(7)
        # self.gripper_state = JointState()
        # self.robot_state = JointState()
        self.command_header = None
        self.gripper_state = {
            'position': np.zeros(2),
            'velocity': np.zeros(2),
            'acceleration': np.zeros(2)}
        self.robot_state = {
            'position': np.zeros(7),
            'velocity': np.zeros(7),
            'acceleration': np.zeros(7)}
        self.ee_goal_pos = None
        self.ee_goal_quat = None
        self.point_array =None

        #ROS Initialization
        self.command_pub = rospy.Publisher(self.joint_command_topic, JointState, queue_size=1, tcp_nodelay=True, latch=False)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)
        self.ee_goal_sub = rospy.Subscriber(self.ee_goal_topic, PoseStamped, self.ee_goal_callback, queue_size=1)
        self.env_pc_sub = rospy.Subscriber(self.env_pc_topic, PointCloud2, self.env_pc_callback, queue_size=5)

        self.control_freq = float(1.0/self.control_dt)
        self.rate = rospy.Rate(self.control_freq)

        self.state_sub_on = False
        self.goal_sub_on = False
        self.tstep = 0
        self.start_t = None
        self.first_iter = True

        self.marker_msg = Marker()
        self.marker_init()
        self.coll_create_init()


    def coll_create_init(self):

        world_objs = self.world_params['world_model']['coll_objs']
        sphere_objs = world_objs['sphere']
        if('cube' in world_objs):
            cube_objs = world_objs['cube']
        else:
            cube_objs = []

        for j_idx, j in enumerate(sphere_objs):
            position = sphere_objs[j]['position']
            
            r = sphere_objs[j]['radius']

            self.coll_msg_pub("sphere",position,r,j_idx)
            rospy.sleep(0.1)
            
        for j_idx, j in enumerate(cube_objs):
            pose = cube_objs[j]['pose']
            dims = cube_objs[j]['dims']
            self.coll_msg_pub("cube",pose,dims,j_idx)
            rospy.sleep(0.1)

    def coll_msg_pub(self,coll_type,pose,dims,i):

        coll_msg = Marker()
        coll_msg.header.stamp = rospy.Time.now()  
        coll_msg.header.frame_id = "panda_link0"    
        coll_msg.action = Marker.ADD
        coll_msg.pose.orientation.w = 1.0
        coll_msg.pose.position.x = pose[0]
        coll_msg.pose.position.y = pose[1]
        coll_msg.pose.position.z = pose[2]

        if(coll_type == "cube"):
            coll_msg.id=i # 用来区别differ msg
            coll_msg.ns = "cube"
            coll_msg.type = Marker.CUBE
            coll_msg.scale.x = dims[0]
            coll_msg.scale.y = dims[1]
            coll_msg.scale.z = dims[2]
            coll_msg.color.g = 1
        if(coll_type == "sphere"):
            coll_msg.id=i*10 # 用来区别differ msg
            coll_msg.ns = "sphere"
            coll_msg.type = Marker.SPHERE
            coll_msg.scale.x = dims
            coll_msg.scale.y = dims
            coll_msg.scale.z = dims
            coll_msg.color.g = 1

        coll_msg.color.r = 0
        coll_msg.color.b = 0
        coll_msg.color.a = 0.8
        coll_msg.lifetime = rospy.Duration()
        self.coll_marker_pub.publish(coll_msg)


    def marker_init(self):
        # 创建一个marker消息

        self.marker_msg.header.stamp = rospy.Time.now()
        self.marker_msg.header.frame_id = "panda_link0"
        self.marker_msg.ns = ""
        self.marker_msg.action = Marker.ADD
        self.marker_msg.pose.orientation.w = 1.0
        self.marker_msg.type = Marker.SPHERE_LIST
        self.marker_msg.scale.x = 0.005
        self.marker_msg.scale.y = 0.005
        self.marker_msg.scale.z = 0.005

    def robot_state_callback(self, msg):
        self.state_sub_on = True
        self.command_header = msg.header
        self.robot_state['position'] = np.array(msg.position[2:])
        self.robot_state['velocity'] = np.array(msg.velocity[2:])
        self.robot_state['acceleration'] = np.zeros_like(self.robot_state['velocity'])


    def ee_goal_callback(self, msg):
        self.goal_sub_on = True
        self.new_ee_goal = True
        self.ee_goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z])
        self.ee_goal_quat = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z])

    def env_pc_callback(self, msg):
        point_generator = pc2.read_points(msg)
        self.point_array = np.array(list(point_generator))

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


    def control_loop(self):
        self._initialize_rospy()
        rospy.loginfo('[MPCPoseReacher]: Controller running')
        self.start_t = rospy.get_time()

        while not rospy.is_shutdown():
            #only do something if state and goal have been received
            if self.state_sub_on and self.goal_sub_on:
                #check if goal was updated
                if self.new_ee_goal:
                    self.policy.update_params(goal_ee_pos = self.ee_goal_pos,
                        goal_ee_quat = self.ee_goal_quat)
                    self.new_ee_goal = False

                # 更新scene_grid 
                        # Assuming self.point_array is your original point cloud data
                # It should have shape (N, 3) where N is the number of points
                # Each row is a point (x, y, z)
                point_array = np.hstack((self.point_array, np.ones((self.point_array.shape[0], 1))))  # Adding homogenous coordinate
                transformed_points = np.dot(point_array, self.world_T_cam.T)  # Transform all points at once
                self.point_array = transformed_points[:, :3]  # Removing the homogenous coordinate
                # mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll._compute_dynamic_sdfgrid(scene_pc)
                collision_grid = self.policy.controller.rollout_fn. \
                                    primitive_collision_cost.robot_world_coll.world_coll. \
                                    _compute_dynamic_voxeltosdf(self.point_array, visual = True)
                
                self.tstep = rospy.get_time() - self.start_t
                #get mpc command
                command , _= self.policy.get_multimodal_command(
                    self.tstep, self.robot_state, control_dt=self.control_dt)
                
                #publish mpc command
                self.mpc_command.header = self.command_header
                self.mpc_command.header.stamp = rospy.Time.now()
                self.mpc_command.position = command['position']
                self.mpc_command.velocity = command['velocity']
                self.mpc_command.effort =  command['acceleration']
                self.command_pub.publish(self.mpc_command)


                top_trajs = self.policy.top_trajs.cpu().float().numpy()  # .numpy()
                batch = 10
                horizen = 30
                for i in range(batch):
                      # 将颜色列表中的颜色分配给该组轨迹的marker
                    self.marker_msg.color.r = float(i+1) / float(batch)
                    self.marker_msg.color.g = 1.0 - float(i+1) / float(batch)
                    self.marker_msg.color.b = 0
                    self.marker_msg.color.a = 1
                    # 将该组轨迹的点转换为ROS消息中的点列表
                    points = []
                    for j in range(horizen):
                        point = Point()
                        point.x = top_trajs[i][j][0]
                        point.y = top_trajs[i][j][1]
                        point.z = top_trajs[i][j][2]
                        points.append(point)
                    # 将点列表添加到marker消息中
                    self.marker_msg.points = points
                     # 更新header中的时间戳和ID
                    self.marker_msg.header.stamp = rospy.Time.now()
                    self.marker_msg.id = i
                    # 发布marker消息
                    self.marker_pub.publish(self.marker_msg)

                # pub env_pointcloud and robot_link_spheres
                w_batch_link_spheres = self.policy.controller.rollout_fn.primitive_collision_cost.robot_world_coll.robot_coll.w_batch_link_spheres 
                spheres = [s[0][:, :3].cpu().numpy() for s in w_batch_link_spheres]
                # 将所有球体位置信息合并为一个NumPy数组
                robotsphere_positions = np.concatenate(spheres, axis=0)
                self.pub_pointcloud(robotsphere_positions, self.pub_robot_link_pc)

                collision_grid_pc = collision_grid.cpu().numpy() 
                self.pub_pointcloud(collision_grid_pc, self.pub_env_pc)
       
            else:
                if (not self.state_sub_on) and (self.first_iter):
                    rospy.loginfo('[MPCPoseReacher]: Waiting for robot state.')
                if (not self.goal_sub_on) and (self.first_iter):
                    rospy.loginfo('[MPCPoseReacher]: Waiting for ee goal.')
            
            self.first_iter = False
            # self.rate.sleep()
    
    def close(self):
        self.command_pub.unregister()
        self.state_sub.unregister()
        self.ee_goal_sub.unregister()
        self.marker_pub.unregister()


if __name__ == "__main__":
    rospy.init_node("mpc_reacher_node", anonymous=True, disable_signals=True)    

    mpc_node = MPCReacherNode()

    try:
        mpc_node.control_loop()
    except KeyboardInterrupt:
        print('Exiting')
        mpc_node.close()