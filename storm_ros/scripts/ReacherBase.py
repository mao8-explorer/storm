
import os
import numpy as np
import torch
import rospy
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from storm_ros.utils.tf_translation import get_world_T_cam


class ReacherEnvBase():
    def __init__(self):
        self.world_T_cam = get_world_T_cam() # transform : "world", "rgb_camera_link"
        self.pkg_path = "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_ros"
        self.storm_path = os.path.dirname(self.pkg_path)
        rospy.loginfo(self.storm_path)
        self.world_description = os.path.join(self.storm_path, rospy.get_param('~world_description', 'content/configs/gym/collision_primitives_3d.yml'))
        self.mpc_config = os.path.join(self.storm_path, rospy.get_param('~mpc_config', 'content/configs/mpc/franka_real_robot_reacher.yml'))
        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.env_pc_topic = rospy.get_param('~env_pc_topic', '/points2_filter')
        self.control_dt = rospy.get_param('~control_dt', 0.05)
        self.joint_names = rospy.get_param('~robot_joint_names', ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'])
        self.device = torch.device('cuda', 0)
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}

        #buffers for different messages
        self.buffer_differMsg_init()

        self.ee_goal_pos = None
        self.ee_goal_quat = None
        self.point_array =None

        #ROS Initialization
        self.command_pub = rospy.Publisher(self.joint_command_topic, JointState, queue_size=1, tcp_nodelay=True, latch=False)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)
        self.ee_goal_sub = rospy.Subscriber(self.ee_goal_topic, PoseStamped, self.ee_goal_callback, queue_size=1)
        self.env_pc_sub = rospy.Subscriber(self.env_pc_topic, PointCloud2, self.env_pc_callback, queue_size=5)
        self.marker_pub = rospy.Publisher('trajectory_pub', Marker, queue_size=10)
        self.coll_robot_pub = rospy.Publisher('robot_collision', Float32, queue_size=10)
        self.pub_env_pc = rospy.Publisher('env_pc', PointCloud2, queue_size=5)
        self.pub_robot_link_pc = rospy.Publisher('robot_link_pc', PointCloud2, queue_size=5)

        #  Flag Declaration
        self.State_Sub_On = False
        self.Goal_Sub_On = False


    def buffer_differMsg_init(self):
        # 末端轨迹msg
        self.marker_EE_trajs = Marker()
        self.marker_EE_trajs.header.stamp = rospy.Time.now()
        self.marker_EE_trajs.header.frame_id = "panda_link0"
        self.marker_EE_trajs.action = Marker.ADD
        self.marker_EE_trajs.pose.orientation.w = 1.0
        self.marker_EE_trajs.type = Marker.SPHERE_LIST
        self.marker_EE_trajs.scale.x = 0.005
        self.marker_EE_trajs.scale.y = 0.005
        self.marker_EE_trajs.scale.z = 0.005
        self.marker_EE_trajs.color.r = 0
        self.marker_EE_trajs.color.g = 1
        self.marker_EE_trajs.color.b = 0
        self.marker_EE_trajs.color.a = 1

        # joint_command_msg
        self.mpc_command = JointState()
        self.mpc_command.name = self.joint_names
        self.mpc_command.effort = np.zeros(7)

        # joint_state_msg
        self.robot_state = {
            'position': np.zeros(7),
            'velocity': np.zeros(7),
            'acceleration': np.zeros(7)}

        # pointcloud msg
        self.pc_msg = PointCloud2()
        self.pc_msg.header.frame_id = "world"
        self.pc_msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        self.pc_msg.is_bigendian = False
        self.pc_msg.point_step = 12
        self.pc_msg.is_dense = False

        # robot_self_collision_msg
        self.coll_msg = Float32()


    def robot_state_callback(self, joint_state_msg):
        self.State_Sub_On = True
        self.robot_state['position'] = np.array(joint_state_msg.position[2:])
        self.robot_state['velocity'] = np.array(joint_state_msg.velocity[2:])

    def ee_goal_callback(self, ee_goal_msg):
        self.Goal_Sub_On = True
        self.New_EE_Goal = True
        self.ee_goal_pos = np.array([
            ee_goal_msg.pose.position.x,
            ee_goal_msg.pose.position.y,
            ee_goal_msg.pose.position.z])
        self.ee_goal_quat = np.array([
            ee_goal_msg.pose.orientation.w,
            ee_goal_msg.pose.orientation.x,
            ee_goal_msg.pose.orientation.y,
            ee_goal_msg.pose.orientation.z])

    def env_pc_callback(self, env_pc_msg):
        point_generator = pc2.read_points(env_pc_msg)
        self.point_array = np.array(list(point_generator))

        
    def pub_pointcloud(self,pc,pub_handle):

        self.pc_msg.header.stamp = rospy.Time().now()
        if len(pc.shape) == 3:
            self.pc_msg.height = pc.shape[1]
            self.pc_msg.width = pc.shape[0]
        else:
            self.pc_msg.height = 1
            self.pc_msg.width = len(pc)
        self.pc_msg.row_step = self.pc_msg.point_step * pc.shape[0]
        self.pc_msg.data = np.asarray(pc, np.float32).tobytes()

        pub_handle.publish(self.pc_msg)   
    
    def visual_top_trajs(self):

        # 可视化末端规划轨迹 MPPI.py --> top_trajs
        top_trajs = self.policy.top_trajs.cpu().float().numpy()  # shape is 10*30*3
        # 将该组轨迹的点转换为ROS消息中的点列表
        points = [Point(x=top_trajs[i][j][0], 
                        y=top_trajs[i][j][1], 
                        z=top_trajs[i][j][2]) 
                for i in range(top_trajs.shape[0]) for j in range(top_trajs.shape[1])]
            # 将点列表添加到marker消息中
        self.marker_EE_trajs.points = points
            # 更新header中的时间戳和ID
        self.marker_EE_trajs.header.stamp = rospy.Time.now()
        # 发布marker消息
        self.marker_pub.publish(self.marker_EE_trajs)

    
    def close(self):
        self.command_pub.unregister()
        self.state_sub.unregister()
        self.ee_goal_sub.unregister()
        self.marker_pub.unregister()
