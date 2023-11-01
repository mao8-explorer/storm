
import os
import numpy as np
import torch
import rospy
import copy
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)


class ReacherEnvBase():
    def __init__(self):
        self.pkg_path = "/home/zm/MotionPolicyNetworks/storm_ws/storm/storm_ros"
        self.storm_path = os.path.dirname(self.pkg_path)
        rospy.loginfo(self.storm_path)
        self.world_description = os.path.join(self.storm_path, rospy.get_param('~world_description', 'content/configs/gym/collision_primitives_3d.yml'))
        self.mpc_config = os.path.join(self.storm_path, rospy.get_param('~mpc_config', 'content/configs/mpc/franka_real_robot_reacher_simplify.yml'))
        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.env_pc_topic = rospy.get_param('~env_pc_topic', '/points2_filter')
        self.goal_command_fromMPC_topic = rospy.get_param("~goal_state", "goal_state")
        self.control_dt = rospy.get_param('~control_dt', 0.07)
        self.joint_names = rospy.get_param('~robot_joint_names', ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'])
        self.device = torch.device('cuda', 0)
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}
        self.traj_log = {'position':[], 'velocity':[], 'acc':[] , 'des':[] , 'weights':[] , 'robot_position': [], 'robot_velocity': []}

        #buffers for different messages
        self.buffer_differMsg_init()
        self.last_ee_goal_pos = np.zeros(3)
        self.last_ee_goal_quat = np.zeros(4)
        self.point_array =None
        self.curr_collision = 0
        #  Flag Declaration
        self.State_Sub_On = False
        self.Goal_Sub_On = False


    def ros_handle_init(self):
        #ROS Initialization
        self.command_pub = rospy.Publisher(self.joint_command_topic, JointState, queue_size=1, tcp_nodelay=True, latch=False)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)
        self.ee_goal_sub = rospy.Subscriber(self.ee_goal_topic, PoseStamped, self.ee_goal_callback, queue_size=1)
        self.env_pc_sub = rospy.Subscriber(self.env_pc_topic, PointCloud2, self.env_pc_callback, queue_size=5)
        self.marker_pub = rospy.Publisher('trajectory_pub', Marker, queue_size=10)
        self.coll_robot_pub = rospy.Publisher('robot_collision', Float32, queue_size=10)
        self.pub_env_pc = rospy.Publisher('env_pc', PointCloud2, queue_size=5)
        self.pub_robot_link_pc = rospy.Publisher('robot_link_pc', PointCloud2, queue_size=5)
        self.goal_command_fromMPC_pub = rospy.Publisher(self.goal_command_fromMPC_topic , PoseStamped, queue_size=1, tcp_nodelay=True, latch=False)

        while not self.State_Sub_On:
            rospy.logwarn('[MPCPoseReacher]: Waiting for robot state.')
            rospy.sleep(0.5)

        while not self.Goal_Sub_On:
            rospy.logwarn('[MPCPoseReacher]: Waiting for ee goal.')
            rospy.sleep(0.5)
    

        self.goal_command_fromMPC = PoseStamped()
        self.goal_command_fromMPC.header.stamp = rospy.Time.now()
        self.goal_command_fromMPC.pose.position.x = self.ee_goal_pos[0]
        self.goal_command_fromMPC.pose.position.y = self.ee_goal_pos[1]
        self.goal_command_fromMPC.pose.position.z = self.ee_goal_pos[2]
        self.goal_command_fromMPC_pub.publish(self.goal_command_fromMPC)



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
        self.mpc_command.position = np.zeros(7)
        self.mpc_command.velocity = np.zeros(7)
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
        ee_goal_pos = np.array([
            ee_goal_msg.pose.position.x,
            ee_goal_msg.pose.position.y,
            ee_goal_msg.pose.position.z])
        ee_goal_quat = np.array([
            ee_goal_msg.pose.orientation.w,
            ee_goal_msg.pose.orientation.x,
            ee_goal_msg.pose.orientation.y,
            ee_goal_msg.pose.orientation.z])
        #check if goal was updated
        if  (np.linalg.norm(self.last_ee_goal_pos - ee_goal_pos) > 0.0) or (
             np.linalg.norm(self.last_ee_goal_quat - ee_goal_quat) > 0.0):
            self.last_ee_goal_pos = ee_goal_pos
            self.last_ee_goal_quat = ee_goal_quat
            self.policy.update_params(goal_ee_pos = ee_goal_pos,
                                    goal_ee_quat = ee_goal_quat)  
            self.rollout_fn.goal_jnq = None
            
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
    
    def GoalUpdate(self):
        
        # TODO: can it get from topic? call robotmodel to get ee_pos may costly
        # pose_state = self.rollout_fn.get_ee_pose(self.curr_state_tensor)
        # cur_e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
        cur_e_pos = self.rollout_fn.curr_ee_pos.cpu().detach().numpy()
        # cur_e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
        # if current_ee_pose in goal_pose thresh ,update to next goal_pose
        if (np.linalg.norm(np.array(self.ee_goal_pos - cur_e_pos)) < self.thresh):
            self.goal_flagi += 1
            self.ee_goal_pos = self.goal_list[(self.goal_flagi+1) % len(self.goal_list)]
            self.rollout_fn.goal_jnq = None

            self.goal_command_fromMPC.header.stamp = rospy.Time.now()
            self.goal_command_fromMPC.pose.position.x = self.ee_goal_pos[0]
            self.goal_command_fromMPC.pose.position.y = self.ee_goal_pos[1]
            self.goal_command_fromMPC.pose.position.z = self.ee_goal_pos[2]        
            self.goal_command_fromMPC_pub.publish(self.goal_command_fromMPC)

            log_message = "next goal: {}, lap_count: {}, collision_count: {}".format(self.goal_flagi, self.goal_flagi / len(self.goal_list), self.curr_collision)
            rospy.loginfo(log_message)
            if self.goal_flagi %  ( 2*len(self.goal_list) )== 1 : 
                self.traj_log = {'position':[], 'velocity':[], 'acc':[] , 'des':[] , 'weights':[] , 'robot_position': [], 'robot_velocity': []}
                rospy.loginfo("置零")


    def visual_top_trajs(self):

        # 可视化末端规划轨迹 MPPI.py --> top_trajs
        top_trajs = self.policy.controller.top_trajs.cpu().detach().numpy()  # shape is 10*30*3
        # 将该组轨迹的点转换为ROS消息中的点列表
        points = [Point(x=top_trajs[i][j][0], 
                        y=top_trajs[i][j][1], 
                        z=top_trajs[i][j][2]) 
                for i in range(1) for j in range(top_trajs.shape[1])]
                # for i in range(top_trajs.shape[0]) for j in range(top_trajs.shape[1])]
            # 将点列表添加到marker消息中
        self.marker_EE_trajs.points = points
            # 更新header中的时间戳和ID
        self.marker_EE_trajs.header.stamp = rospy.Time.now()
        # 发布marker消息
        self.marker_pub.publish(self.marker_EE_trajs)

    def visual_top_trajs_multimodal(self):

        # 可视化末端规划轨迹 MPPI.py --> top_trajs
        top_trajs = self.rollout_fn.top_trajs.cpu().detach().numpy()  # shape is 10*30*3
        # 将该组轨迹的点转换为ROS消息中的点列表
        points = [Point(x=top_trajs[i][j][0], 
                        y=top_trajs[i][j][1], 
                        z=top_trajs[i][j][2]) 
                # for i in range(1) for j in range(top_trajs.shape[1])]
                for i in range(top_trajs.shape[0]) for j in range(top_trajs.shape[1])]
            # 将点列表添加到marker消息中
        self.marker_EE_trajs.points = points
            # 更新header中的时间戳和ID
        self.marker_EE_trajs.header.stamp = rospy.Time.now()
        # 发布marker消息
        self.marker_pub.publish(self.marker_EE_trajs)


    def traj_append(self):
        self.traj_log['position'].append(self.command['position'])
        self.traj_log['velocity'].append(self.command['velocity'])
        self.traj_log['acc'].append(self.command['acceleration'])
        self.traj_log['des'].append(self.jnq_des)
        # visual robot_state | command_state relationship
        self.traj_log['robot_position'].append(self.robot_state['position'])
        self.traj_log['robot_velocity'].append(self.robot_state['velocity'])
 
    def traj_append_multimodal(self):
        self.traj_log['weights'].append(self.policy.controller.weights_divide.cpu().detach().numpy())


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

        # try to compare "command" with "robotstate"
        robot_position = np.matrix(self.traj_log['robot_position'])
        robot_vel = np.matrix(self.traj_log['robot_velocity'])
        plt.figure()
        axs = [plt.subplot(4,1,i+1) for i in range(4)]
        if(len(axs) >= 3):
            axs[0].set_title('Position_Command_RobotState')
            axs[2].set_title('Velocity_Command_RoborState')
            axs[0].plot(position[:,0], 'r', label='joint1_commandpos')
            axs[0].plot(robot_position[:,0], 'g',label='joint1_robotpos')
            axs[0].legend()
            axs[1].plot(position[:,2], 'r', label='joint3_commandpos')
            axs[1].plot(robot_position[:,2], 'g',label='joint3_robotpos')
            axs[1].legend()
            axs[2].plot(vel[:,0], 'r', label='joint1__commandvel')
            axs[2].plot(robot_vel[:,0], 'g',label='joint1__robotvel')
            axs[2].legend()
            axs[3].plot(vel[:,2], 'r', label='joint3_commandvel')
            axs[3].plot(robot_vel[:,2], 'g',label='joint3_robotvel')
            axs[3].legend()
        plt.savefig('command_robotstate.png')

        plt.show()



    def close(self):

        self.command_pub.unregister() 
        self.state_sub.unregister()
        self.ee_goal_sub.unregister() 
        self.env_pc_sub.unregister() 
        self.marker_pub.unregister() 
        self.coll_robot_pub.unregister() 
        self.pub_env_pc.unregister() 
        self.pub_robot_link_pc.unregister()
        self.goal_command_fromMPC_pub.unregister()