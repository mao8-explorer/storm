import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2 , JointState
from moveit_msgs.msg import DisplayTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory , RobotState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointField
import numpy as np

class RosVisualBase(object):
    def __init__(self):
        # robotmodel controller 
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.command_pub = rospy.Publisher(self.joint_command_topic, JointState, queue_size=1, tcp_nodelay=True, latch=False)
        self.mpc_command = JointState()
        joint_names = rospy.get_param('~robot_joint_names', ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'])
        self.mpc_command.name = joint_names
        self.mpc_command.effort = np.zeros(7)

        # trajectory visualization
        # display_trajectory 包括两类 RobotTrajectory and RobotState
        self.display_trajectory = DisplayTrajectory()
        self.display_trajectory.model_id = "panda"  # 替换为你的机器人模型ID    

        # 创建关节轨迹消息
        self.robot_trajectory = RobotTrajectory()
        # robot_trajectory.joint_trajectory.header.stamp = rospy.Time.now() # 时间缺失
        self.robot_trajectory.joint_trajectory.header.frame_id = "panda_link0"
        self.robot_trajectory.joint_trajectory.joint_names = joint_names
        self.trajectory_start = RobotState()
        self.trajectory_start.joint_state.header.frame_id = "panda_link0"
        self.trajectory_start.joint_state.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        # trajectory_start.joint_state.position = trajectory[0] state_state缺失

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
        self.marker_publisher = rospy.Publisher('arrow_markers', MarkerArray, queue_size=10)
        self.display_mean_trajectory_pub = rospy.Publisher('/move_group/display_mean_planned_path', DisplayTrajectory, queue_size=10)
        self.display_greedy_trajectory_pub = rospy.Publisher('/move_group/display_greedy_planned_path', DisplayTrajectory, queue_size=10)
        self.display_sensi_trajectory_pub = rospy.Publisher('/move_group/display_sensi_planned_path', DisplayTrajectory, queue_size=10)




        # 初始化MarkerArray消息
        self.arrow_markers = MarkerArray()
        self.marker = Marker()
        self.marker.header.frame_id = 'world'
        self.marker.type = Marker.ARROW
        self.marker.scale.x = 0.01
        self.marker.scale.y = 0.03
        self.marker.scale.z = 0.1
        self.marker.color.a = 1.0
        self.marker.pose.orientation.w = 1.0  # 固定方向为单位向量

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

    def pub_arrow(self): 
        """
        visualize gradient and velocity
        """
        self.marker.header.stamp = rospy.Time().now()
        self.arrow_markers.markers = []  # 清空之前的箭头
        # 创建current_vel_orient的箭头，设置为红色
        for i, gradient in enumerate(self.current_gradient*3.0):
            marker = Marker()
            marker.header = self.marker.header
            marker.type = self.marker.type
            marker.scale = self.marker.scale
            marker.color = self.marker.color
            marker.pose = self.marker.pose
            marker.id = i
            marker.action = Marker.ADD
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            start_point = Point()
            start_point.x = self.current_sphere_pos[i][0]
            start_point.y = self.current_sphere_pos[i][1]
            start_point.z = self.current_sphere_pos[i][2]
            marker.points.append(start_point)
            end_point = Point()
            end_point.x = gradient[0] + self.current_sphere_pos[i][0]
            end_point.y = gradient[1] + self.current_sphere_pos[i][1]
            end_point.z = gradient[2] + self.current_sphere_pos[i][2]
            marker.points.append(end_point)         

            self.arrow_markers.markers.append(marker)
        self.marker_publisher.publish(self.arrow_markers)
        self.marker.header.stamp = rospy.Time().now()
        self.arrow_markers.markers = []  # 清空之前的箭头
        # 创建current_gradient的箭头，设置为绿色
        for i, vel_orient in enumerate(self.current_vel_orient):
            marker = Marker()
            marker.header = self.marker.header
            marker.type = self.marker.type
            marker.scale = self.marker.scale
            marker.color = self.marker.color
            marker.pose = self.marker.pose
            marker.id = i + len(self.current_vel_orient)
            marker.action = Marker.ADD
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            start_point = Point()
            start_point.x = self.current_sphere_pos[i][0]
            start_point.y = self.current_sphere_pos[i][1]
            start_point.z = self.current_sphere_pos[i][2]
            marker.points.append(start_point)
            end_point = Point()
            end_point.x = vel_orient[0] + self.current_sphere_pos[i][0]
            end_point.y = vel_orient[1] + self.current_sphere_pos[i][1]
            end_point.z = vel_orient[2] + self.current_sphere_pos[i][2]
            marker.points.append(end_point)
            self.arrow_markers.markers.append(marker)
        
        self.marker_publisher.publish(self.arrow_markers)

    def updateRosMsg(self,visual_gradient = False):
        # pub robot_self_collision --------------------------------
        # robot_collision_cost = self.mpc_control.controller.rollout_fn \
        #                             .robot_self_collision_cost(self.curr_state_tensor.unsqueeze(0)[:,:,:7]) \
        #                             .squeeze().cpu().numpy()
        # self.coll_msg.data = robot_collision_cost
        # self.coll_robot_pub.publish(self.coll_msg)

        # pub env_pointcloud and robot_link_spheres --------------------------------
        # w_batch_link_spheres = self.mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.robot_coll.w_batch_link_spheres 
        # spheres = [s[-1*30][:, :3].cpu().numpy() for s in w_batch_link_spheres]
        # # 将所有球体位置信息合并为一个NumPy数组
        # robotsphere_positions = np.concatenate(spheres, axis=0)
        # self.pub_pointcloud(robotsphere_positions, self.pub_robot_link_pc)

        # pub collision_grid_map --------------------------------
        # self.pub_pointcloud(self.collision_grid.cpu().numpy() , self.pub_env_pc)
        self.pub_pointcloud(self.envpc_filter.scene_pc, self.pub_env_pc)
        if visual_gradient:
            # 发布两种箭头数据（current_vel_orient ,current_gradient）（这是因为current_vel_orient,current_gradient都是向量，需要将指向可视化），在rviz中显示。
            # 箭头的位置都由current_sphere_pos提供。应该如何做
            self.current_gradient = self.mpc_control.controller.rollout_fn.primitive_collision_cost.current_grad.cpu().numpy() # torch数据，shape为 7 * 3
            self.current_vel_orient = self.mpc_control.controller.rollout_fn.primitive_collision_cost.current_vel_orient.cpu().numpy() # torch数据，shape为 7 * 3
            self.current_sphere_pos = self.mpc_control.controller.rollout_fn.primitive_collision_cost.current_sphere_pos.cpu().numpy() # torch数据，shape为 7 * 3
            self.pub_arrow()

        self.mpc_command.header.stamp = rospy.Time.now()
        self.mpc_command.position = self.command['position']
        # self.mpc_command.velocity = self.command['velocity']
        # self.mpc_command.effort =   self.command['acceleration']
        self.command_pub.publish(self.mpc_command)

    def pub_joint_trajectory(self,trajectory , pub_handle):
        """
        传参的joint_trajectory is 30 * 7 
        start state : trajectory[0]
        trajectory is trajectory
        """
        # 装填数据之前 要将数据清零
        self.display_trajectory.trajectory = []
        self.robot_trajectory.joint_trajectory.points = []
        self.robot_trajectory.joint_trajectory.header.stamp = rospy.Time.now()
        self.trajectory_start.joint_state.position = trajectory[0]

        # 选择跳跃间隔，例如每隔一个时间步选择一次
        skip_interval = 2  # 每隔一个时间步选择一次

        for i in range(0, len(trajectory), skip_interval):
            joint_angles = trajectory[i]
            # 创建轨迹点
            point = JointTrajectoryPoint()
            point.positions = joint_angles
            self.robot_trajectory.joint_trajectory.points.append(point)

        # 将关节轨迹消息添加到显示轨迹消息中
        self.display_trajectory.trajectory.append(self.robot_trajectory)
        self.display_trajectory.trajectory_start = self.trajectory_start
        # 发布显示轨迹消息
        pub_handle.publish(self.display_trajectory)

    def pub_multi_joint_trajectory(self, mean_joint_visual_trajectory , sensi_joint_visual_trajectory, greedy_joint_visual_trajectory):
        self.pub_joint_trajectory(mean_joint_visual_trajectory, self.display_mean_trajectory_pub)
        self.pub_joint_trajectory(sensi_joint_visual_trajectory, self.display_sensi_trajectory_pub)
        self.pub_joint_trajectory(greedy_joint_visual_trajectory, self.display_greedy_trajectory_pub)