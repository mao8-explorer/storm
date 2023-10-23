import rospy
from moveit_msgs.msg import DisplayTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory , RobotState

def publish_trajectory(trajectory):
    rospy.init_node('trajectory_publisher')
    display_trajectory_pub = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory, queue_size=10)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        display_trajectory = DisplayTrajectory()
        display_trajectory.model_id = "panda"  # 替换为你的机器人模型ID

        # 创建关节轨迹消息
        robot_trajectory = RobotTrajectory()
        robot_trajectory.joint_trajectory.header.stamp = rospy.Time.now()
        robot_trajectory.joint_trajectory.header.frame_id = "panda_link0"
        robot_trajectory.joint_trajectory.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        
        trajectory_start = RobotState()
        trajectory_start.joint_state.header.frame_id = "panda_link0"
        trajectory_start.joint_state.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        trajectory_start.joint_state.position = trajectory[0]


        for joint_angles in trajectory:
            # 创建轨迹点
            point = JointTrajectoryPoint()
            point.positions = joint_angles
            # point.time_from_start = rospy.Duration(1.0)  # 替换为你的轨迹点的时间戳

            # 将轨迹点添加到关节轨迹消息中
            robot_trajectory.joint_trajectory.points.append(point)

        # 将关节轨迹消息添加到显示轨迹消息中
        display_trajectory.trajectory.append(robot_trajectory)
        display_trajectory.trajectory_start = trajectory_start
        # 发布显示轨迹消息
        display_trajectory_pub.publish(display_trajectory)
        rate.sleep()

if __name__ == '__main__':
    trajectory = [
        [0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483],  # 第一个时刻的关节角度
        [-0.004579455706702351, -0.8071287210600578, -0.011819738545822498, -2.368345153309152, -0.008537755298979115, 1.5612516506010075, 0.7725628302072198],
        [-0.1878336619348481, -1.2387967069135497, -0.47331510240376545, -2.4116236845016825, -0.4719012630469172, 1.2469251848943579, 0.27115420307573096],
        [-0.19814099802502802, -1.2507709331108077, -0.5359972887683253, -2.3726385344782304, -0.5418981467517183, 1.2223571208028963, 0.19979814090394868],
        [-0.2018159677557885, -1.2546355453196758, -0.5823928277782349, -2.337673610909734, -0.5936277662661397, 1.2070005138778626, 0.1459478855331385],
        [-0.20183053548312532, -1.254576229301709, -0.6281272024947062, -2.2979472898231856, -0.6442381991732087, 1.1941688686304823, 0.09190712967802102],
        [-0.19974747232555176, -1.2525570871559024, -0.6582331436621545, -2.2688665203159006, -0.6771844984792915, 1.1869504897310423, 0.055797608366439155]
        # ... 添加更多的关节角度
    ]
    publish_trajectory(trajectory)
 