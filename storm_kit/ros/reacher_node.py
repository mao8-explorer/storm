#General imports
import os
import numpy as np
import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#ROS Imports
import rospy
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import JointState, PointCloud2
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Header

#STORM imports
from storm_kit.mpc.task.reacher_task import ReacherTask


class MPCReacherNode():
    def __init__(self) -> None:
        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.joint_command_topic = rospy.get_param('~joint_command_topic', 'franka_motion_control/joint_command')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.pointcloud_topic = rospy.get_param('~pointcloud_topic', '/camera/depth/color/points_filtered')
        self.diagnostics_topic = rospy.get_param('~diagnostics_topic', 'mpc_diagnostics')
        self.fixed_frame = rospy.get_param('~fixed_frame', 'panda_link0')
        self.pointcloud_frame = rospy.get_param('~pointcloud_frame', '/camera/depth/color/points_filtered')
        self.joint_names = rospy.get_param('robot_joint_names', None)
        self.world_description = rospy.get_param('world_description', os.path.abspath('../../content/configs/gym/collision_table.yml'))
        self.mpc_config = rospy.get_param('mpc_config', os.path.abspath('../../content/configs/mpc/franka_reacher_real_robot.yml'))
        self.device = torch.device('cuda', 0)
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}

        #STORM Initialization
        self.robot_coll_description = os.path.abspath('../../content/configs/robot/franka_real_robot.yml')
        self.policy = ReacherTask(self.mpc_config, self.robot_coll_description, self.world_description, self.tensor_args)

        #ROS Initialization
        self.command_pub = rospy.Publisher(self.joint_command_topic, JointState, queue_size=1, tcp_nodelay=True, latch=False)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)
        self.ee_goal_sub = rospy.Subscriber(self.ee_goal_topic, PoseStamped, self.ee_goal_callback, queue_size=1)
        self.control_freq = 50.0
        self.rate = rospy.Rate(self.control_freq)

        #buffers for different messages
        self.mpc_command = JointState()
        self.mpc_command.name = self.joint_names
        self.mpc_command.effort = np.zeros(7)
        self.gripper_state = JointState()
        self.robot_state = JointState()
        self.ee_goal_pos = None
        self.ee_goal_quat = None

        self.state_sub_on = False
        self.goal_sub_on = False
        self.tstep = 0
        self.start_t = None

    def robot_state_callback(self, msg):
        self.state_sub_on = True

        #save gripper state
        self.gripper_state.header = msg.header
        self.gripper_state.position = msg.position[0:2]
        self.gripper_state.velocity = msg.velocity[0:2]
        self.gripper_state.effort = msg.effort[0:2]

        #save robot state
        self.robot_state.header = msg.header
        self.robot_state.position = msg.position[2:]
        self.robot_state.velocity = msg.velocity[2:]
        self.robot_state.effort = msg.effort[2:]

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


    def control_loop(self):
        while not rospy.is_shutdown():
            #only do something if state and goal have been received
            if self.state_sub_on and self.goal_sub_on:
                #check if goal was updated
                if self.new_ee_goal:
                    self.policy.update_params(goal_ee_pos = self.ee_goal_pos,
                        goal_ee_quat = self.ee_goal_quat)
                    self.new_ee_goal = False
                    pass

                if self.tstep == 0:
                    self.start_t = rospy.get_time()
                self.tstep = rospy.get_time() - self.start_t

            self.rate.sleep()
    
    def close(self):
        pass


if __name__ == "__main__":
    rospy.init_node("mpc_reacher_node", anonymous=True, disable_signals=True)    

    mpc_node = MPCReacherNode()

    try:
        mpc_node.control_loop()
    except KeyboardInterrupt:
        print('Exiting')
        mpc_node.close()