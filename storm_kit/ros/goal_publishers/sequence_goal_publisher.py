from copy import deepcopy
import numpy as np
import torch
import yaml

import rospy
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState
from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import *

from storm_kit.differentiable_robot_model import DifferentiableRobotModel
from storm_kit.differentiable_robot_model.coordinate_transform import matrix_to_quaternion


class SequenceGoalPublisher():
    def __init__(self):
        self.joint_states_topic = rospy.get_param('~joint_states_topic', 'joint_states')
        self.ee_goal_topic = rospy.get_param('~ee_goal_topic', 'ee_goal')
        self.goal_pub_freq = rospy.get_param('~goal_pub_freq', 10)
        self.fixed_frame = rospy.get_param('~fixed_frame', 'panda_link0')
        self.robot_urdf = '../../../content/assets/urdf/franka_description/franka_panda_no_gripper.urdf'
        self.ee_frame = rospy.get_param('~ee_frame', 'ee_link')
        self.goal_file = rospy.get_param('goal_list_file', './left_right_goal.yaml')

        with open(self.goal_file, "r") as f:
            try:
                self.goal_list = yaml.safe_load(f)['goal_list']
            except yaml.YAMLError as exc:
                print(exc)
        self.num_goals = len(self.goal_list)
        self.curr_goal_idx = 0
        self.goal_update_secs = 10

        #ROS Initialization
        self.ee_goal = PoseStamped()
        self.gripper_state = JointState()
        self.robot_state = JointState()

        self.ee_goal_pub = rospy.Publisher(self.ee_goal_topic, PoseStamped, queue_size=1, tcp_nodelay=True, latch=False)
        self.state_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.robot_state_callback, queue_size=1)

        #STORM Related
        self.tensor_args = {'device': 'cpu', 'dtype': torch.float32}
        self.robot_model = DifferentiableRobotModel(self.robot_urdf, None, 
                            tensor_args=self.tensor_args)
            
        #Buffers
        self.tstep = 0
        self.last_goal_tstep = 0.0
        self.start_t = None
        self.rate = rospy.Rate(self.goal_pub_freq)
        self.state_received = False
        while not self.state_received:
            pass
        #we set self.ee_goal to the initial robot pose
        self.update_ee_goal_to_current()

        self.setup_goal_marker()

    def marker_callback(self, msg):
        self.ee_goal.header = msg.header
        self.ee_goal.pose = deepcopy(msg.pose)

    def goal_pub_loop(self):
        while not rospy.is_shutdown():
            curr_goal = self.goal_list[self.curr_goal_idx]
            self.ee_goal.pose.position.x = curr_goal[0]
            self.ee_goal.pose.position.y = curr_goal[1]
            self.ee_goal.pose.position.z = curr_goal[2]

            self.ee_goal_pub.publish(self.ee_goal)

            #update goal idx based on some criterion
            if self.tstep == 0:
                self.start_t = rospy.get_time()
            
            self.tstep = rospy.get_time() - self.start_t
            
            if (self.tstep - self.last_goal_tstep) >= self.goal_update_secs :
                print('updating goal')
                self.curr_goal_idx = (self.curr_goal_idx + 1) % self.num_goals
                self.last_goal_tstep = deepcopy(self.tstep)  
            
            self.rate.sleep()
    
    def robot_state_callback(self, msg):
        self.state_received = True
        # save gripper state
        self.gripper_state.header = msg.header
        self.gripper_state.position = msg.position[0:2]
        self.gripper_state.velocity = msg.velocity[0:2]
        self.gripper_state.effort = msg.effort[0:2]

        #save robot state
        self.robot_state.header = msg.header
        self.robot_state.position = msg.position[2:]
        self.robot_state.velocity = msg.velocity[2:]
        self.robot_state.effort = msg.effort[2:]


    def get_ee_pose(self):
        q_robot = torch.as_tensor(self.robot_state.position, **self.tensor_args).unsqueeze(0)
        qd_robot = torch.as_tensor(self.robot_state.velocity, **self.tensor_args).unsqueeze(0)


        curr_ee_pos, curr_ee_rot = self.robot_model.compute_forward_kinematics(
            q_robot, qd_robot, link_name=self.ee_frame)
        curr_ee_quat = matrix_to_quaternion(curr_ee_rot)
        return curr_ee_pos, curr_ee_quat

    def update_ee_goal_to_current(self):
        q_robot = torch.as_tensor(self.robot_state.position, **self.tensor_args).unsqueeze(0)
        qd_robot = torch.as_tensor(self.robot_state.velocity, **self.tensor_args).unsqueeze(0)
        # q_gripper = torch.as_tensor(self.gripper_state.position, **self.tensor_args).unsqueeze(0)
        # qd_gripper = torch.as_tensor(self.gripper_state.velocity, **self.tensor_args).unsqueeze(0)
        # q = torch.cat((q_robot, q_gripper), dim=-1)
        # qd = torch.cat((qd_robot, qd_gripper), dim=-1)


        curr_ee_pos, curr_ee_rot = self.robot_model.compute_forward_kinematics(
            q_robot, qd_robot, link_name=self.ee_frame)
        curr_ee_quat = matrix_to_quaternion(curr_ee_rot)
        # self.curr_ee_quat = self.curr_ee_quat / torch.norm(self.curr_ee_quat) #normalize quaternion


        #convert to pose stamped message
        self.ee_goal.header.stamp = rospy.Time.now()
        self.ee_goal.pose.position.x = curr_ee_pos[0][0].item() 
        self.ee_goal.pose.position.y = curr_ee_pos[0][1].item() 
        self.ee_goal.pose.position.z = curr_ee_pos[0][2].item() 
        self.ee_goal.pose.orientation.w = curr_ee_quat[0][0].item() 
        self.ee_goal.pose.orientation.x = curr_ee_quat[0][1].item() 
        self.ee_goal.pose.orientation.y = curr_ee_quat[0][2].item() 
        self.ee_goal.pose.orientation.z = curr_ee_quat[0][3].item()
    
    def setup_goal_marker(self):
        pass


    def close(self):
        self.ee_goal_pub.unregister()
        self.state_sub.unregister()

if __name__ == "__main__":
    rospy.init_node("interactive_marker_goal_node", anonymous=True, disable_signals=True)    

    goal_node = SequenceGoalPublisher()

    try:
        goal_node.goal_pub_loop()
    except KeyboardInterrupt:
        print('Exiting')
        goal_node.close()
