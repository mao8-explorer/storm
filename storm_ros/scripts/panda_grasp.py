import actionlib
import franka_gripper.msg
import rospy
import sensor_msgs.msg

class PandaCommander(object):
    def __init__(self):
        self._connect_to_gripper()
        rospy.loginfo("PandaGripper ready")

    def _connect_to_gripper(self):
        self.grasp_client = actionlib.SimpleActionClient(
            "/franka_gripper/grasp", franka_gripper.msg.GraspAction
        )
        self.grasp_client.wait_for_server()
        rospy.loginfo("Connected to grasp action server")
        self.move_client = actionlib.SimpleActionClient(
            "/franka_gripper/move", franka_gripper.msg.MoveAction
        )
        self.move_client.wait_for_server()
        rospy.loginfo("Connected to move action server")

        # gripper_width
        rospy.Subscriber(
            "/franka_gripper/joint_states", sensor_msgs.msg.JointState, self.joints_cb, queue_size=1
        )

    def joints_cb(self, msg):
        self.gripper_width = msg.position[0] + msg.position[1]

    def grasp(self, width=0.0, e_inner=0.1, e_outer=0.1, speed=0.1, force=10.0):
        epsilon = franka_gripper.msg.GraspEpsilon(e_inner, e_outer)
        goal = franka_gripper.msg.GraspGoal(width, epsilon, speed, force)
        self.grasp_client.send_goal(goal)
        return self.grasp_client.wait_for_result(rospy.Duration(2.0))

    def move_gripper(self, width, speed=0.1):
        goal = franka_gripper.msg.MoveGoal(width, speed)
        self.move_client.send_goal(goal)
        return self.move_client.wait_for_result(rospy.Duration(2.0))


if __name__ == "__main__":
    rospy.init_node("panda_grasp")
    panda_grasp = PandaCommander()
    while not rospy.is_shutdown():
        panda_grasp.move_gripper(0.08)
        rospy.loginfo("move_gripper width: {}".format(panda_grasp.gripper_width))
        panda_grasp.grasp(width=0.01, force=0.05)
        rospy.loginfo("grasp width: {}".format(panda_grasp.gripper_width))
        panda_grasp.move_gripper(0.08)
        rospy.loginfo("move_gripper width: {}".format(panda_grasp.gripper_width))
