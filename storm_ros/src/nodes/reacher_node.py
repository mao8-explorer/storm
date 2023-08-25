#!/usr/bin/env python

#General imports
import os
import numpy as np

from isaacgym import gymapi
from isaacgym import gymutil

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#ROS Imports
import rospy
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState
import rospkg
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField

# from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
# from std_msgs.msg import String, Header

#STORM imports
from storm_kit.mpc.task.reacher_task import ReacherTask

#marker init
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
import yaml

class MPCReacherNode():
    def __init__(self) -> None:
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

        self.pub_env_pc = rospy.Publisher('env_pc', PointCloud2, queue_size=5)


    
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
        #save gripper state
        # self.gripper_state.header = msg.header
        # self.gripper_state.position = msg.position[0:2]
        # self.gripper_state.velocity = msg.velocity[0:2]
        # self.gripper_state.effort = msg.effort[0:2]

        # self.gripper_state['position'] = np.array(msg.position[0:2])
        # self.gripper_state['velocity'] = np.array(msg.velocity[0:2])

        # #save robot state
        # self.robot_state.header = msg.header
        # self.robot_state.position = msg.position[2:]
        # self.robot_state.velocity = msg.velocity[2:]
        # self.robot_state.effort = msg.effort[2:]
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
        point_list = list(point_generator)
        self.point_array = np.array(point_list)

        # point_array is now a NumPy array of shape (N, 3), containing the x, y, z coordinates of the points
        # You can use point_array as needed for further processing or visualization


    def control_loop(self):

        msg = PointCloud2()
        msg.header.frame_id = "world"
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.is_dense = False
        while not rospy.is_shutdown():
            #only do something if state and goal have been received
            if self.state_sub_on and self.goal_sub_on:
                #check if goal was updated
                if self.new_ee_goal:
                    self.policy.update_params(goal_ee_pos = self.ee_goal_pos,
                        goal_ee_quat = self.ee_goal_quat)
                    self.new_ee_goal = False

                # 更新scene_grid 
                
                # mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll._compute_dynamic_sdfgrid(scene_pc)
                collision_grid = self.policy.controller.rollout_fn. \
                                    primitive_collision_cost.robot_world_coll.world_coll. \
                                    _compute_dynamic_voxeltosdf(self.point_array, visual = True)
                
                collision_grid_pc = collision_grid.cpu().numpy() 
                msg.header.stamp = rospy.Time().now()
                if len(collision_grid_pc.shape) == 3:
                    msg.height = collision_grid_pc.shape[1]
                    msg.width = collision_grid_pc.shape[0]
                else:
                    msg.height = 1
                    msg.width = len(collision_grid_pc)

                msg.row_step = msg.point_step * collision_grid_pc.shape[0]
                msg.data = np.asarray(collision_grid_pc, np.float32).tostring()

                self.pub_env_pc.publish(msg)

                #get mpc command
                command = self.policy.get_command(
                    self.tstep, self.robot_state, control_dt=self.control_dt, WAIT=True)
                

                #publish mpc command
                self.mpc_command.header = self.command_header
                self.mpc_command.header.stamp = rospy.Time.now()
                self.mpc_command.position = command['position']
                self.mpc_command.velocity = command['velocity']
                self.mpc_command.effort =  command['acceleration']
                self.command_pub.publish(self.mpc_command)

                #update tstep
                if self.tstep == 0:
                    rospy.loginfo('[MPCPoseReacher]: Controller running')
                    self.start_t = rospy.get_time()
                self.tstep = rospy.get_time() - self.start_t

          

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