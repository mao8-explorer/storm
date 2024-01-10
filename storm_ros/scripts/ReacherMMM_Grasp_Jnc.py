""" Example spawning a robot on real-machine
加入抓取物体的功能 尝试将动态避障融入到整个抓取的行为中
stage 1 :
    goto Pre_Grasp POSE 
    call for grasp_detect
    wait for grasp_detect_result

stage 2 :
    goto Grasp POSE
    close gripper 

stage 3 :
    goto Place_Object POSE
    open gripper
"""

from tkinter import Place
from ReacherBase_grasp import ReacherEnvBase
from pyparsing import PrecededBy
import torch
import numpy as np
import rospy
from std_msgs.msg import String , Float32
from geometry_msgs.msg import Pose, Point, Quaternion
from storm_kit.mpc.task import ReacherTaskRealMultiModal
from storm_examples.multimodal_franka.utils import LimitedQueue , IKProc
import queue
import time
import copy
from panda_grasp import PandaCommander


class IKSolve:
    def __init__(self):
        self.num_proc = 1
        self.maxsize = 1
        self.output_queue = LimitedQueue(self.maxsize)
        self.ik_procs = []
        for _ in range(self.num_proc):
            self.ik_procs.append(
                IKProc(
                    self.output_queue,
                    input_queue_maxsize=self.maxsize,
                )
            )
            self.ik_procs[-1].daemon = True #守护进程 主进程结束 IKProc进程随之结束
            self.ik_procs[-1].start()    

class MPCReacherNode(ReacherEnvBase):
    def __init__(self, ik_mSolve):
        super().__init__()
        #STORM Initialization
        # self.policy = ReacherTask(self.mpc_config, self.world_description, self.tensor_args)
        self.policy = ReacherTaskRealMultiModal(self.mpc_config, self.world_description, self.tensor_args)
        self.goal_list = [
             [0.40, -0.45,  0.20],
             [0.35, 0.40, 0.50]]
        self.Place_Object_POSE = self.goal_list[0]
        self.Pre_Grasp_POSE = self.goal_list[1]

        self.thresh = 0.020              # goal next thresh in Cart
        self.ik_mSolve = ik_mSolve
        self.goal_ee_transform = np.eye(4)
        self.rollout_fn = self.policy.controller.rollout_fn
        # self.world_T_cam = get_world_T_cam() # transform : "world", "rgb_camera_link"
        self.ee_goal_pos = self.Place_Object_POSE
        self.ros_handle_init()
        self.panda_grasp = PandaCommander()
        self.panda_grasp.move_gripper(0.08) # initial gripper and open
        self.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll. \
                            TEMP_opt_compute_dynamic_voxeltosdf()
        self.jnq_des = np.zeros(7)
        rospy.Subscriber('/grasp_transform', Pose, self.grasp_response_callback, queue_size=1)

    def grasp_response_callback(self, pose):
        self.grasp_Target = pose

    def angle_distance(self, q):
        dot_product = np.dot(self.q_home, q)
        squared_distance = 2 * (dot_product**2) - 1
        distance = np.arccos(squared_distance)
        return distance

    def control_loop(self):
        """
        Reacher 模组化 
        """
        opt_step_count = 0
        opt_time_sum = 0  
        last_shape = 0
        start_time = time.time()
        pointcloud_SDF_time_sum = 0 # calculate voxel to sdf SUM time
        while not rospy.is_shutdown() :
            try:
                opt_step_count += 1
                pointcloud_SDF_time_last = rospy.get_time()
                if  self.point_array.shape[0] > 0 and abs(self.point_array.shape[0] - last_shape) > 3: # 点云dt相似度 ，没必要每次都更新代价地图 简单的相似度阈值限制 在计算量与判断上做平衡 如何简单有效的判断点云变化程度？需要平衡
                    self.collision_grid = self.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll. \
                                        _opt_compute_dynamic_voxeltosdf(self.point_array)
                    last_shape = self.point_array.shape[0]
                    # rospy.logwarn("update-->")
                pointcloud_SDF_time_sum += rospy.get_time() - pointcloud_SDF_time_last
          
                # 逆解获取请求发布 input_queue
                qinit = self.robot_state['position']
                self.goal_ee_transform[:3,3] =  self.rollout_fn.goal_ee_pos.cpu().detach().numpy()
                self.goal_ee_transform[:3,:3] = self.rollout_fn.goal_ee_rot.cpu().detach().numpy()
                self.ik_mSolve.ik_procs[-1].ik(self.goal_ee_transform , qinit , ind = opt_step_count)
                opt_time_last = time.time()
                command = self.policy.get_real_multimodal_command(self.robot_state)
                opt_time_sum += time.time() - opt_time_last
                self.command = command
                q_des ,qd_des = command['position'] ,command['velocity']
                #publish mpc command
                self.mpc_command.header.stamp = rospy.Time.now()
                self.mpc_command.position = q_des
                self.mpc_command.velocity = qd_des
                self.command_pub.publish(self.mpc_command)
                if torch.norm(self.rollout_fn.goal_ee_pos - self.rollout_fn.curr_ee_pos) < self.thresh:
                    end_time = time.time() - start_time
                    rospy.loginfo("oneLoop: {:.3f}, oneOpt: {:.3f}, pointcloudSDF: {:.3f}".format(
                                        end_time / opt_step_count * 1000,
                                        opt_time_sum / opt_step_count * 1000, 
                                        pointcloud_SDF_time_sum / opt_step_count * 1000))
                    break

                # self.visual_top_trajs_multimodal()
                self.visual_multiTraj()
                self.traj_append()
                self.traj_append_multimodal()
                # 逆解获取查询 output_queue
                try :
                    output = self.ik_mSolve.output_queue.get()
                    if output[1] is not None: # 无解
                        self.rollout_fn.goal_jnq = torch.as_tensor(output[1], **self.tensor_args).unsqueeze(0) # 1 x n_dof
                        self.jnq_des = output[1]
                        # rospy.loginfo("output : {}, command['position'] : {} , error : {}".format(output[1],q_des, (output[1]-q_des)*57.3 ) )
                    else : 
                        self.rollout_fn.goal_jnq = None
                        self.jnq_des = np.zeros(7)
                        rospy.logwarn("warning: no iksolve")
                except queue.Empty:
                    "针对 output_queue队列为空的问题 会出现queue.Empty的情况发生"
                    continue

            except KeyboardInterrupt:
                rospy.logerr("Control Loop Error --- *~* ---")  
                return False
        
        return True
    
    def Grasp_Loop(self):
        """
        stage 1 :
            goto Pre_Grasp POSE 
            call for grasp_detect
            wait for grasp_detect_response

        stage 2 :
            goto Grasp POSE
            close gripper 

        stage 3 :
            goto Place_Object POSE
            open gripper
        """

        Home_Pose = Pose() 
        Pre_Grasp = Pose()
        Place_Object = Pose()


        Home_Pose.position = Point(x=0.31, y=0.02, z=0.56) 
        HomeQuat = Quaternion(x=-0.015399, y=0.670534, z=0.00638, w=0.741)
        self.q_home = np.array([HomeQuat.w, HomeQuat.x, HomeQuat.y, HomeQuat.z])
        Home_Pose.orientation = HomeQuat
        
        Pre_Grasp.position = Point(*np.array(self.Pre_Grasp_POSE))
        Pre_Grasp.orientation = HomeQuat

        Place_Object.position = Point(*np.array(self.Place_Object_POSE))
        Place_Object.orientation = HomeQuat

        self.Grasp_update_Pose(Home_Pose)
        if not self.control_loop() : return False
        while not rospy.is_shutdown():
            try:
                self.traj_log = {'position':[], 'velocity':[], 'acc':[] , 'des':[] , 'weights':[] , 'robot_position': [], 'robot_velocity': []}
                # 1.1 goto Pre_Grasp POSE 
                self.Grasp_update_Pose(Pre_Grasp)
                if not self.control_loop() : break

                # 1.2 subscribe grasp pose and filp_or_not to modify gripper
                grasp_pose = copy.deepcopy(self.grasp_Target)
                q_origin = np.array([grasp_pose.orientation.w, grasp_pose.orientation.x, grasp_pose.orientation.y, grasp_pose.orientation.z])
                q_flip = np.array([-grasp_pose.orientation.x, grasp_pose.orientation.w, grasp_pose.orientation.z, -grasp_pose.orientation.y])
                final_q = q_origin if self.angle_distance(q_origin) < self.angle_distance(q_flip) else q_flip
                grasp_pose.orientation.w , grasp_pose.orientation.x , grasp_pose.orientation.y , grasp_pose.orientation.z = final_q 
                pre_grasp_pose = copy.deepcopy(grasp_pose)
                pre_grasp_pose.position.z = 0.35

                # 2.1 goto Grasp POSE
                self.thresh = 0.030
                self.Grasp_update_Pose(pre_grasp_pose)
                if not self.control_loop() : break
                self.thresh = 0.015
                self.Grasp_update_Pose(grasp_pose)
                if not self.control_loop() : break
                # 2.2 close gripper
                self.panda_grasp.grasp(width=0.01, force=0.05, timeout=rospy.Duration(1.0)) 

                self.thresh = 0.05
                self.Grasp_update_Pose(pre_grasp_pose)
                if not self.control_loop() : break
                self.thresh = 0.05
                self.Grasp_update_Pose(Pre_Grasp)
                if not self.control_loop() : break

                # 3.1 goto Place_Object POSE
                self.thresh = 0.05
                self.Grasp_update_Pose(Place_Object)
                if not self.control_loop() : break
                # 3.2 open gripper
                self.panda_grasp.move_gripper(0.08, timeout=rospy.Duration(1.0))

            except KeyboardInterrupt:
                rospy.logerr("Grasp Loop Error --- *~* ---") 
                break
        
        
        self.Grasp_update_Pose(Home_Pose)
        self.control_loop()
        self.close()
        self.panda_grasp.move_gripper(0.08,timeout=rospy.Duration(5.0)) # exit gripper and set open state
        self.plot_traj_multimodal()
        self.plot_traj()
        rospy.loginfo("Closing ---all ---")
    


if __name__ == "__main__":

    ik_mSolve = IKSolve() # 多进程的问题 （应该是没有正确的解决 含有糊弄的成分 主要就像要让 IKProc在主进程启动 同时 在spawn之前启动）
    torch.multiprocessing.set_start_method('spawn', force=True)
    # torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rospy.init_node("mpc_reacher_node", anonymous=True, disable_signals=True)    
    mpc_node = MPCReacherNode(ik_mSolve)

    try:
        mpc_node.Grasp_Loop()
    except KeyboardInterrupt:
        print('Exiting')
        mpc_node.close()
        
    mpc_node.close()