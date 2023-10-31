""" Example spawning a robot on real-machine
只关心 运动规划问题 mpc with TrackIK_Guild
无碰撞
无SDF参与
无MultiModal
"""

from ReacherBase import ReacherEnvBase

import torch
import numpy as np
import rospy
from storm_kit.mpc.task import ReacherTask, ReacherTaskThread
from storm_ros.utils.tf_translation import get_world_T_cam
from storm_examples.multimodal_franka.utils import LimitedQueue , IKProc
import queue


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
        self.policy = ReacherTaskThread(self.mpc_config, self.world_description, self.tensor_args)
        # goal list control
        # self.goal_list = np.array([
        #      [0.45, -0.45, 0.45],
        #      [0.45,  0.45, 0.45],
        #      ])
        x,y,z = 0.30 , 0.55 , 0.30
        self.goal_list = [
             [x,y,z],
             [x,-y,z]]
        self.ee_goal_pos = self.goal_list[0]
        # self.policy.update_params(goal_ee_pos = self.ee_goal_pos,
        #                           goal_ee_quat = self.ee_goal_quat)  

        self.thresh = 0.03 # goal next thresh in Cart
        self.ik_mSolve = ik_mSolve
        self.goal_ee_transform = np.eye(4)
        self.rollout_fn = self.policy.controller.rollout_fn
        self.world_T_cam = get_world_T_cam() # transform : "world", "rgb_camera_link"
        self.ros_handle_init()

    def control_loop(self):
        rospy.loginfo('[MPCPoseReacher]: Controller running')
        start_t = rospy.get_time()
        lap_count = 5 # 跑5轮次
        self.jnq_des = np.zeros(7)
        opt_step_count = 0 
        opt_time_sum = 0 
        pointcloud_SDF_time_sum = 0
        GoalUpdate_last_sum = 0
        VisualUpdate_last_sum = 0
        self.goal_flagi = -1 # 调控目标点
        while not rospy.is_shutdown() and \
                self.goal_flagi / len(self.goal_list) != lap_count:
            try:
                opt_step_count += 1
                # TODO: scene_grid update at here ... 
                # 1. transform pointcloud to world frame 
                # 2. compute sdf from pointcloud 
                # 更新scene_grid 
                        # Assuming self.point_array is your original point cloud data
                # It should have shape (N, 3) where N is the number of points
                # Each row is a point (x, y, z)
                #TODO :  pointcloudSDF_TIME_COST : 3.8244729934141732
                pointcloud_SDF_time_last = rospy.get_time()
                point_array = np.hstack((self.point_array, np.ones((self.point_array.shape[0], 1))))  # Adding homogenous coordinate
                transformed_points = np.dot(point_array, self.world_T_cam.T)  # Transform all points at once
                self.point_array = transformed_points[:, :3]  # Removing the homogenous coordinate
                # mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll._compute_dynamic_sdfgrid(scene_pc)
                # pointcloud_SDF_time_last = rospy.get_time()
                self.collision_grid = self.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll. \
                                     _opt_compute_dynamic_voxeltosdf(self.point_array,visual = True)
                pointcloud_SDF_time_sum += rospy.get_time() - pointcloud_SDF_time_last

                tstep = rospy.get_time() - start_t 
                # 逆解获取请求发布 input_queue
                qinit = self.robot_state['position']
                self.goal_ee_transform[:3,3] = self.rollout_fn.goal_ee_pos.cpu().numpy()
                self.goal_ee_transform[:3,:3] = self.rollout_fn.goal_ee_rot.cpu().numpy()
                self.ik_mSolve.ik_procs[-1].ik(self.goal_ee_transform , qinit , ind = tstep)
                #get mpc command
                # TODO: tstep 与 control_dt的关系是什么？ 没有穿透
                opt_time_last = rospy.get_time()
                command = self.policy.get_command(
                    tstep, self.robot_state, control_dt=self.control_dt)
                opt_time_sum += rospy.get_time() - opt_time_last
                self.command = command
                q_des ,qd_des ,qdd_des = command['position'] ,command['velocity'] , command['acceleration']
                self.curr_state_tensor = torch.as_tensor(np.hstack((q_des,qd_des,qdd_des)), **self.tensor_args).unsqueeze(0) # "1 x 3*n_dof"

                #  TODO: costly : goalupdate_or_not is cost ! need change! GoalUpdate_last_sum : 3.297654991475945 
                self.GoalUpdate()
                #publish mpc command
                self.mpc_command.header.stamp = rospy.Time.now()
                self.mpc_command.position = q_des
                self.mpc_command.velocity = qd_des
                self.command_pub.publish(self.mpc_command)

                self.visual_top_trajs()
                self.traj_append()
                collision_grid_pc = self.collision_grid.cpu().numpy() 
                self.pub_pointcloud(collision_grid_pc, self.pub_env_pc)

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
                rospy.logerr("Error --- *~* ---")  
                break


        rospy.loginfo("whole_time: {}, opt_step_count: {}, collison_count: {}, "
                      "oneLoop: {}, oneOpt: {}, pointcloudSDF: {}, "
                      "generate_rollouts_time_sum: {}, FK_time_sum: {}, cost_time_sum: {} ,mpc_time_sum: {}".format(tstep, opt_step_count, self.curr_collision, 
                                                       tstep / opt_step_count * 1000, 
                                                       opt_time_sum / opt_step_count * 1000,
                                                       pointcloud_SDF_time_sum / opt_step_count * 1000,
                                                       self.policy.controller.generate_rollouts_time_sum / opt_step_count *1000,
                                                       self.rollout_fn.fk_time_sum / opt_step_count * 1000,
                                                       self.rollout_fn.cost_time_sum / opt_step_count * 1000,
                                                       self.policy.control_process.mpc_time_sum / opt_step_count * 1000))   
        
        self.close()
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
        mpc_node.control_loop()
    except KeyboardInterrupt:
        print('Exiting')
        mpc_node.close()