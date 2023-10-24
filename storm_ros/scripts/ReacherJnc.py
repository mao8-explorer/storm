from ReacherBase import ReacherEnvBase
import torch
import numpy as np
import rospy
from storm_kit.mpc.task.reacher_task import ReacherTask
from examples.multimodal_franka.utils import LimitedQueue , IKProc
import queue


class IKSolve:
    def __init__(self):
        self.num_proc = 1
        self.maxsize = 5
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
        self.policy = ReacherTask(self.mpc_config, self.world_description, self.tensor_args)
        self.ik_mSolve = ik_mSolve
        self.goal_ee_transform = np.eye(4)
        self.rollout_fn = self.policy.controller.rollout_fn
        
    def control_loop(self):
        self._initialize_rospy()
        rospy.loginfo('[MPCPoseReacher]: Controller running')
        start_t = rospy.get_time()

        while not rospy.is_shutdown():
            try:
                #only do something if state and goal have been received
                if self.State_Sub_On and self.Goal_Sub_On:
                    #check if goal was updated
                    if self.New_EE_Goal:
                        self.policy.update_params(goal_ee_pos = self.ee_goal_pos,
                                                goal_ee_quat = self.ee_goal_quat)
                        self.New_EE_Goal = False

                    # TODO: scene_grid update at here ... 
                    # 1. transform pointcloud to world frame 
                    # 2. compute sdf from pointcloud 
                    tstep = rospy.get_time() - start_t

                    # 逆解获取请求发布 input_queue
                    qinit = self.robot_state['position']
                    self.goal_ee_transform[:3,3] = self.rollout_fn.goal_ee_pos.cpu().numpy()
                    self.goal_ee_transform[:3,:3] = self.rollout_fn.goal_ee_rot.cpu().numpy()
                    self.ik_mSolve.ik_procs[-1].ik(self.goal_ee_transform , qinit , ind = tstep)
                    #get mpc command
                    # TODO: tstep 与 control_dt的关系是什么？ 没有穿透
                    command = self.policy.get_command(
                        tstep, self.robot_state, control_dt=self.control_dt)
                    #publish mpc command
                    self.mpc_command.header.stamp = rospy.Time.now()
                    self.mpc_command.position = command['position']
                    self.mpc_command.velocity = command['velocity']
                    self.command_pub.publish(self.mpc_command)

                    self.visual_top_trajs()
                    # 逆解获取查询 output_queue
                    try :
                        output = self.ik_mSolve.output_queue.get()
                        if output[1] is not None: # 无解
                            self.rollout_fn.goal_jnq = torch.as_tensor(output[1], **self.tensor_args).unsqueeze(0) # 1 x n_dof
                            self.jnq_des = output[1]
                        else : 
                            self.rollout_fn.goal_jnq = None
                            self.jnq_des = np.zeros(7)
                            print("warning: no iksolve")
                    except queue.Empty:
                        "针对 output_queue队列为空的问题 会出现queue.Empty的情况发生"
                        continue

                else:
                    if not self.State_Sub_On:
                        rospy.logwarn('[MPCPoseReacher]: Waiting for robot state.')
                        rospy.sleep(0.5)
                        continue
                    if not self.Goal_Sub_On:
                        rospy.logwarn('[MPCPoseReacher]: Waiting for ee goal.')
                        rospy.sleep(0.5)

            except KeyboardInterrupt:
                rospy.logerr("Error --- *~* ---")  

        self.close()
        rospy.loginfo("Closing ---all ---")
    



if __name__ == "__main__":

    ik_mSolve = IKSolve() # 多进程的问题 （应该是没有正确的解决 含有糊弄的成分 主要就像要让 IKProc在主进程启动 同时 在spawn之前启动）
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_num_threads(8)
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