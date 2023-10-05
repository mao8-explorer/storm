""" Example spawning a robot in gym 

"""
from FrankaEnvBase import FrankaEnvBase
from utils import LimitedQueue , IKProc
import torch
import numpy as np
from storm_kit.gym.core import Gym
from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml
import rospy

class IKsolver:
    def __init__(self):
        # glabal Joint_des 进程设计
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


class MPCRobotController(FrankaEnvBase):
    def __init__(self, gym_instance,iksover):
        super().__init__(gym_instance = gym_instance)
        self._environment_init()
        x,z,y = 0.50 , 0.40 , 0.519
        self.goal_list = [
             [x,y,-z],
             [x,y,z],
             [-x,y,z],
             [-x,y,-z]]
        self.goal_state = self.goal_list[0]
        self.update_goal_state()
        self.rollout_fn = self.mpc_control.controller.rollout_fn
        self.goal_ee_transform = np.eye(4)
        self.iksover = iksover

    def run(self):
        self.goal_flagi = 0 # 调控目标点
        sim_dt = self.mpc_control.exp_params['control_dt']
        t_step = gym_instance.get_sim_time()
        while not rospy.is_shutdown():
            try:
                self.gym_instance.step()
                self.gym_instance.clear_lines()
                # monitor ee_pose_gym and update goal_param_mpc
                self.monitorGoalupdate()
                # seed goal to MPC_Policy _ get Command
                t_step += sim_dt
                current_robot_state = self.robot_sim.get_state(self.env_ptr, self.robot_ptr) # "dict: pos | vel | acc"

                qinit = current_robot_state['position'] # shape is (7,)
                self.goal_ee_transform[:3,3] = self.rollout_fn.goal_ee_pos
                self.goal_ee_transform[:3,:3] = self.rollout_fn.goal_ee_rot

                command = self.mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
                # get position command:
                q_des ,qd_des ,qdd_des = command['position'] ,command['velocity'] , command['acceleration']
                self.curr_state_tensor = torch.as_tensor(np.hstack((q_des,qd_des,qdd_des)), **self.tensor_args).unsqueeze(0) # "1 x 3*n_dof"
                # trans ee_pose in robot_coordinate to world coordinate
                self.updateGymVisual_GoalUpdate()
                # Command_Robot_State include keyboard control : SPACE For Pause | ESCAPE For Exit 
                successed = self.robot_sim.command_robot_state(q_des, qd_des, self.env_ptr, self.robot_ptr)
                if not successed : break 

            except KeyboardInterrupt:
                print('Closing')

        self.mpc_control.close()
        self.coll_robot_pub.unregister() 
        self.pub_env_pc.unregister()
        self.pub_robot_link_pc.unregister()
        print("mpc_close...")
        
if __name__ == '__main__':

    rospy.init_node('pointcloud_publisher_node')

    sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
    sim_params['headless'] = False
    gym_instance = Gym(**sim_params)

    iksover = IKsolver()
    controller = MPCRobotController(gym_instance,iksover)
    
    controller.run()
