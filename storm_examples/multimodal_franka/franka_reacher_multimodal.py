""" Example spawning a robot in gym 

"""
from FrankaEnvBase import FrankaEnvBase
from FilterPointCloud import FilterPointCloud
import copy
import torch
import argparse
import numpy as np
from storm_kit.gym.core import Gym
from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml
import rospy

class MPCRobotController(FrankaEnvBase):
    def __init__(self, gym_instance):
        super().__init__(gym_instance = gym_instance)
        # 实验： dynamic object moveDesign
        # 首先获取物体当前位姿 并将其转到robot坐标系下
        move_pose = self.world_instance.get_pose(self.collision_body_handle)
        self.move_pose = self.w_T_r.inverse() * move_pose
        # 已知 物体在robot坐标系下的移动边界值为 bounds
        self.move_bounds = self.move_pose.p.x
        # 根据两个边界点 确定物体移动方向  这一段代码有误 typeError: unsupported operand type(s) for -: 'list' and 'list'
        self.velocity_vector = np.array([[0.707, 0.707 , 0.  ]])
   
    def run(self):
        sim_dt = self.mpc_control.exp_params['control_dt']
        t_step = gym_instance.get_sim_time()
        envpc_filter = FilterPointCloud(self.robot_sim.camObsHandle.cam_pose) #sceneCollisionNet 句柄 现在只是用来获取点云
        obs = {}
        while not rospy.is_shutdown():
            try:
                self.gym_instance.step()
                self.gym_instance.clear_lines()
                # generate pointcloud
                self.robot_sim.updateCamImage()
                obs.update(self.robot_sim.ImageToPointCloud()) #耗时大！
                envpc_filter._update_state(obs) 
                # compute pointcloud to sdf_map
                # mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll._compute_dynamic_sdfgrid(scene_pc)
                self.collision_grid = self.mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll. \
                                     _compute_dynamic_voxeltosdf(envpc_filter.cur_scene_pc, visual = True)
            
                # monitor ee_pose_gym and update goal_param_mpc
                self.monitorGoalupdate()
                # seed goal to MPC_Policy _ get Command
                t_step += sim_dt
                current_robot_state = self.robot_sim.get_state(self.env_ptr, self.robot_ptr) # "dict: pos | vel | acc"
                curr_state = np.hstack((current_robot_state['position'], current_robot_state['velocity'], current_robot_state['acceleration']))
                self.curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0) # "1 x 3*n_dof"
                command = self.mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
                # get position command:
                q_des ,qd_des = command['position'] ,command['velocity']
                # trans ee_pose in robot_coordinate to world coordinate
                self.updateGymVisual()
                self.updateRosMsg()
                # Command_Robot_State include keyboard control : SPACE For Pause | ESCAPE For Exit 
                successed = self.robot_sim.command_robot_state(q_des, qd_des, self.env_ptr, self.robot_ptr)
                if not successed : break 
                # 实验： dynamic object moveDesign
                # actor  :  collision_obj_base_handle 
                # self._dynamic_object_moveDesign()


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

    controller = MPCRobotController(gym_instance)
    controller.run()
