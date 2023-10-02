""" Example spawning a robot in gym 

"""
from setfrankaEnv import setfrankaEnv
from filterPointCloud import filterPointCloud
import copy
from isaacgym import gymapi
import torch
import argparse
import numpy as np
from storm_kit.gym.core import Gym
from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml

import rospy

class MPCRobotController(setfrankaEnv):
    def __init__(self, args, gym_instance):
        super().__init__(args = args, gym_instance = gym_instance)
        # 实验： dynamic object moveDesign
        # 首先获取物体当前位姿 并将其转到robot坐标系下
        move_pose = copy.deepcopy(self.world_instance.get_pose(self.collision_body_handle))
        self.move_pose = copy.deepcopy(self.w_T_r.inverse() * move_pose)
        # 已知 物体在robot坐标系下的移动边界值为 bounds
        self.move_bounds = np.array([[-0.58, -0.57,  0.63],[0.58, 0.57,  0.63]])
        # 根据两个边界点 确定物体移动方向  这一段代码有误 typeError: unsupported operand type(s) for -: 'list' and 'list'
        velocity_vector = np.array([self.move_bounds[1] - self.move_bounds[0]])
        # 将 方向向量 velocity_vector 归一化
        self.velocity_vector = velocity_vector / np.linalg.norm(velocity_vector)
   
    def run(self):
        g_pos = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        g_q = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())



        sim_dt = self.mpc_control.exp_params['control_dt']
        q_des = None
        qd_des = None
        t_step = gym_instance.get_sim_time()

        envpc_filter = filterPointCloud(self.robot_sim.camObsHandle.cam_pose) #sceneCollisionNet 句柄 现在只是用来获取点云

        obs = {}
        i = 0
        while not rospy.is_shutdown():
            try:
                i += 1
                self.gym_instance.step()
                self.gym_instance.clear_lines()

                # get_env_pointcloud
                self.robot_sim.updateCamImage()
                obs.update(self.robot_sim.ImageToPointCloud()) #耗时大！
                envpc_filter._update_state(obs) 

                # compute pointcloud to sdf_map
                # mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll._compute_dynamic_sdfgrid(scene_pc)
                collision_grid = self.mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll._compute_dynamic_voxeltosdf(envpc_filter.cur_scene_pc, visual = True)

                pose = copy.deepcopy(self.world_instance.get_pose(self.target_body_handle))
                pose = copy.deepcopy(self.w_T_r.inverse() * pose) #将world坐标系下的目标点转到robot坐标系下

                if (np.linalg.norm(g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (
                        np.linalg.norm(g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z])) > 0.0)):
                    g_pos[0] = pose.p.x
                    g_pos[1] = pose.p.y
                    g_pos[2] = pose.p.z
                    g_q[1] = pose.r.x   
                    g_q[2] = pose.r.y
                    g_q[3] = pose.r.z
                    g_q[0] = pose.r.w
                    self.mpc_control.update_params(goal_ee_pos=g_pos,goal_ee_quat=g_q)
                
                # seed goal to MPC_Policy _ get Command
                t_step += sim_dt
                current_robot_state = copy.deepcopy(self.robot_sim.get_state(self.env_ptr, self.robot_ptr))
                command = self.mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
                # get position command:
                q_des = copy.deepcopy(command['position'])
                qd_des = copy.deepcopy(command['velocity'])  # * 0.5

                # current state modify
                filtered_state_mpc = current_robot_state  # mpc_control.current_state
                curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
                curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0)

                robot_collision_cost = self.mpc_control.controller.rollout_fn \
                                            .robot_self_collision_cost(curr_state_tensor.unsqueeze(0)[:,:,:7]) \
                                            .squeeze().cpu().numpy()
                self.coll_msg.data = robot_collision_cost
                self.coll_robot_pub.publish(self.coll_msg)
                
                # trans ee_pose in robot_coordinate to world coordinate
                self.updateGymVisual(curr_state_tensor)
                # robot_sim.command_robot_position(q_des, env_ptr, robot_ptr)
                successed = self.robot_sim.set_robot_state(q_des, qd_des, self.env_ptr, self.robot_ptr)
                if not successed :  
                    break
                # 实验： dynamic object moveDesign
                # actor  :  collision_obj_base_handle 
                self._dynamic_object_moveDesign()

                # pub env_pointcloud and robot_link_spheres
                w_batch_link_spheres = self.mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.robot_coll.w_batch_link_spheres 
                spheres = [s[0][:, :3].cpu().numpy() for s in w_batch_link_spheres]
                # 将所有球体位置信息合并为一个NumPy数组
                robotsphere_positions = np.concatenate(spheres, axis=0)
                self.pub_pointcloud(robotsphere_positions, self.pub_robot_link_pc)

                collision_grid_pc = collision_grid.cpu().numpy() 
                self.pub_pointcloud(collision_grid_pc, self.pub_env_pc)


            except KeyboardInterrupt:
                print('Closing')
                # break

        self.mpc_control.close()
        self.coll_robot_pub.unregister() 
        self.pub_env_pc.unregister()
        self.pub_robot_link_pc.unregister()
        print("mpc_close...")
        
if __name__ == '__main__':

    rospy.init_node('pointcloud_publisher_node')
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()

    sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
    sim_params['headless'] = args.headless
    gym_instance = Gym(**sim_params)

    controller = MPCRobotController(args, gym_instance)
    controller.run()
