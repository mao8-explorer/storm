#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
""" Example spawning a robot in gym 

"""
import copy
from shutil import move
from isaacgym import gymapi
from isaacgym import gymutil

import torch
from trimesh import PointCloud
import trimesh.transformations as tra

torch.multiprocessing.set_start_method('spawn', force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#
# import matplotlib
# matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import time
import yaml
import argparse
import numpy as np


from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask

from storm_kit.geom.utils import get_pointcloud_from_depth

np.set_printoptions(precision=2)



from mppi_scn import MPPIPolicy

import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField

def view_point_cloud(obs):
    pc = np.array(obs['pc'][0])
    x , y , z = pc[:,0], pc[:,1], pc[:,2]
    color = obs['pc_label'][0]
    ax = plt.axes(projection='3d')
    ax.scatter3D(x,y,z, c=color, cmap='coolwarm')
    plt.show()


def set_scene(obs):
    orig_scene_pc = obs["pc"]
    scene_labels = obs["pc_label"]
    label_map = obs["label_map"]

    # Remove robot points plus excluded
    scene_pc_mask = np.logical_and(
        scene_labels != label_map["robot"],
        scene_labels != label_map["target"],
    )

    # Transform into robot frame (z up)
    camera_pose = obs["camera_pose"]
    scene_pc = tra.transform_points(orig_scene_pc, camera_pose)


def mpc_robot_interactive(args, gym_instance):
    vis_ee_target = True
    # yml配置
    robot_file = args.robot + '.yml'
    task_file = args.robot + '_reacher.yml'
    world_file = 'collision_primitives_3d.yml'
    gym = gym_instance.gym
    sim = gym_instance.sim
    env_ptr = gym_instance.env_list[0]
    viewer = gym_instance.viewer
    world_yml = join_path(get_gym_configs_path(), world_file)
    with open(world_yml) as file:
        world_params = yaml.load(file, Loader=yaml.FullLoader)  # world_mode
    robot_yml = join_path(get_gym_configs_path(), args.robot + '.yml')
    with open(robot_yml) as file:
        robot_params = yaml.load(file, Loader=yaml.FullLoader)
    sim_params = robot_params['sim_params']  # get from -->'/home/zm/MotionPolicyNetworks/storm_ws/src/storm/content/configs/gym/franka.yml'
    sim_params['asset_root'] = get_assets_path()
    sim_params['collision_model']=None
    device = torch.device('cuda', 0) 
    tensor_args = {'device': device, 'dtype': torch.float32}

    # create robot simulation: contains a generic robot class that can load a robot asset into sim and gives access to robot's state and control.
    robot_sim = RobotSim(gym_instance=gym, sim_instance=sim, env_instance = env_ptr, viewer = viewer ,**sim_params, device=device)
    # create gym environment:
    robot_pose = sim_params['robot_pose']  # robot_pose: [0, 0.0, 0, -0.707107, 0.0, 0.0, 0.707107]
    robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)


    # spawn camera:
    external_transform = tra.euler_matrix(0, 0, np.pi / 9).dot(
            tra.euler_matrix(0, np.pi / 2, 0)
    )
    external_transform[0, 3] = 2.0
    external_transform[1, 3] = 1.0
    external_transform[2, 3] = 0
    robot_sim.spawn_camera(env_ptr, 60, 640, 480, external_transform)

    # get pose
    w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)
    
    w_T_robot = torch.eye(4)
    quat = torch.tensor([w_T_r.r.w,w_T_r.r.x,w_T_r.r.y,w_T_r.r.z]).unsqueeze(0)
    rot = quaternion_to_matrix(quat)
    w_T_robot[0,3] = w_T_r.p.x
    w_T_robot[1,3] = w_T_r.p.y
    w_T_robot[2,3] = w_T_r.p.z
    w_T_robot[:3,:3] = rot[0]


    w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3,3].unsqueeze(0),
                                        rot=w_T_robot[0:3,0:3].unsqueeze(0))

    world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)

    # in one word, may be all above just part of this ReacherTask 初始化所有
    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)

    # update goal:
    franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_des_list = [franka_bl_state]
    x_des = x_des_list[0]
    mpc_control.update_params(goal_state=x_des)

    # spawn object:
    x, y, z = 1.0, 0.0, 0.0
    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(x, y, z)
    object_pose.r = gymapi.Quat(0.801, 0.598, 0.0, 0)
    obj_asset_file = "urdf/mug/movable_mug.urdf"
    obj_asset_root = get_assets_path()

    collision_obj_asset_file = "urdf/mug/movable_collision_test.urdf"
    if (vis_ee_target):
        target_object = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose,
                                                    name='ee_target_object')
        collision_obj = world_instance.spawn_collision_obj(collision_obj_asset_file, obj_asset_root, object_pose,
                                                    name='collision_move_test')        

        collision_obj_base_handle = gym.get_actor_rigid_body_handle(env_ptr, collision_obj, 0)
        collision_body_handle = gym.get_actor_rigid_body_handle(env_ptr, collision_obj, 6)

        obj_base_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 0)
        obj_body_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 6)

        tray_color = gymapi.Vec3(0.8, 0.1, 0.1)
        gym.set_rigid_body_color(env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
        gym.set_rigid_body_color(env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
        tray_color = gymapi.Vec3(0.80, 0.42, 0.13)
        gym.set_rigid_body_color(env_ptr, collision_obj, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
        gym.set_rigid_body_color(env_ptr, collision_obj, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
        obj_asset_file = "urdf/mug/mug.urdf"
        obj_asset_root = get_assets_path()

        ee_handle = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, 
                                                name='ee_current_as_mug')
        ee_body_handle = gym.get_actor_rigid_body_handle(env_ptr, ee_handle, 0)
        tray_color = gymapi.Vec3(0.0, 0.8, 0.0)
        gym.set_rigid_body_color(env_ptr, ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    # object_pose.p = gymapi.Vec3(g_pos[0], g_pos[1], g_pos[2])
    # object_pose.r = gymapi.Quat(g_q[1], g_q[2], g_q[3], g_q[0])
    # object_pose = w_T_r * object_pose

    object_pose.p = gymapi.Vec3(0.280,0.469,0.118)
    object_pose.r = gymapi.Quat(0.392,0.608,-0.535,0.436)


    if (vis_ee_target):
        gym.set_rigid_transform(env_ptr, obj_base_handle, object_pose)

        # object_pose.p = gymapi.Vec3(0.2, 0.4, 0.2)
        # object_pose.r = gymapi.Quat(g_q[1], g_q[2], g_q[3], g_q[0])
        # object_pose = w_T_r * object_pose
        object_pose.p = gymapi.Vec3(0.580,0.626, -0.274)
        object_pose.r = gymapi.Quat(0.278,0.668,-0.604,0.334)
        gym.set_rigid_transform(env_ptr, collision_obj_base_handle, object_pose)


    sim_dt = mpc_control.exp_params['control_dt']
    q_des = None
    qd_des = None
    t_step = gym_instance.get_sim_time()



    policy = MPPIPolicy() #sceneCollisionNet 句柄
    # policy = mpc_control.controller.rollout_fn.scene_collision_cost.policy

    # SCN get ee_pose (copy from SCN_ updata_state):
    # env_states = robot_sim._get_gym_state()
    # obs = {
    #     "robot_q": np.array(list(env_states[-1]["robot"].values()))
    # }
    # robot_q = obs["robot_q"].astype(np.float64).copy()
    # policy.robot.set_joint_cfg(robot_q)
    # scn_ee_pose = policy.robot.ee_pose[0].cpu().numpy()

    # STORM get ee_pose
    current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
    filtered_state_mpc = current_robot_state  # mpc_control.current_state
    curr_state = np.hstack(
        (filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
    curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
    pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
    # summary : SCN与STORM在ee_pose的计算上是一致的 或者说 SCN与STORM的FK模型具有一致性 输入关节角 输出各link的pose


    ee_pose = gymapi.Transform()

    #  all ros_related
    pub_env_pc = rospy.Publisher('env_pc', PointCloud2, queue_size=5)
    pub_robot_link_pc = rospy.Publisher('robot_link_pc', PointCloud2, queue_size=5)
    msg = PointCloud2()
    msg.header.frame_id = "world"
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.is_dense = False

    coll_msg = Float32()
    coll_robot_pub = rospy.Publisher('robot_collision', Float32, queue_size=10)


    # 实验： dynamic object moveDesign
    # 首先获取物体当前位姿 并将其转到robot坐标系下
    move_pose = copy.deepcopy(world_instance.get_pose(collision_body_handle))
    move_pose = copy.deepcopy(w_T_r.inverse() * move_pose)
    # 已知 物体在robot坐标系下的移动边界值为 bounds
    move_bounds = np.array([[-0.58, -0.57,  0.63],
                        [0.58, 0.57,  0.63]])
    # 根据两个边界点 确定物体移动方向  这一段代码有误 typeError: unsupported operand type(s) for -: 'list' and 'list'
    velocity_vector = np.array([move_bounds[1] - move_bounds[0]])
    # 将 方向向量 velocity_vector 归一化
    velocity_vector = velocity_vector / np.linalg.norm(velocity_vector)
    
    loop_last_time = time.time_ns()
    while not rospy.is_shutdown():
        try:

            loop_time = (time.time_ns() - loop_last_time)/1e+6
            loop_last_time = time.time_ns()

            gym_instance.step()
            gym_instance.clear_lines()

            scene_last_time = time.time_ns()         
            robot_sim._observe_all_cameras()
            obs = {}
            obs.update(robot_sim._build_pc_observation())
            policy.set_scene(obs)  # you must know how the scene coordinate changes !!!
            scene_pc_time = (time.time_ns() - scene_last_time)/1e+6

            sdf_last_time = time.time_ns() 
            scene_pc = policy.scene_collision_checker.cur_scene_pc
            # mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll._compute_dynamic_sdfgrid(scene_pc)
            collision_grid = mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll._compute_dynamic_voxeltosdf(scene_pc, visual = True)
            voxeltosdf_time = (time.time_ns() - sdf_last_time)/1e+6
            
            # print("Control Loop: {:<10.3f}sec | Scene PC: {:<10.3f}sec voxel SDF: {:<10.3f}sec | Percent: {:<5.2f}%".format(loop_time, scene_pc_time, voxeltosdf_time, (scene_pc_time / loop_time) * 100))

            if mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.robot_coll.w_batch_link_spheres is not None :
                w_batch_link_spheres = mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll.robot_coll.w_batch_link_spheres 
                spheres = [s[0][:, :3].cpu().numpy() for s in w_batch_link_spheres]
                # 将所有球体位置信息合并为一个NumPy数组
                all_positions = np.concatenate(spheres, axis=0)
    
                msg.header.stamp = rospy.Time().now()
                if len(all_positions.shape) == 3:
                    msg.height = all_positions.shape[1]
                    msg.width = all_positions.shape[0]
                else:
                    msg.height = 1
                    msg.width = len(all_positions)

                msg.row_step = msg.point_step * all_positions.shape[0]
                msg.data = np.asarray(all_positions, np.float32).tostring()

                pub_robot_link_pc.publish(msg)   

            # SCN get ee_pose (copy from SCN_ updata_state):
            # env_states = robot_sim._get_gym_state()
            # robot_q = np.array(list(env_states[-1]["robot"].values())).astype(np.float64).copy()
            # robot_q = robot_q.reshape(1, -1)

            # last_time = time.time_ns()
            
            # colls_value = policy._check_collisions(robot_q)
            # check_time = (time.time_ns() - last_time)/1000000
            # print(check_time)
            # print(colls_value.reshape(-1).cpu().numpy())

            # policy._fcl_check_collisions(robot_q) # low frequency

            # cur_scene_pc visualize SCN
            # if policy.scene_collision_checker.cur_scene_pc.cpu().numpy() is not None:
            #     scene_pc = policy.scene_collision_checker.cur_scene_pc.cpu().numpy() 


            #     msg.header.stamp = rospy.Time().now()
            #     if len(scene_pc.shape) == 3:
            #         msg.height = scene_pc.shape[1]
            #         msg.width = scene_pc.shape[0]
            #     else:
            #         msg.height = 1
            #         msg.width = len(scene_pc)

            #     msg.row_step = msg.point_step * scene_pc.shape[0]
            #     msg.data = np.asarray(scene_pc, np.float32).tostring()

            #     pub_env_pc.publish(msg)

    
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

            pub_env_pc.publish(msg)
                # trans_robot_link_pc = policy.scene_collision_checker._link_trans.cpu().numpy().squeeze()

                # if len(trans_robot_link_pc.shape) == 3:
                #     msg.height = trans_robot_link_pc.shape[1]
                #     msg.width = trans_robot_link_pc.shape[0]
                # else:
                #     msg.height = 1
                #     msg.width = len(trans_robot_link_pc)

                # msg.row_step = msg.point_step * trans_robot_link_pc.shape[0]
                # msg.data = np.asarray(trans_robot_link_pc, np.float32).tostring()

                # pub_robot_link_pc.publish(msg)


            if (vis_ee_target):
                pose = copy.deepcopy(world_instance.get_pose(obj_body_handle))
                pose = copy.deepcopy(w_T_r.inverse() * pose)

                if (np.linalg.norm(g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (
                        np.linalg.norm(g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z])) > 0.0)):
                    g_pos[0] = pose.p.x
                    g_pos[1] = pose.p.y
                    g_pos[2] = pose.p.z
                    g_q[1] = pose.r.x   
                    g_q[2] = pose.r.y
                    g_q[3] = pose.r.z
                    g_q[0] = pose.r.w

                    mpc_control.update_params(goal_ee_pos=g_pos,
                                              goal_ee_quat=g_q)
            t_step += sim_dt

            current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))

            command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)

            filtered_state_mpc = current_robot_state  # mpc_control.current_state
            curr_state = np.hstack(
                (filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

            curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
            # get position command:
            q_des = copy.deepcopy(command['position'])
            qd_des = copy.deepcopy(command['velocity'])  # * 0.5
     
            ee_error = mpc_control.get_current_error(filtered_state_mpc)
            
            # use for robot_self_collision_check : mpc_control.controller.rollout_fn.robot_self_collision_cost()
            robot_collision_cost = mpc_control.controller.rollout_fn \
                                        .robot_self_collision_cost(curr_state_tensor.unsqueeze(0)[:,:,:7]) \
                                        .squeeze().cpu().numpy()
            coll_msg.data = robot_collision_cost
            coll_robot_pub.publish(coll_msg)
            

            pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)

            # get current pose:
            e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
            e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
            ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
            ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])

            ee_pose = copy.deepcopy(w_T_r) * copy.deepcopy(ee_pose)

            if (vis_ee_target):
                gym.set_rigid_transform(env_ptr, ee_body_handle, copy.deepcopy(ee_pose))




            # print(["{:.3f}".format(x) for x in ee_error], " opt_dt: {:.3f}".format(mpc_control.opt_dt),
            #       " mpc_dt: {:.3f}".format(mpc_control.mpc_dt),
            #       " t_step: {:.3f}".format(t_step),
            #       " gym_sim_time: {:.3f}".format(gym_instance.get_sim_time()),
            #       " run_hz: {:.3f}".format(run_hz))                       
               

            # gym_instance.clear_lines() 放在while初始，在订阅点云前清屏
            top_trajs = mpc_control.top_trajs.cpu().float()  # .numpy()
            n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
            w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)


            top_trajs = w_pts.cpu().numpy()
            color = np.array([0.0, 1.0, 0.0])
            for k in range(top_trajs.shape[0]):
                pts = top_trajs[k, :, :]
                color[0] = float(k) / float(top_trajs.shape[0])
                color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
                gym_instance.draw_lines(pts, color=color)

            # robot_sim.command_robot_position(q_des, env_ptr, robot_ptr)
            robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)

            # 实验： dynamic object moveDesign
            # actor  :  collision_obj_base_handle 
            #  如果越界，将速度向量取反 超过两个边界 都要变更速度方向
            if move_pose.p.x <= move_bounds[0][0] or move_pose.p.x >= move_bounds[1][0]:
               velocity_vector *= -1
            # 速度积分  vel to pos
            dt_scale = 0.01
            move_pose.p.x += velocity_vector[0][0] * dt_scale
            move_pose.p.y += velocity_vector[0][1] * dt_scale
            move_pose.p.z += velocity_vector[0][2] * dt_scale
            # 坐标系变更
            gym.set_rigid_transform(env_ptr, collision_obj_base_handle, w_T_r * move_pose)


        except KeyboardInterrupt:
            print('Closing')
            done = True
            break
    mpc_control.close()

  


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


    mpc_robot_interactive(args, gym_instance)

