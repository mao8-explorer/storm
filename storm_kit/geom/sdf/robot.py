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

import copy
import yaml

import numpy as np
import torch
import trimesh

from ...differentiable_robot_model.coordinate_transform import CoordinateTransform, rpy_angles_to_matrix, multiply_transform, transform_point
from ...differentiable_robot_model.urdf_utils import URDFRobotModel
from ...geom.geom_types import tensor_capsule, tensor_sphere
from ...util_file import join_path, get_mpc_configs_path
from ...geom.nn_model.robot_self_collision import RobotSelfCollisionNet
from ...mpc.model.integration_utils import sphere_pos_sphere_vel
from typing import List, Tuple


class RobotCapsuleCollision:
    """ This class holds a batched collision model where the robot is represented as capsules [one per link]
    """
    def __init__(self, robot_collision_params, batch_size=1, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        # read capsules
        self.batch_size = batch_size
        self.tensor_args = tensor_args
        # keep track of their pose in world frame
        self._link_capsules = None
        self.link_capsules = None
        self.l_T_c = CoordinateTransform(tensor_args=self.tensor_args)
        self.robot_collision_params = robot_collision_params
        self.load_robot_collision_model(robot_collision_params)
    
    def load_robot_collision_model(self, robot_collision_params):
        
        robot_links = robot_collision_params['link_objs']

        # we store as [Batch, n_link, 7]
        self._link_capsules = torch.empty((self.batch_size, len(robot_links), 7), **self.tensor_args)
        for j_idx, j in enumerate(robot_links):
            pose = robot_links[j]['pose_offset']
            # create a transform from pose offset:
            trans = torch.tensor(pose[0:3], **self.tensor_args).unsqueeze(0)
            rpy = torch.tensor(pose[3:], **self.tensor_args).unsqueeze(0)
            # rotation matrix from euler:
            rot = rpy_angles_to_matrix(rpy)
            
            
            l_T_c = CoordinateTransform(trans=trans, rot=rot, tensor_args=self.tensor_args)
            
            r = robot_links[j]['radius']

            # transform base, tip by pose_offset:
            
            base = torch.tensor(robot_links[j]['base'], **self.tensor_args).unsqueeze(0)
            
            tip = torch.tensor(robot_links[j]['tip'], **self.tensor_args).unsqueeze(0)
            base = l_T_c.transform_point(base)
            tip = l_T_c.transform_point(tip)
            self._link_capsules[:, j_idx,:] = tensor_capsule(base, tip, r, tensor_args=self.tensor_args).unsqueeze(0).repeat(self.batch_size, 1)
        #print(self.link_capsules)
        self.link_capsules = self._link_capsules.clone()
    
    def update_robot_link_poses(self, links_pos, links_rot):
        """
        Update link collision poses
        Args:
           link_pos: [batch, n_links , 3]
           link_rot: [batch, n_links , 3 , 3]
        """
        if(links_pos.shape[0] != self.batch_size):
            self.batch_size = links_pos.shape[0]
            self.load_robot_collision_model(self.robot_collision_params)
        
        # This contains coordinate tranforms as [batch_size * n_links ]
        self.l_T_c.set_translation(links_pos)
        self.l_T_c.set_rotation(links_rot)
        
        # Update tranform of link points:
        self.link_capsules[:,:,:3] = self.l_T_c.transform_point(self._link_capsules[:,:,:3])
        self.link_capsules[:,:,3:6] = self.l_T_c.transform_point(self._link_capsules[:,:,3:6])
        
       
    def get_robot_link_objs(self):
        # return capsule spheres in world frame
        
        return self.link_capsules
    
    def get_robot_link_points(self):
        
        raise NotImplementedError


class RobotMeshCollision: 
    """ This class holds a batched collision model with meshes loaded using trimesh. 
    Points are sampled from the mesh which can be used for collision checking.
    """
    def __init__(self, robot_collision_params, batch_size=1, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        # read capsules
        self.batch_size = batch_size
        self.tensor_args = tensor_args
        # keep track of their pose in world frame
        
        #self.link_points = None
        self._batch_link_points = None
        self._link_points = None
        self._link_collision_trans = None
        self._link_collision_rot = None
        self._batch_link_collision_trans = None
        self._batch_link_collision_rot = None

        self._robot_collision_trans = None
        self._robot_collision_rot = None

        self._batch_robot_collision_trans = None
        self._batch_robot_collision_rot = None

        self.w_link_points = None
        self.w_batch_link_points = None
        
        self.l_T_c = CoordinateTransform(tensor_args=self.tensor_args)
        self.robot_collision_params = robot_collision_params
        self.load_robot_collision_model(robot_collision_params)
        
        
    def load_robot_collision_model(self, robot_collision_params):
        
        robot_links = robot_collision_params['link_objs']
        robot_urdf = robot_collision_params['urdf']
        n_pts = robot_collision_params['sample_points']

        # read robot urdf
        robot_urdf = URDFRobotModel(robot_urdf, self.tensor_args)
        
        # read meshes, sample points and store
        
        
        # we store as [n_link, 7]
        self._link_points = torch.empty((len(robot_links), n_pts, 3), **self.tensor_args)
        self._link_collision_trans = torch.empty((len(robot_links), 3), **self.tensor_args)
        self._link_collision_rot = torch.empty((len(robot_links), 3, 3), **self.tensor_args)
        
        for j_idx, j in enumerate(robot_links):
            # read mesh
            mesh_fname, mesh_origin = robot_urdf.get_link_collision_mesh(j)
            
            # sample points
            mesh = trimesh.load_mesh(mesh_fname)
            mesh_centroid = mesh.centroid 
            mesh.vertices = mesh.vertices - mesh_centroid #* 0.0
            points = torch.tensor(trimesh.sample.sample_surface(mesh, n_pts)[0], **self.tensor_args)
            #points = torch.tensor(trimesh.sample.volume_mesh(mesh, n_pts), **self.tensor_args)

            
            # transform points from mesh frame to link frame:
            pose = mesh_origin
            # create a transform from pose offset:
            trans = torch.tensor(pose[0:3], **self.tensor_args).unsqueeze(0)
            rpy = torch.tensor(pose[3:], **self.tensor_args).unsqueeze(0)
            
            # rotation matrix from euler:
            rot = rpy_angles_to_matrix(rpy)
            mesh_cent = torch.as_tensor(mesh_centroid, **self.tensor_args).unsqueeze(0)#.unsqueeze(0)


            trans = trans + (mesh_cent @ rot.transpose(-1,-2))

            l_T_c = CoordinateTransform(trans=trans, rot=rot, tensor_args=self.tensor_args)

            
                        
            #points = l_T_c.transform_point(points)

            # store points

            self._link_points[j_idx, :,:] = points

            # store tranform:
            self._link_collision_rot[j_idx,:,:] = l_T_c.rotation().squeeze(0)
            self._link_collision_trans[j_idx,:] = l_T_c.translation().squeeze(0)
        
    def build_batch_features(self, clone_points=False, clone_pose=True, batch_size=None):
        if(batch_size is not None):
            self.batch_size = batch_size
        if(clone_points):
            
            self._batch_link_points = self._link_points.unsqueeze(0).repeat(self.batch_size, 1, 1,1).clone()
        if(clone_pose):
            self._batch_link_collision_trans = self._link_collision_trans.unsqueeze(0).repeat(self.batch_size, 1, 1).clone()
            self._batch_link_collision_rot = self._link_collision_rot.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).clone()
    def update_batch_robot_collision_pose(self, links_pos, links_rot):
        """
        Update link collision poses
        Args:
           link_pos: [batch, n_links , 3] 
           link_rot: [batch, n_links , 3 , 3] 

        """
        
        (self._batch_robot_collision_rot,
         self._batch_robot_collision_trans) = multiply_transform(links_rot, links_pos,
                                                                 self._batch_link_collision_rot,
                                                                 self._batch_link_collision_trans)
        return True
        
    def update_robot_collision_pose(self, links_pos, links_rot):
        """
        Update link collision poses
        Args:
           link_pos: [n_links, 3]
           link_rot: [n_links, 3, 3]

        """

        self._robot_collision_rot, self._robot_collision_trans = (
            multiply_transform(links_rot, links_pos,
                               self._link_collision_rot,
                               self._link_collision_trans,
                               ))
        
        
        return True

    def update_robot_collision_points(self, links_pos, links_rot):

        self.update_robot_collision_pose(links_pos, links_rot)

        self.w_link_points = transform_point(self._link_points, self._robot_collision_rot, self._robot_collision_trans)
        

    def update_batch_robot_collision_points(self, links_pos, links_rot):
        self.update_batch_robot_collision_pose(links_pos, links_rot)
        self.w_batch_link_points = transform_point(self._batch_link_points,
                                                   self._batch_robot_collision_rot,
                                                   self._batch_robot_collision_trans.unsqueeze(-2))
    def get_robot_link_objs(self):
        raise NotImplementedError
    def get_batch_robot_link_points(self):
        return self.w_batch_link_points
    def get_robot_link_points(self):
        return self.w_link_points
    def get_link_points(self):
        return self._link_points

class RobotSphereCollision:
    """ This class holds a batched collision model where the robot is represented as spheres.
        All points are stored in the world reference frame, obtained by using update_pose calls.
    """
    
    def __init__(self, ndofs, robot_collision_params, batch_size=1, tensor_args={'device':"cpu", 'dtype':torch.float32},
                 traj_dt=None,_fd_matrix_sphere=None):
        """ Initialize with robot collision parameters, look at franka_reacher.py for an example.

        Args:
            robot_collision_params (Dict): collision model parameters
            batch_size (int, optional): Batch size of parallel sdf computation. Defaults to 1.
            tensor_args (dict, optional): compute device and data type. Defaults to {'device':"cpu", 'dtype':torch.float32}.
        """        
        # read capsules
        self.batch_size = batch_size
        self.tensor_args = tensor_args

        # used for sphere vel compute
        self.traj_dt = traj_dt
        self._fd_matrix_sphere = _fd_matrix_sphere

        
        # keep track of their pose in world frame
        
        #self.link_points = None
        self._link_spheres = None
        self._batch_link_spheres = None

        self._link_points = None
        self._link_collision_trans = None
        self._link_collision_rot = None
        self._batch_link_collision_trans = None
        self._batch_link_collision_rot = None

        self._robot_collision_trans = None
        self._robot_collision_rot = None

        self._batch_robot_collision_trans = None
        self._batch_robot_collision_rot = None

        self.w_link_points = None
        self.w_batch_link_spheres = None
        
        self.l_T_c = CoordinateTransform(tensor_args=self.tensor_args)
        self.robot_collision_params = robot_collision_params
        self.load_robot_collision_model(robot_collision_params)
        
        self.dist = None

        # load nn collision model:
        self.robot_nn = RobotSelfCollisionNet(n_joints=ndofs)
        self.robot_nn.load_weights(robot_collision_params['self_collision_weights'], tensor_args)
    
    def load_robot_collision_model(self, robot_collision_params):
        """Load robot collision model, called from constructor

        Args:
            robot_collision_params (Dict): loaded from yml file
        """        
        self.robot_links = robot_collision_params['link_objs']
        self.body_links = robot_collision_params['body_objs']
        self.all_links = self.robot_links + self.body_links
        # load collision file:
        # print(robot_collision_params)
        coll_yml = join_path(get_mpc_configs_path(), robot_collision_params['collision_spheres'])
        with open(coll_yml) as file:
            coll_params = yaml.load(file, Loader=yaml.FullLoader)

        coll_params = coll_params['collision_spheres']

        self._link_spheres = []

        
        # we store as [n_link, 7]
        self._link_collision_trans = torch.empty((len(self.all_links), 3), **self.tensor_args)
        self._link_collision_rot = torch.empty((len(self.all_links), 3, 3), **self.tensor_args)

        for j_idx, j in enumerate(self.all_links):
            
            n_spheres = len(coll_params[j])
            link_spheres = torch.zeros((n_spheres, 4), **self.tensor_args)

            for i in range(n_spheres):
                
                link_spheres[i,:] = tensor_sphere(coll_params[j][i]['center'], coll_params[j][i]['radius'], tensor_args=self.tensor_args, tensor=link_spheres[i,:])
            self._link_spheres.append(link_spheres)
            
        self._w_link_spheres = self._link_spheres
    def build_batch_features(self, clone_objs=False, clone_pose=True, batch_size=None):
        """clones poses/object instances for computing across batch. Use this once per batch size change to avoid re-initialization over repeated calls.

        Args:
            clone_objs (bool, optional): clones objects. Defaults to False.
            clone_pose (bool, optional): clones pose. Defaults to True.
            batch_size ([type], optional): batch_size to clone. Defaults to None.
        """        
        if(batch_size is not None):
            self.batch_size = batch_size
        if(clone_objs):
            self._batch_link_spheres = []
            for i in range(len(self._link_spheres)):
                self._batch_link_spheres.append(self._link_spheres[i].unsqueeze(0).repeat(self.batch_size, 1, 1).clone())
        self.w_batch_link_spheres = copy.deepcopy(self._batch_link_spheres) # 这里使用深拷贝，_batch_link_spheres存的是 initial状态下的对应link下的局部坐标； 至于w_batch_link_spheres是借此申请了一块相同大小的新内存，数据可任意更改

        self.arm_count = len(self.robot_links)
        self.body_count = len(self.body_links)

        # 如果没有 body，就直接把 group_spheres 设为所有手臂 spheres
        if self.body_count == 0:
            self.group_spheres = self.w_batch_link_spheres
        else:
            # 记录每个 body link 上 sphere 数量，用于后面切片
            self.body_ns = [self._batch_link_spheres[i].shape[1]
                            for i in range(self.arm_count,
                                        self.arm_count + self.body_count)]
            self.body_total = sum(self.body_ns)     

            # 预分配一个整体腰身张量 (batch_size, body_total, 4)
            self.body_group_spheres = torch.empty(
                (self.batch_size, self.body_total, 4),
                **self.tensor_args
            )

            # offsets 同前
            offset = 0
            for idx, n in zip(
                range(self.arm_count, self.arm_count + self.body_count),
                self.body_ns
            ):
                self.body_group_spheres[:, offset:offset+n, :] = \
                    self.w_batch_link_spheres[idx][:, :, :]
                offset += n

            # 构造一个不变的 group_spheres 列表：前面 arm_count 个是各 link Tensor，
            # 最后一个永远是 body_group_spheres
            # 注意：self.w_batch_link_spheres 会在 update 时被原位更新
            self.group_spheres = (
                self.w_batch_link_spheres[:self.arm_count]
                + [self.body_group_spheres]
            )

        # 预分配距离矩阵，group 数量也要对应
        n_groups = self.arm_count + (1 if self.body_count>0 else 0)
        self.dist = torch.zeros(
            (self.batch_size, n_groups, n_groups), **self.tensor_args
        ) - 100.0

        
    def _env_build_batch_features(self, clone_objs=False, clone_pose=True, batch_size=None):
        """clones poses/object instances for computing across batch. Use this once per batch size change to avoid re-initialization over repeated calls.

        Args:
            clone_objs (bool, optional): clones objects. Defaults to False.
            clone_pose (bool, optional): clones pose. Defaults to True.
            batch_size ([type], optional): batch_size to clone. Defaults to None.
        """        
        if(batch_size is not None):
            self.batch_size = batch_size
        if(clone_objs):
            self._batch_link_spheres = []
            tmp = []
            for i in range(len(self._link_spheres)):
                self._batch_link_spheres.append(self._link_spheres[i][:,:3].unsqueeze(0).repeat(self.batch_size, 1, 1).clone())
                tmp.append(self._link_spheres[i][:,:4].unsqueeze(0).repeat(self.batch_size, 1, 1).clone())
        self.w_batch_link_spheres = copy.deepcopy(self._batch_link_spheres)
        self.w_batch_link_spheres_vel = copy.deepcopy(tmp)

         
    def update_batch_robot_collision_pose(self, links_pos, links_rot):
        """
        Update link collision poses
        Args:
           link_pos: [batch, n_links , 3] 
           link_rot: [batch, n_links , 3 , 3] 

        """
        '''
        (self._batch_robot_collision_rot,
         self._batch_robot_collision_trans) = multiply_transform(links_rot, links_pos,
                                                                 self._batch_link_collision_rot,
                                                                 self._batch_link_collision_trans)
        '''
        return True
        
    def update_robot_collision_pose(self, links_pos, links_rot):
        """
        Update link collision poses
        Args:
           link_pos: [n_links, 3]
           link_rot: [n_links, 3, 3]

        """
        '''
        self._robot_collision_rot, self._robot_collision_trans = (
            multiply_transform(links_rot, links_pos,
                               self._link_collision_rot,
                               self._link_collision_trans,
                               ))
        
        '''
        return True

    def update_robot_collision_objs(self, links_pos, links_rot):
        '''update pose of link spheres

        Args:
        links_pos: nx3
        links_rot: nx3x3
        '''
        
        # transform link points:
        for i in range(len(self._link_spheres)):
            self._w_link_spheres[i][:,:3] = transform_point(self._link_spheres[:,:3], links_rot[i,:,:], links_pos[i,:,:])
        

    def update_batch_robot_collision_objs(self, links_pos, links_rot):
        '''update pose of link spheres

        Args:
        links_pos: bxnx3
        links_rot: bxnx3x3
        '''
        b, n, _ = links_pos.shape
        for i in range(n):
            # link_pts = self._batch_link_spheres[i][:,:,:3]
            self.w_batch_link_spheres[i][:,:,:3] = transform_point(self._batch_link_spheres[i][:,:,:3], links_rot[:,i,:,:], links_pos[:,i,:].unsqueeze(-2))

    def _vel_update_batch_robot_collision_objs(self, links_pos, links_rot):
        '''update pose of link spheres

        Args:
        links_pos: bxnx3
        links_rot: bxnx3x3

        traj_dt
        state_current as pos(0)
        transform_vel_jt

        '''
        b_h, n, _ = links_pos.shape
        horizon = self._fd_matrix_sphere.shape[0]
        b = b_h // horizon


        for i in range(n):
            
            # self.w_batch_link_spheres[i] = transform_point(self._batch_link_spheres[i], links_rot[:,i,:,:], links_pos[:,i,:].unsqueeze(-2))
           
            # 15000*8*3 | 30 | 30*30   -> 15000 * 8 * 3 
            # self.w_batch_link_spheres_vel[i] = sphere_pos_sphere_vel(self.w_batch_link_spheres[i], self.traj_dt, self._fd_matrix_sphere)
            # 去脚本化方案
            self.w_batch_link_spheres[i] = (self._batch_link_spheres[i] @ links_rot[:,i,:,:].transpose(-1,-2)) + links_pos[:,i,:].unsqueeze(-2)
            state_vel_seq = (torch.matmul(self._fd_matrix_sphere, self.w_batch_link_spheres[i].view(b,horizon,-1)) / self.traj_dt.view(1, -1, 1)).view(b_h,-1,3)
            final = torch.norm(state_vel_seq, dim=-1,keepdim=True) # 15000 * 8 * 3
            self.w_batch_link_spheres_vel[i] = torch.cat((state_vel_seq,final),dim=-1) #15000 * 8 * 4

    def check_self_collisions_nn(self, q):
        """compute signed distance using NN, uses an instance of :class:`.nn_model.robot_self_collision.RobotSelfCollisionNet`

        Args:
            q ([type]): [description]

        Returns:
            [type]: [description]
        """        
        dist = self.robot_nn.compute_signed_distance(q)
        # print(f'Self-collision distance: max = {dist.max().item():.4f}, min = {dist.min().item():.4f}')
        return dist


    def check_self_collisions(self, link_trans, link_rot):
        """Analytic method to compute signed distance between links. This is used to train the NN method :func:`check_self_collisions_nn` amd is not used directly as it is slower.

        Args:
            link_trans ([tensor]): link translation as batch [b,3]
            link_rot ([type]): link rotation as batch [b,3,3]

        Returns:
            [tensor]: signed distance [b,1]

        自碰撞的数据集准备 该函数被用到 很多次；
        """        
        n_links = len(self.w_batch_link_spheres)
        b, _, _ = link_trans.shape
        # if self.dist is None or b != self.dist.shape[0]:
        #     self.dist = torch.zeros((b,n_links,n_links), **self.tensor_args) - 100.0
        # 1) 更新所有 link 的球体位置（in-place）
        self.update_batch_robot_collision_objs(link_trans, link_rot)
        # dist = self.dist

        # 2) 把各 body 链杆的 spheres 原位写入到 body_group_spheres
        if self.body_count > 0:
            offset = 0
            for idx, n_spheres in zip(
                range(self.arm_count, self.arm_count + self.body_count),
                self.body_ns
            ):
                # 拷贝 xyz
                self.body_group_spheres[:, offset:offset+n_spheres, :3] = \
                    self.w_batch_link_spheres[idx][:, :, :3]
                # # 拷贝 radius
                # self.body_group_spheres[:, offset:offset+n_spheres, 3:] = \
                #     self.w_batch_link_spheres[idx][:, :, 3:].clone()
                offset += n_spheres

        # 3) 直接用预先构造好的 list 调用
        link_dist = find_link_distance(self.group_spheres, self.dist, self.arm_count)
        return link_dist
    
        # self.w_batch_group_spheres = self.w_batch_link_spheres
        # # 对 self.w_batch_link_spheres 做文章 ： 包括 link-objs + body-objs, 后者看成一个整体，内部不做distance计算， 只是通过link-pos/rot 更新spheres的位置，给到every link计算distance
        # body_count = len(self.body_links)
        # if body_count > 0: 
        #     # 取出这几个 body 对应的 batch_spheres
        #     body_spheres_list = self.w_batch_link_spheres[-body_count:]
        #     # 在通用维度（第 1 维）上拼接
        #     self.w_batch_body_spheres = torch.cat(body_spheres_list, dim=1)
        #     self.w_batch_group_spheres = self.w_batch_link_spheres[:-body_count] + [self.w_batch_body_spheres]


        # dist = find_link_distance(self.w_batch_group_spheres, dist)
        
        # return dist
    def get_robot_link_objs(self):
        raise NotImplementedError

    def get_batch_robot_link_spheres(self):
        return self.w_batch_link_spheres , self.w_batch_link_spheres_vel

    def get_robot_link_points(self):
        return self.w_link_points

    def get_link_points(self):
        return self._link_points


@torch.jit.script
def compute_spheres_distance(spheres_1, spheres_2):
    
    b, n, _ = spheres_1.shape
    b_l, n_l, _ = spheres_2.shape
    
    #dist = torch.zeros((b, n), device=spheres_1.device,
    #                   dtype=spheres_2.dtype)
    
    


    j = 0
    link_sphere_pts = spheres_1[:,j,:]
    link_sphere_pts = link_sphere_pts.unsqueeze(1)
    # find closest distance to other link spheres:
    
    
    
    #print(l_spheres.shape, link_sphere_pts.shape)
    s_dist = torch.norm(spheres_2[:,:,:3] - link_sphere_pts[:,:,:3], dim=-1)
    s_dist = spheres_2[:,:,3] + link_sphere_pts[:,:,3] - s_dist # 这里是r1 + r2 - L(center_1 - center_2)
    max_dist = torch.max(s_dist, dim=-1)[0]
    
    
    for j in range(1,n):
        link_sphere_pts = spheres_1[:,j,:]
        link_sphere_pts = link_sphere_pts.unsqueeze(1)
        # find closest distance to other link spheres:
        s_dist = torch.norm(spheres_2[:,:,:3] - link_sphere_pts[:,:,:3], dim=-1)
        s_dist = spheres_2[:,:,3] + link_sphere_pts[:,:,3] - s_dist
        s_dist = torch.max(s_dist, dim=-1)[0]
        max_dist = torch.maximum(max_dist, s_dist)
        
    dist = max_dist #torch.max(dist,dim=-1)[0]
    return dist

@torch.jit.script
def find_closest_distance(link_idx, links_sphere_list):
    # type: (int, List[Tensor]) -> Tensor
    """closet distance computed via iteration between sphere sets.

    Args:
        link_idx ([type]): [description]
        links_sphere_list ([type]): [description]

    Returns:
        [type]: [description]
    """

    spheres = links_sphere_list[link_idx]
    b, n, _ = spheres.shape
    #spheres = spheres.view(b * n, 4)
    #link_pts = spheres[:,:,:3]
    #link_dist = torch.zeros((b,len(links_sphere_list)), **self.tensor_args)
    dist = torch.zeros((b,len(links_sphere_list), n), device=spheres.device,
                       dtype=spheres.dtype)
    for j in range(n):
        # for every sphere in current link
        link_sphere_pts = spheres[:,j,:]
        link_sphere_pts = link_sphere_pts.unsqueeze(1)
        # find closest distance to other link spheres:
        
        for i in range(len(links_sphere_list)):
            if(i == link_idx or i==link_idx-1 or i==link_idx+1):
                dist[:,i,j] = -100.0
                continue
            # transform link_idx spheres to current link frame:
            # given a link and another link, find closest distance between them:
            l_spheres = links_sphere_list[i]
        
            b_l, n_l, _ = l_spheres.shape
            
            #print(l_spheres.shape, link_sphere_pts.shape)
            s_dist = torch.norm(l_spheres[:,:,:3] - link_sphere_pts[:,:,:3], dim=-1)
            s_dist = l_spheres[:,:,3] + link_sphere_pts[:,:,3] - s_dist 

            # dist: b, n_l -> b
            dist[:,i,j] = torch.max(s_dist, dim=-1)[0]
    link_dist = torch.max(dist,dim=-1)[0]
    return link_dist

@torch.jit.script
def find_link_distance(links_sphere_list, dist, arm_count: int = 7):
    # type: (List[Tensor], Tensor, int) -> Tensor
    futures: List[Tuple[int,int,torch.jit.Future[torch.Tensor]]] = []
    # futures : List[torch.jit.Future[torch.Tensor]] = []
    b, n, _ = links_sphere_list[0].shape
    # spheres = links_sphere_list[0]
    n_links = len(links_sphere_list)
    dist.mul_(0.0).sub_(100.0)
    #dist = torch.zeros((b,n_links,n_links), device=spheres.device,
    #                   dtype=spheres.dtype) - 100.0


    # 1) 先收集所有需要计算的 (i,j) 对
    pairs: List[Tuple[int,int]] = []

    # —— 手臂内部：跳过相邻，只要 j>=i+2 —— 
    for i in range(arm_count):
        for j in range(i+2, arm_count):
            pairs.append((i, j))

    # —— 手臂 ↔ 腰身 group —— 
    if n_links > arm_count:
        body_idx = arm_count
        for i in range(arm_count):
            pairs.append((i, body_idx))


    # （如果未来有多个 body group，就把它们全都加进来）

    # 2) 并行 fork
    for i, j in pairs:
        f = torch.jit.fork(compute_spheres_distance,
                           links_sphere_list[i],
                           links_sphere_list[j])
        futures.append((i, j, f))


    # 3) wait + 写回
    for i, j, f in futures:
        d = torch.jit.wait(f)
        dist[:, i, j] = d
        dist[:, j, i] = d

    # 4) 归约
    return torch.max(dist, dim=-1)[0]
        

    # for i in range(n_links):
    #     # for every link, compute the distance to the other links:
    #     current_spheres = links_sphere_list[i]
    #     for j in range(i + 2, n_links): # i+2 保证不是计算相邻link
    #         compute_spheres = links_sphere_list[j]

    #         # find the distance between the two links:
    #         d = torch.jit.fork(compute_spheres_distance, current_spheres, compute_spheres)
    #         futures.append(d)

    # k = 0
    # for i in range(n_links):
    #     # for every link, compute the distance to the other links:
    #     for j in range(i + 2, n_links):
    #         d = torch.jit.wait(futures[k])
    #         dist[:,i,j] = d
    #         dist[:,j,i] = d
    #         k += 1
    # link_dist = torch.max(dist,dim=-1)[0]
    # return link_dist
