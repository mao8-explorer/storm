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
import torch
import torch.nn as nn
# import torch.nn.functional as F
from ...geom.sdf.robot_world import RobotWorldCollisionPrimitive
from .gaussian_projection import GaussianProjection


@torch.jit.script
def CostCompute(sdf_grad, vel, w1, w2,batch_size,horizon,n_links,vec_weight):
    # type: (Tensor, Tensor, int, int, int, int ,int ,Tensor) -> Tensor

    potential , grad = sdf_grad[:,:,0], sdf_grad[:,:,1:] # batch * n_links , batch * n_links * 3
    vel_abs = vel[:,:,-1]
    vel_orient = vel[:,:,:-1] # batch * n_links *3
    # 计算SDF梯度向量的绝对值
    grad_abs = torch.linalg.norm(grad, ord=2, dim=-1) # grad : batch * n_links * 3 -> batch * n_links
    # 计算速度向量和SDF梯度向量的点积
    dot_product = torch.sum(vel_orient * grad, dim=-1) #  batch * n_links * 3  - > batch * n_links
    # 计算余弦值
    cos_theta = dot_product / (vel_abs * grad_abs + 1e-6)
    cos_theta[potential==1.0] = -1.0
    cost_sdf = w1 * potential +\
               w2 * potential * vel_abs * (1.0 +\
                                                    1.0 * (torch.max(-cos_theta, torch.tensor(0.0))) +\
                                                    0.5 * (torch.min(-cos_theta, torch.tensor(0.0)))
                                                    )
    # cost_sdf = w1 * potential + w2 * potential * vel_abs
    # cost_sdf = w1 * potential
    # cost_sdf = w2 * potential * vel_abs
    cost_sdf = cost_sdf.view(batch_size, horizon, n_links) 
    disp_vec = vec_weight * cost_sdf
    cost = torch.sum(disp_vec, dim=-1) # 对每个link分配相同的权重 做sum
    

    return cost





class PrimitiveCollisionCost(nn.Module):
    def __init__(self, weight=None, vec_weight = None, pv_weight =None, world_params=None, robot_params=None, gaussian_params={},
                 distance_threshold=0.1, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32},
                 traj_dt=None, _fd_matrix_sphere = None):
        super(PrimitiveCollisionCost, self).__init__()
        
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight,**self.tensor_args)
        self.vec_weight = torch.as_tensor(vec_weight, **self.tensor_args)
        self.pv_weight = torch.as_tensor(pv_weight, **self.tensor_args)
        self.w1 = self.pv_weight[0]
        self.w2 = self.pv_weight[1]
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)

        robot_collision_params = robot_params['robot_collision_params']
        # print(robot_collision_params)
        self.batch_size = -1
        # BUILD world and robot:
        self.robot_world_coll = RobotWorldCollisionPrimitive(robot_collision_params,
                                                             world_params['world_model'],
                                                             tensor_args=self.tensor_args,
                                                             bounds=robot_params['world_collision_params']['bounds'],
                                                             grid_resolution=robot_params['world_collision_params']['grid_resolution'],
                                                             traj_dt = traj_dt,
                                                             _fd_matrix_sphere = _fd_matrix_sphere)
        
        self.n_world_objs = self.robot_world_coll.world_coll.n_objs
        self.t_mat = None
        self.distance_threshold = distance_threshold
        self.current_state_collision =None



    def optimal_forward(self, link_pos_seq, link_rot_seq): # 6.5ms

        
        inp_device = link_pos_seq.device
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        n_links = link_pos_seq.shape[2]

        link_pos_batch = link_pos_seq.view(batch_size * horizon, n_links, 3)
        link_rot_batch = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)
        # 基于点云数据的 voxel grid to SDF
        self.robot_world_coll.optimal_check_robot_sphere_collisions_voxeltosdf(link_pos_batch, link_rot_batch)
        potential, grad, vel_orient, vel_abs = self.robot_world_coll.sdf1_grad3_vel4[:,:,0] , self.robot_world_coll.sdf1_grad3_vel4[:,:,1:4], \
                                      self.robot_world_coll.sdf1_grad3_vel4[:,:,4:-1] , self.robot_world_coll.sdf1_grad3_vel4[:,:,-1]
        self.current_state_collision = potential[-1*horizon,:] #best_traj index shape is （7，）查询potential
        self.current_grad = grad[-1*horizon,:,:] #best_traj index shape (7,3) # gradient 查询
        self.current_vel_orient = vel_orient[-1*horizon,:] #best_traj index shape (7,3) # 速度查询
        self.current_sphere_pos = self.robot_world_coll.sphere_pos_links

        # cost = CostCompute(sdf_grad, vel, self.w1, self.w2,batch_size,horizon,n_links, self.vec_weight)
        # type: (Tensor, Tensor, int, int, int, int ,int ,Tensor) -> Tensor
        # potential , grad = sdf_grad[:,:,0], sdf_grad[:,:,1:] # batch * n_links , batch * n_links * 3
        # vel_abs = vel[:,:,-1]
        # vel_orient = vel[:,:,:-1] # batch * n_links *3
        # 计算SDF梯度向量的绝对值
        grad_abs = torch.linalg.norm(grad, ord=2, dim=-1) # grad : batch * n_links * 3 -> batch * n_links
        # 计算速度向量和SDF梯度向量的点积
        dot_product = torch.sum(vel_orient * grad, dim=-1) #  batch * n_links * 3  - > batch * n_links
        # 计算余弦值
        cos_theta = dot_product / (vel_abs * grad_abs + 1e-6)
        cos_theta[potential==1.0] = -1.0
        #  在权重上面多做文章 提升性能呀！！！
        cost_sdf = self.w1 * potential +\
                   self.w2 * potential * vel_abs * (1.0 +\
                                                        2.0 * (torch.max(-cos_theta, torch.tensor(0.0).to(inp_device))) +\
                                                        0.5 * (torch.min(-cos_theta, torch.tensor(0.0).to(inp_device)))
                                                        )
        judge_cost_sdf = self.w1 * potential + self.w2 * potential * vel_abs
        # cost_sdf = self.w1 * potential
        # cost_sdf = self.w2 * potential * vel_abs
        cost_sdf = cost_sdf.view(batch_size, horizon, n_links) 
        disp_vec = self.weight * self.vec_weight * cost_sdf 
        cost = torch.sum(disp_vec, dim=-1) # 对每个link分配相同的权重 做sum

        judge_cost_sdf = judge_cost_sdf.view(batch_size, horizon, n_links) 
        judge_disp_vec = self.weight * self.vec_weight * judge_cost_sdf
        judge_cost = torch.sum(judge_disp_vec, dim=-1) # 对每个link分配相同的权重 做sum


        return cost.to(inp_device) , judge_cost.to(inp_device)

    def voxel_forward(self, link_pos_seq, link_rot_seq):

        
        inp_device = link_pos_seq.device
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        n_links = link_pos_seq.shape[2]

        if(self.batch_size != batch_size):
            self.batch_size = batch_size
            self.robot_world_coll.build_batch_features(self.batch_size * horizon, clone_pose=True, clone_points=True)

        link_pos_batch = link_pos_seq.view(batch_size * horizon, n_links, 3)
        link_rot_batch = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)
        # 原始的基于图形学的碰撞检测
        # dist = self.robot_world_coll.check_robot_sphere_collisions(link_pos_batch,
        #                                                            link_rot_batch)
        # 基于点云数据的 离散体素化 voxel grid
        dist = self.robot_world_coll.check_robot_collisions_pointCloud(link_pos_batch,
                                                                   link_rot_batch)
        
        dist = dist.view(batch_size, horizon, n_links)#, self.n_world_objs)
        # cost only when dist is less
        dist += self.distance_threshold

        # dist[dist <= 0.0] = 0.0
        # dist[dist > 0.2] = 0.2
        dist = dist / 0.25
        
        cost = torch.sum(dist, dim=-1)


        cost = self.weight * cost 

        return cost.to(inp_device)


    def forward(self, link_pos_seq, link_rot_seq):

        
        inp_device = link_pos_seq.device
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        n_links = link_pos_seq.shape[2]

        if(self.batch_size != batch_size):
            self.batch_size = batch_size
            self.robot_world_coll.build_batch_features(self.batch_size * horizon, clone_pose=True, clone_points=True)

        link_pos_batch = link_pos_seq.view(batch_size * horizon, n_links, 3)
        link_rot_batch = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)
        # 基于点云数据的 voxel grid to SDF
        dist = self.robot_world_coll.check_robot_sphere_collisions_voxeltosdf(link_pos_batch,
                                                                   link_rot_batch)
        
        cost_sdf = torch.zeros_like(dist)

        # 对dist大于0.05小于0.30的区域进行运算
        mask_mid = (dist > 0.05) & (dist <= 0.20)
        cost_sdf[mask_mid] = torch.exp(-20 * (dist[mask_mid] - 0.05))

        # 对dist小于等于0.05的区域直接设置为1
        cost_sdf[dist <= 0.05] = 1.0

        # 对dist大于0.30的区域直接设置为0
        cost_sdf[dist > 0.20] = 0.0

        cost_sdf = cost_sdf.view(batch_size, horizon, n_links) 
        
        self.current_state_collision = cost_sdf[-1,0,:] #mean_traj index
        cost = torch.sum(cost_sdf, dim=-1) # 对每个link分配相同的权重 做sum
        cost = self.weight * cost

        return cost.to(inp_device)

