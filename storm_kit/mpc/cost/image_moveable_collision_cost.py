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
from ...geom.geom_types import tensor_circle
from .gaussian_projection import GaussianProjection
from ...util_file import get_assets_path, join_path
from ...geom.sdf.world import WorldMoveableImageCollision

class ImagemoveCollisionCost(nn.Module):
    def __init__(self, weight=None, vec_weight=None, collision_file=None, bounds=[], dist_thresh=0.01, gaussian_params={}, tensor_args={'device': torch.device('cpu'), 'dtype': torch.float32}):
        super(ImagemoveCollisionCost, self).__init__()
        
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight, **self.tensor_args)

        self.vec_weight = torch.as_tensor(vec_weight, **self.tensor_args)
        self.w1 = self.vec_weight[0]
        self.w2 = self.vec_weight[1]
        
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        
        # BUILD world and robot:
        world_image = join_path(get_assets_path(), collision_file)
        
        self.world_coll = WorldMoveableImageCollision(bounds=bounds, world_image=world_image, tensor_args=tensor_args)
        self.dist_thresh = dist_thresh  # meters
        
    def nogradforward(self, state_seq):
        inp_device = state_seq.device
        batch_size = state_seq.shape[0]
        horizon = state_seq.shape[1]
        pos_batch = state_seq[:, :, :2].view(batch_size * horizon, 2)
        vel_batch = state_seq[:, :, 2:4].view(batch_size * horizon, 2)
  
        # 查询SDF获取点的值
        dist = self.world_coll.get_pt_value(pos_batch)
        
        # 计算速度向量的绝对值
        vel_abs = torch.linalg.norm(vel_batch, ord=2, dim=1)
        
        # 将距离值乘以速度绝对值得到大小为（batch_size, horizon, 1）的代价值
        cost = dist.view(batch_size, horizon, 1)
        
        # 仅当距离小于阈值时应用代价
        res = self.weight * cost
        res = res.squeeze(-1)
        return res.to(inp_device)

    def forward(self, state_seq):

        """       
        利用SDF的势场potential 和 梯度gradient 计算cost
        构造初级代价函数(minimize cost function) : 
        cost = w1 * sdf_potential + w2 * max{ sdf_potential * car_velocity *[- cos(theta) ], 0 }
        其中 theta  = arccos (sdf_gradient,car_velocity_direction), 是SDF梯度与小车速度的夹角
        需要注意的是： 这里只有当theta > pi/2 时，速度惩罚才生效，其他的暂时不考虑 
        """
        inp_device = state_seq.device
        batch_size = state_seq.shape[0]
        horizon = state_seq.shape[1]
        pos_batch = state_seq[:, :, :2].view(batch_size * horizon, 2)
        vel_batch = state_seq[:, :, 2:4].view(batch_size * horizon, 2)
        
        # 获取势场potential
        potential = self.world_coll.get_pt_value(pos_batch)

        # 获取SDF的梯度方向
        grad_y, grad_x = self.world_coll.get_pt_gradxy(pos_batch)
        
        # 计算速度向量的绝对值
        vel_abs = torch.linalg.norm(vel_batch, ord=2, dim=1)
        # 计算SDF梯度向量的绝对值
        grad_abs = torch.linalg.norm(torch.stack([grad_x, grad_y], dim=1), ord=2, dim=1)

        # 计算速度向量和SDF梯度向量的点积
        dot_product = torch.sum(vel_batch * torch.stack([grad_x, grad_y], dim=1), dim=1)

        # 计算余弦值
        cos_theta = dot_product / (vel_abs * grad_abs + 1e-6)

        # 计算夹角（弧度）
        theta = torch.acos(cos_theta)

        # # 根据代价函数计算cost
        # cost = self.w1 * potential +\
        #             self.w2 * potential * vel_abs * (1.0 + (torch.max(-torch.cos(theta), torch.tensor(0.0).to(inp_device))))
                # self.w1* 8.0 * potential*vel_abs
                # torch.max( self.w2 * potential * vel_abs * (-torch.cos(theta)), torch.tensor(0.0).to(inp_device)) 
        # 根据代价函数计算cost
        # judge_cost = self.w1 * potential #    13 2coll
        # cost = self.w2 * potential * vel_abs
        judge_cost = self.w1 * potential + self.w2 * potential * vel_abs 
        # cost = judge_cost
        # cost = self.w1 * potential +\
        #             self.w2 * potential * vel_abs * (1.0 + (torch.max(-torch.cos(theta), torch.tensor(0.0).to(inp_device))))
        # cost = self.w1 * potential +\
        #             self.w2* potential * vel_abs * (1.0 - 0.50* torch.cos(theta))
        cost = self.w1 * potential +\
                    self.w2 * potential * vel_abs * (1.0 +\
                                                     1.0 * (torch.max(-torch.cos(theta), torch.tensor(0.0).to(inp_device))) +\
                                                     0.8 * (torch.min(-torch.cos(theta), torch.tensor(0.0).to(inp_device)))
                                                     )
        

        cost = self.weight * cost.view(batch_size, horizon, 1)
        res = cost.squeeze(-1)

        judge_cost = self.weight * judge_cost.view(batch_size, horizon, 1)
        judge_res = judge_cost.squeeze(-1)

        return res.to(inp_device) , judge_res.to(inp_device) 

