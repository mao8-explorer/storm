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
import yaml
# import torch.nn.functional as F
from ...geom.geom_types import tensor_circle
from .gaussian_projection import GaussianProjection
from ...util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from ...geom.sdf.world import WorldImageCollision
from ...geom.sdf.world import WorldMoveableImageCollision


class ImagemoveCollisionCost(nn.Module):
    def __init__(self, weight=None, collision_file=None, bounds=[], dist_thresh=0.01, gaussian_params={}, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        super(ImagemoveCollisionCost, self).__init__()
        
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight,**self.tensor_args)
        
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        
        # BUILD world and robot:
        world_image = join_path(get_assets_path(), collision_file)
        

        self.world_coll = WorldMoveableImageCollision(bounds=bounds, world_image = world_image ,tensor_args=tensor_args)
        self.dist_thresh = dist_thresh # meters
        
    def forward(self, state_seq):
        
        inp_device = state_seq.device
        batch_size = state_seq.shape[0]
        horizon = state_seq.shape[1]
        pos_batch = state_seq[:,:,:2].view(batch_size * horizon, 2)
        vel_batch = state_seq[:,:,2:4].view(batch_size * horizon, 2)

        
        # query sdf for points:
        dist = self.world_coll.get_pt_value(pos_batch)
        
        #step 1:  计算根据 vel_batch 计算速度的绝对值|vel_batch|  =  sqrt(vel_batch[:,0]^2 + vel_batch[：,1]^2)
        vel_abs = torch.linalg.norm(vel_batch,ord=2,dim=1)
        #step 2:  将速度绝对值与dist相乘 dist * |vel_batch| 获得size为（batch_size, horizon, 1）的代价值
        cost =  dist
        cost = cost.view(batch_size, horizon, 1)
        # cost only when dist is less

        res = self.weight * cost
        


        res = res.squeeze(-1)
        
        return res.to(inp_device)



