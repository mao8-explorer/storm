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
from sklearn.linear_model import ModifiedHuber
from numpy import roll
import torch
import torch.nn as nn
# import torch.nn.functional as F

from ...differentiable_robot_model.coordinate_transform import CoordinateTransform, quaternion_to_matrix

from ...util_file import get_assets_path, join_path
from ...geom.sdf.robot_world import RobotWorldCollisionVoxel
from .gaussian_projection import GaussianProjection


from ..utils.mppi_scn import MPPIPolicy 

class ScenecollisionCost(nn.Module):
    def __init__(self, 
                 mppi_params = None, robot_params=None,
                 weight=None,
                 gaussian_params={}, 
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        super(ScenecollisionCost, self).__init__()
        self.tensor_args = tensor_args
        self.device = tensor_args['device']
        self.float_dtype = tensor_args['dtype']

        self.weight = torch.as_tensor(weight, **self.tensor_args)
        
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)


        self.policy = MPPIPolicy()


        self.dof = robot_params['robot_collision_params']['dof']
        self.batch_size = mppi_params['num_particles']
        self.horizen = mppi_params['horizon']


    def forward(self, rollouts,link_pos_batch,link_rot_batch):
 

        # 500 * 30 * 7 -> squeeze to 500 * 1 * 7
        # squeeze_rollouts = rollouts[:,10,:].squeeze()

        

        # link_pos_batch  size is 500 * 30 * 9 * 3 - > 9 * （500 * 30） * 3
        

        colls_value = self.policy._check_collisions(link_pos_batch = link_pos_batch ,link_rot_batch = link_rot_batch,modify = True).T.view(self.batch_size,self.horizen,-1)

        # (
        #  colls_by_link,
        #  colls_all, 
        #  colls_value,
        #  ) =  self.policy._check_collisions(rollouts)
        # current_colls_value =  self.policy._check_collisions(rollouts[0,:].view(1,-1))

        # not modify : safe but maybe slow
        # rollouts = rollouts.view(-1,self.dof).to(
        #             device=self.device,
        #             dtype=self.float_dtype)
        # colls_value =  self.policy._check_collisions(rollouts =rollouts ).T.view(self.batch_size,self.horizen,-1)

        
        # colls_value  是  num_links  *  lens(rollouts) 也就是 9*3000  rollouts 输入的格式是 3000*7 要想还原回去 要转置



        # cost = torch.sum(colls_value[:,:,:4],dim = -1)*10 + torch.sum(colls_value[:,:,4:6],dim = -1)*10 + torch.sum(colls_value[:,:,6:],dim = -1) * 0.5
        cost = torch.sum(colls_value[:,:,:],dim = -1)*10

        
        
        # print(
        #     # current_colls_value.view(-1).cpu().numpy(),  
        #       cost[0,:].cpu().numpy()
        #       )


        # cost = self.weight * self.proj_gaussian(cost)
        # cost = self.weight * cost
        
        return cost.to(
                    device=self.device,
                    dtype=self.float_dtype)