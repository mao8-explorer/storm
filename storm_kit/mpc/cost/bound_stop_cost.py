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
from .gaussian_projection import GaussianProjection

class Bound_StopCost(nn.Module):
    def __init__(self, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float64},
                 bounds=[], bound_weight = 1.0, stop_weight = 1.0, bound_gaussian_params = {}, bound_thresh=0.1,
                 max_nlimit=None,  stop_gaussian_params={}, traj_dt=None,**kwargs):
        super(Bound_StopCost, self).__init__()

        self.tensor_args = tensor_args
        self.bound_gaussian = GaussianProjection(gaussian_params=bound_gaussian_params)
        self.stop_gaussian = GaussianProjection(gaussian_params=stop_gaussian_params)
        self.traj_dt = traj_dt

        self.bound_weight = torch.as_tensor(bound_weight, **self.tensor_args)
        self.stop_weight = torch.as_tensor(stop_weight, **self.tensor_args)
        
        # bound cost setting
        self.bounds = torch.as_tensor(bounds, **tensor_args)
        self.bnd_range = (self.bounds[:,1] - self.bounds[:,0]) / 2.0
        self.t_mat = None
        self.bound_thresh = bound_thresh * self.bnd_range
        self.bounds[:,1] -= self.bound_thresh
        self.bounds[:,0] += self.bound_thresh

        # stop cost setting
        # compute max velocity across horizon:
        self.horizon = self.traj_dt.shape[0]
        sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), **self.tensor_args)).T
        if(max_nlimit is not None):
            # every timestep max acceleration:
            sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), **self.tensor_args)).T
            delta_vel = self.traj_dt * max_nlimit
            self.max_vel = ((sum_matrix @ delta_vel).unsqueeze(-1))

        
    def forward(self, state_batch):
        inp_device = state_batch.device


        # bound cost : state_batch[:,:,:self.n_dofs * 3]
        bound_mask = torch.logical_and(state_batch < self.bounds[:,1],
                                       state_batch > self.bounds[:,0])

        cost = torch.minimum(torch.square(state_batch - self.bounds[:,0]),torch.square(self.bounds[:,1] - state_batch))
        
        cost[bound_mask] = 0.0

        cost = (torch.sum(cost, dim=-1))
        bound_cost = self.bound_weight * self.bound_gaussian(torch.sqrt(cost))
        

        # stop cost : vels = state_batch[:, :, self.n_dofs:self.n_dofs * 2]
        vels = state_batch[:, :, self.n_dofs:self.n_dofs * 2]
        vel_abs = torch.abs(vels.to(**self.tensor_args))
        vel_abs = vel_abs - self.max_vel
        vel_abs[vel_abs < 0.0] = 0.0
        stop_cost = self.stop_weight * self.stop_gaussian(((torch.sum(torch.square(vel_abs), dim=-1))))

        
        return (bound_cost + stop_cost).to(inp_device)
