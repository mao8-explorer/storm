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

import math
# import torch.nn.functional as F

from .gaussian_projection import GaussianProjection
from storm_kit.differentiable_robot_model.coordinate_transform import matrix_to_quaternion

class PoseCostQuaternion(nn.Module):
    """ Pose cost using quaternion distance for orienatation 

    .. math::
     
    r  &=  \sum_{i=0}^{num_rows} (R^{i,:} - R_{g}^{i,:})^2 \\
    cost &= \sum w \dot r

    
    """
    def __init__(self, weight, rotposweight=[],vec_weight=[], position_gaussian_params={}, orientation_gaussian_params={}, tensor_args={'device':"cpu", 'dtype':torch.float32}, hinge_val=100.0,
                 convergence_val=[0.0,0.0]):

        super(PoseCostQuaternion, self).__init__()
        self.tensor_args = tensor_args
        self.I = torch.eye(3,3, **tensor_args)
        self.weight = weight
        self.rotposweight = rotposweight
        self.vec_weight = torch.as_tensor(vec_weight, **tensor_args)
        self.rot_weight = self.vec_weight[0:3]
        self.pos_weight = self.vec_weight[3:6]

        self.px = torch.tensor([1.0,0.0,0.0], **self.tensor_args)
        self.py = torch.tensor([0.0,1.0,0.0], **self.tensor_args)
        self.pz = torch.tensor([0.0,0.0,1.0], **self.tensor_args)
        
        self.I = torch.eye(3,3,**self.tensor_args)
        self.Z = torch.zeros(1, **self.tensor_args)


        self.position_gaussian = GaussianProjection(gaussian_params=position_gaussian_params)
        self.orientation_gaussian = GaussianProjection(gaussian_params=orientation_gaussian_params)
        self.hinge_val = hinge_val
        self.convergence_val = convergence_val
        self.dtype = self.tensor_args['dtype']
        self.device = self.tensor_args['device']
    

    def forward(self, ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot):

        
        inp_device = ee_pos_batch.device
        ee_pos_batch = ee_pos_batch.to(device=self.device,
                                       dtype=self.dtype)
        ee_rot_batch = ee_rot_batch.to(device=self.device,
                                       dtype=self.dtype)
        ee_goal_pos = ee_goal_pos.to(device=self.device,
                                     dtype=self.dtype)
        ee_goal_rot = ee_goal_rot.to(device=self.device,
                                     dtype=self.dtype)
    
        ee_quat_batch = matrix_to_quaternion(ee_rot_batch)
        ee_goal_quat = matrix_to_quaternion(ee_goal_rot)
        
        
        #Inverse of goal transform
        # R_g_t = ee_goal_rot.transpose(-2,-1) # w_R_g -> g_R_w
        # R_g_t_d = (-1.0 * R_g_t @ ee_goal_pos.t()).transpose(-2,-1) # -g_R_w * w_d_g -> g_d_g

        
        #Rotation part
        # R_g_ee = R_g_t @ ee_rot_batch # g_R_w * w_R_ee -> g_R_ee
        
        
        #Translation part
        # transpose is done for matmul
        # term1 = (R_g_t @ ee_pos_batch.transpose(-2,-1)).transpose(-2,-1) # g_R_w * w_d_ee -> g_d_ee
        # d_g_ee = term1 + R_g_t_d # g_d_g + g_d_ee
        # goal_dist = torch.norm(self.pos_weight * d_g_ee, p=2, dim=-1, keepdim=True)
        goal_disp = ee_pos_batch - ee_goal_pos
        # goal_dist = torch.norm(self.pos_weight * goal_disp)
        position_err = torch.sqrt((torch.sum(torch.square(self.pos_weight * goal_disp),dim=-1)))


        #compute projection error
        # rot_err = self.I - R_g_ee
        # rot_err = torch.norm(rot_err, dim=-1)
        # rot_err_norm = torch.norm(torch.sum(self.rot_weight * rot_err,dim=-1), p=2, dim=-1, keepdim=True)
        quat_x = ee_quat_batch[:,:,0] * ee_goal_quat[0,0]
        quat_y = ee_quat_batch[:,:,1] * ee_goal_quat[0,1]
        quat_z = ee_quat_batch[:,:,2] * ee_goal_quat[0,2]
        quat_w = ee_quat_batch[:,:,3] * ee_goal_quat[0,3]

        rot_err = quat_x + quat_y + quat_z + quat_w
        # rot_err_norm = torch.norm(rot_err , p=2, dim=-1, keepdim=True)
        rot_err = 2 * torch.square(rot_err) - 1
        # -1 ~ 1 限幅 因为  matrix_to_quaternion 和 四元数内积都不能保证torch.acos()的正确性
        rot_err[rot_err > 1] = 1.0
        rot_err[rot_err < -1] = -1.0
        rot_err = torch.acos(rot_err)
        #  0 - pi 区间 表征 quaternion distance


        # rot_err = 2.0 * torch.acos(rot_err)

        # #normalize to -pi/2,pi/2
        # rot_err = torch.atan2(torch.sin(rot_err), torch.cos(rot_err))
        # # print(rot_err)
        # print(torch.sum(torch.isnan(rot_err)))
        # # if(self.hinge_val > 0.0):
        # #     rot_err = torch.where(goal_dist.squeeze(-1) <= self.hinge_val, rot_err, self.Z) #hard hinge
        #
        # # rot_err = torch.sqrt(torch.square(rot_err)) # 考虑到负角 修正
        # rot_err = torch.abs(rot_err)  # 0 ~ pi/2


        rot_err[rot_err < self.convergence_val[0]] = 0.0
        position_err[position_err < self.convergence_val[1]] = 0.0
        # cost = self.weight[0] * self.orientation_gaussian(torch.sqrt(rot_err)) + self.weight[1] * self.position_gaussian(torch.sqrt(position_err))
        # cost = self.weight[0] * self.orientation_gaussian(rot_err) + self.weight[1] * self.position_gaussian(position_err)
        
        cost = (self.weight*self.rotposweight[0]) * self.orientation_gaussian(rot_err) + (self.weight*self.rotposweight[1]) * self.position_gaussian(position_err)
        # dimension should be bacth * traj_length
        return cost.to(inp_device)


        