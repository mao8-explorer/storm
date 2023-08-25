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


class terminalCost(nn.Module):
    """ 
    add terminal cost to minimize error 
    """
    def __init__(self, weight, horizon, tensor_args={'device':"cpu", 'dtype':torch.float32}):

        super(terminalCost, self).__init__()
        self.tensor_args = tensor_args
        self.weight = weight
        self.horizon = horizon
        self.dtype = self.tensor_args['dtype']
        self.device = self.tensor_args['device']

    def forward(self, ee_pos_batch, ee_goal_pos):

    
        inp_device = ee_pos_batch.device
        ee_pos_batch = ee_pos_batch.to(device=self.device,
                                       dtype=self.dtype)

        ee_goal_pos = ee_goal_pos.to(device=self.device,
                                     dtype=self.dtype)

        cost = self.weight * torch.linalg.norm( ee_pos_batch[:,self.horizon,:]- ee_goal_pos, ord=2, dim=1)
 
        
        return cost.to(inp_device)

