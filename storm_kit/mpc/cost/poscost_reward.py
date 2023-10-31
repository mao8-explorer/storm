"""
在有IK求解的接触上,还要使用pose_quaternion意义不大.特别注意到quaternion计算量过大:
------  
           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
---------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------   
 cart_goal_cost         8.11%       2.578ms        56.17%      17.856ms      17.856ms     962.000us        10.34%       2.089ms       2.089ms             1  
  jnq_goal_cost         1.03%     328.000us         4.18%       1.329ms       1.329ms     845.000us         9.08%       1.329ms       1.329ms             1  
 zero_vel_bound         1.26%     399.000us         5.95%       1.892ms       1.892ms     774.000us         8.32%       1.991ms       1.991ms             1

显然 拭去quaternion是合理的
"""
import torch
import torch.nn as nn
from .gaussian_projection import GaussianProjection
from storm_kit.differentiable_robot_model.coordinate_transform import matrix_to_quaternion

class PoseCost_Reward(nn.Module):
    """ Pose cost using quaternion distance for orienatation 

    .. math::
     
    r  &=  \sum_{i=0}^{num_rows} (R^{i,:} - R_{g}^{i,:})^2 \\
    cost &= \sum w \dot r

    
    """
    def __init__(self, pose_weight, reward_weight, position_gaussian_params={}, sigma = None, tensor_args={'device':"cpu", 'dtype':torch.float32}):

        super(PoseCost_Reward, self).__init__()
        self.tensor_args = tensor_args
        self.pose_weight = torch.as_tensor(pose_weight, **self.tensor_args)
        self.reward_weight = torch.as_tensor(reward_weight, **self.tensor_args)
        self.sigma = sigma
        self.position_gaussian = GaussianProjection(gaussian_params=position_gaussian_params)
        self.dtype = self.tensor_args['dtype']
        self.device = self.tensor_args['device']
    

    def forward(self, ee_pos_batch, ee_goal_pos):

        inp_device = ee_pos_batch.device
        # ee_pos_batch = ee_pos_batch.to(device=self.device,
        #                                dtype=self.dtype)
        # ee_goal_pos = ee_goal_pos.to(device=self.device,
        #                              dtype=self.dtype)
        
        goal_dist = ee_pos_batch - ee_goal_pos
        # position_err = torch.norm(self.pos_weight * goal_dist)
        dist_sq = torch.sum(goal_dist**2, dim=-1, keepdim=False)

        # pose_cost : distance punish
        position_err = torch.sqrt(dist_sq)
        pose_cost =  self.pose_weight * self.position_gaussian(position_err)
    
        # reward design
        # 现在有这样的需求： 在获得了 数据大小为 batch * horizon 的 distance后，这些distance表示的是小车当前位置状态与目标点的欧拉距离，
        # 需要设计这样的一个代价函数 ： y = 1 - exp (-dist-sq/2*σ^2) , 其中σ= 0.03, 
        # 这样的一个函数的物理意义是：当小车靠近目标点，距离约0.05米后，越靠近目标点，惩罚越小或者说奖励越大
        exp_factor = -dist_sq / (2 * self.sigma**2)
        reward_cost = self.reward_weight * (1 - torch.exp(exp_factor))

        cost = pose_cost + reward_cost
        return cost.to(inp_device)

        