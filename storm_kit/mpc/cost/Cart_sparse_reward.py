import torch
import torch.nn as nn


class CartSparseReward(nn.Module):
    def __init__(self, weight=None, sigma=None, tensor_args={'device':"cpu", 'dtype':torch.float32}, **kwargs):
        super(CartSparseReward, self).__init__()
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight, **self.tensor_args)
        self.sigma = sigma

    
    def forward(self, ee_pos_batch,ee_goal_pos):
        inp_device = ee_pos_batch.device
        ee_pos_batch = ee_pos_batch.to(**self.tensor_args)
        ee_goal_pos = ee_goal_pos.to(**self.tensor_args)
        goal_dist = ee_pos_batch - ee_goal_pos

        dist_sq = torch.sum(goal_dist**2, dim=-1, keepdim=False)

        # 现在有这样的需求： 在获得了 数据大小为 batch * horizon 的 distance后，这些distance表示的是小车当前位置状态与目标点的欧拉距离，
        # 需要设计这样的一个代价函数 ： y = 1 - exp (-dist-sq/2*σ^2) , 其中σ= 0.03, 
        # 这样的一个函数的物理意义是：当小车靠近目标点，距离约0.05米后，越靠近目标点，惩罚越小或者说奖励越大
        exp_factor = -dist_sq / (2 * self.sigma**2)
        cost = self.weight * (1 - torch.exp(exp_factor))
        return cost.to(inp_device)
