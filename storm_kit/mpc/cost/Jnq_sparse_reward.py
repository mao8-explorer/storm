import torch
import torch.nn as nn


class JnqSparseReward(nn.Module):
    def __init__(self, weight=None, vec_weight=None, sigma=None, tensor_args={'device':"cpu", 'dtype':torch.float32}, **kwargs):
        super(JnqSparseReward, self).__init__()
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight, **self.tensor_args)
        self.sigma = sigma
        if(vec_weight is not None):
            self.vec_weight = torch.as_tensor(vec_weight, **self.tensor_args)
        else:
            self.vec_weight = 1.0
    
    def forward(self, disp_vec):
        inp_device = disp_vec.device
        disp_vec = self.vec_weight * disp_vec.to(**self.tensor_args)
        dist_sq = torch.sum(disp_vec**2, dim=-1, keepdim=False)

        # 现在有这样的需求： 在获得了 数据大小为 batch * horizon 的 distance后，这些distance表示的是小车当前位置状态与目标点的欧拉距离，
        # 需要设计这样的一个代价函数 ： y = 1 - exp (-dist-sq/2*σ^2) , 其中σ= 0.03, 
        # 这样的一个函数的物理意义是：当小车靠近目标点，距离约0.05米后，越靠近目标点，惩罚越小或者说奖励越大

        exp_factor = -dist_sq / (2 * self.sigma**2)
        cost = self.weight * (1 - torch.exp(exp_factor))
        return cost.to(inp_device)
