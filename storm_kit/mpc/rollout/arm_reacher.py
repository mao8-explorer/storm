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
import torch.autograd.profiler as profiler

from ...differentiable_robot_model.coordinate_transform import matrix_to_quaternion, quaternion_to_matrix
from ..cost import DistCost, PoseCost, PoseCostQuaternion,PoseCost_Reward, ZeroCost, FiniteDifferenceCost,terminalCost , JnqSparseReward,CartSparseReward
from ...mpc.rollout.arm_base import ArmBase
import queue

class ArmReacher(ArmBase):
    """
    This rollout function is for reaching a cartesian pose for a robot

Todo: 
    1. Update exp_params to be kwargs
    """

    def __init__(self, exp_params, tensor_args={'device':"cpu", 'dtype':torch.float32}, world_params=None):
        super(ArmReacher, self).__init__(exp_params=exp_params,
                                         tensor_args=tensor_args,
                                         world_params=world_params)
        
        self.goal_state = None
        self.goal_ee_pos = None
        self.goal_ee_rot = None
        self.goal_jnq = None
        self.jnq_mask = None
        self.curr_ee_pos = None

        device = self.tensor_args['device']
        float_dtype = self.tensor_args['dtype']
        self.dist_cost = DistCost(**self.exp_params['cost']['joint_l2'], device=device,float_dtype=float_dtype)

        # todo : PoseCost | PoseCostQuaternion compare

        # self.goal_cost = PoseCost(**exp_params['cost']['goal_pose'],
        #                           tensor_args=self.tensor_args)
        #
        self.goal_pq_cost = PoseCostQuaternion(**exp_params['cost']['goal_pose'],
                                  tensor_args=self.tensor_args)
        
        self.goal_cost_reward = PoseCost_Reward(**exp_params['cost']['PoseCost_Reward'], # Cartesian space target
                                  tensor_args=self.tensor_args)
        
        self.jnq_sparse_reward = JnqSparseReward(**exp_params['cost']['Jnq_sparse_reward'], # 目标限制
                                  tensor_args=self.tensor_args)
        
        # self.cart_sparse_reward = CartSparseReward(**exp_params['cost']['Cart_sparse_reward'], # 目标限制
        #                           tensor_args=self.tensor_args)
        
        # self.terminal_cost = terminalCost(**exp_params['cost']['terminal_pos'],
        #                           tensor_args=self.tensor_args)
        
    def cost_fn(self, state_dict, action_batch, no_coll=False, horizon_cost=True, return_dist=False):

        cost = super(ArmReacher, self).cost_fn(state_dict, action_batch, no_coll, horizon_cost)
        l_ee_pos_batch = state_dict['l_ee_pos_seq']
        l_ee_rot_batch = state_dict['l_ee_rot_seq']
        self.curr_ee_pos = l_ee_pos_batch[-1,0,:]
        
        state_batch = state_dict['state_seq']
        l_goal_ee_pos = self.l_goal_ee_pos
        l_goal_ee_rot = self.l_goal_ee_rot
        # 为什么要存在 因为逆解不存在时，也就是全局规划无解时，可以使用该方式引导
        goal_pq_cost = self.goal_pq_cost.forward(l_ee_rot_batch, l_goal_ee_rot)

        #  pose sparse_reward design 加快末端位置收敛 
        cart_goal_cost, cart_sparse_reward = self.goal_cost_reward.forward(l_ee_pos_batch, l_goal_ee_pos)
        cost +=  cart_sparse_reward  + cart_goal_cost + goal_pq_cost

        if self.goal_jnq is not None:
            disp_vec = state_batch[:,:,0:self.n_dofs] - self.goal_jnq[:,0:self.n_dofs]
            if(self.exp_params['cost']['joint_l2']['weight'] > 0.0):
                cost += self.dist_cost.forward(disp_vec)

            if self.exp_params['cost']['Jnq_sparse_reward']['weight'] > 0: #!
                cost += self.jnq_sparse_reward.forward(disp_vec)

            if self.exp_params['cost']['zero_vel']['weight'] > 0:
                cost += self.zero_vel_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs*2], goal_dist=disp_vec)
          
        return cost


    def dual_cost_fn(self, state_dict, action_batch, no_coll=False, horizon_cost=True, return_dist=False):
        cost = super(ArmReacher, self).cost_fn(state_dict, action_batch, no_coll, horizon_cost)

        # 末端执行器轨迹
        l_ee_pos_batch = state_dict['l_ee_pos_seq']
        l_ee_rot_batch = state_dict['l_ee_rot_seq']
        r_ee_pos_batch = state_dict['r_ee_pos_seq']
        r_ee_rot_batch = state_dict['r_ee_rot_seq']
        
        # 当前末端位置（用于调试或 early stop）
        self.curr_ee_pos = l_ee_pos_batch[-1, 0, :]

        # 状态轨迹
        state_batch = state_dict['state_seq']

        # ===============================
        # === 双臂末端目标 Cost（Tracking）
        # ===============================
        l_goal_ee_pos = self.l_goal_ee_pos
        l_goal_ee_rot = self.l_goal_ee_rot
        r_goal_ee_pos = self.r_goal_ee_pos
        r_goal_ee_rot = self.r_goal_ee_rot

        # === 左臂位姿 cost ===
        l_goal_pq_cost = self.goal_pq_cost.forward(l_ee_rot_batch, l_goal_ee_rot)
        l_cart_goal_cost, l_sparse_reward = self.goal_cost_reward.forward(l_ee_pos_batch, l_goal_ee_pos)

        # === 右臂位姿 cost ===
        r_goal_pq_cost = self.goal_pq_cost.forward(r_ee_rot_batch, r_goal_ee_rot)
        r_cart_goal_cost, r_sparse_reward = self.goal_cost_reward.forward(r_ee_pos_batch, r_goal_ee_pos)

        # 加入末端 tracking 代价项
        cost += (
            l_goal_pq_cost + l_cart_goal_cost + l_sparse_reward +
            r_goal_pq_cost + r_cart_goal_cost + r_sparse_reward
        )
        # todo: 优化 if , 将if去除
        # todo: 理清disp_vec本身的逻辑，速度限制下会怎么样？
        # todo: 理清如果goal_jnq为None时，整体的 cost_fn逻辑
        # if self.goal_jnq is not None:
        disp_vec = state_batch[:, :, 0:self.n_dofs] - self.goal_jnq[:, 0:self.n_dofs]
        disp_vec = disp_vec * self.jnq_mask  # 应用掩码

        if self.exp_params['cost']['joint_l2']['weight'] > 0.0:
            cost += self.dist_cost.forward(disp_vec)

        if self.exp_params['cost']['Jnq_sparse_reward']['weight'] > 0:
            cost += self.jnq_sparse_reward.forward(disp_vec)

        if self.exp_params['cost']['zero_vel']['weight'] > 0:
            cost += self.zero_vel_cost.forward(
                state_batch[:, :, self.n_dofs:2 * self.n_dofs],
                goal_dist=disp_vec, mask=self.jnq_mask)

        return cost




    def update_params(self, retract_state=None, goal_state=None, goal_ee_pos=None, goal_ee_rot=None, goal_ee_quat=None):
        """
        Update params for the cost terms and dynamics model.
        goal_state: n_dofs
        goal_ee_pos: 3
        goal_ee_rot: 3,3
        goal_ee_quat: 4

        """
        
        super(ArmReacher, self).update_params(retract_state=retract_state)
        
        if(goal_ee_pos is not None):
            self.goal_ee_pos = torch.as_tensor(goal_ee_pos, **self.tensor_args).unsqueeze(0)
            self.goal_state = None
        if(goal_ee_rot is not None):
            self.goal_ee_rot = torch.as_tensor(goal_ee_rot, **self.tensor_args).unsqueeze(0)
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)
            self.goal_state = None
        if(goal_ee_quat is not None):
            self.goal_ee_quat = torch.as_tensor(goal_ee_quat, **self.tensor_args).unsqueeze(0)
            self.goal_ee_rot = quaternion_to_matrix(self.goal_ee_quat)
            self.goal_state = None
        if(goal_state is not None):
            self.goal_state = torch.as_tensor(goal_state, **self.tensor_args).unsqueeze(0)
            self.goal_ee_pos, self.goal_ee_rot = self.dynamics_model.robot_model.compute_forward_kinematics(self.goal_state[:,0:self.n_dofs], 
                                            self.goal_state[:,self.n_dofs:2*self.n_dofs], link_name=self.exp_params['model']['ee_link_name'])
            self.goal_ee_quat = matrix_to_quaternion(self.goal_ee_rot)
        
        return True
        
    def dual_update_params(self, retract_state=None, goal_state=None, 
                        l_goal_ee_pos=None, l_goal_ee_rot=None, l_goal_ee_quat=None,
                        r_goal_ee_pos=None, r_goal_ee_rot=None, r_goal_ee_quat=None):
        """
        Update params for the cost terms and dual-arm dynamics model.

        Args:
            retract_state: 当前状态 [n_dofs]
            goal_state:     整体状态向量 [3*n_dofs]，可从中计算两个末端位姿
            l_goal_ee_pos:  左臂目标位置 [3]
            l_goal_ee_rot:  左臂目标旋转矩阵 [3,3]
            l_goal_ee_quat: 左臂目标旋转四元数 [4]
            r_goal_ee_pos:  右臂目标位置 [3]
            r_goal_ee_rot:  右臂目标旋转矩阵 [3,3]
            r_goal_ee_quat: 右臂目标旋转四元数 [4]
        """

        super(ArmReacher, self).update_params(retract_state=retract_state)

        # ========== 单独设置左臂目标 ==========
        if l_goal_ee_pos is not None:
            self.l_goal_ee_pos = torch.as_tensor(l_goal_ee_pos, **self.tensor_args).unsqueeze(0)
        if l_goal_ee_rot is not None:
            self.l_goal_ee_rot = torch.as_tensor(l_goal_ee_rot, **self.tensor_args).unsqueeze(0)
            self.l_goal_ee_quat = matrix_to_quaternion(self.l_goal_ee_rot)
        if l_goal_ee_quat is not None:
            self.l_goal_ee_quat = torch.as_tensor(l_goal_ee_quat, **self.tensor_args).unsqueeze(0)
            self.l_goal_ee_rot = quaternion_to_matrix(self.l_goal_ee_quat)

        # ========== 单独设置右臂目标 ==========
        if r_goal_ee_pos is not None:
            self.r_goal_ee_pos = torch.as_tensor(r_goal_ee_pos, **self.tensor_args).unsqueeze(0)
        if r_goal_ee_rot is not None:
            self.r_goal_ee_rot = torch.as_tensor(r_goal_ee_rot, **self.tensor_args).unsqueeze(0)
            self.r_goal_ee_quat = matrix_to_quaternion(self.r_goal_ee_rot)
        if r_goal_ee_quat is not None:
            self.r_goal_ee_quat = torch.as_tensor(r_goal_ee_quat, **self.tensor_args).unsqueeze(0)
            self.r_goal_ee_rot = quaternion_to_matrix(self.r_goal_ee_quat)

        # ========== 从 goal_state 计算两个末端目标 ==========
        if goal_state is not None:
            self.goal_state = torch.as_tensor(goal_state, **self.tensor_args).unsqueeze(0)
            q = self.goal_state[:, :self.n_dofs]
            qd = self.goal_state[:, self.n_dofs:2 * self.n_dofs]

            (l_pos, l_rot), (r_pos, r_rot) = self.dynamics_model.robot_model.compute_dual_forward_kinematics(
                q, qd,
                l_link_name=self.exp_params['model']['left_ee_link_name'],
                r_link_name=self.exp_params['model']['right_ee_link_name']
            )
            self.l_goal_ee_pos, self.l_goal_ee_rot = l_pos, l_rot
            self.r_goal_ee_pos, self.r_goal_ee_rot = r_pos, r_rot
            self.l_goal_ee_quat = matrix_to_quaternion(self.l_goal_ee_rot)
            self.r_goal_ee_quat = matrix_to_quaternion(self.r_goal_ee_rot)

        return True
