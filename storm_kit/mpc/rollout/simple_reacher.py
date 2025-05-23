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

from ast import Assign
import torch

from ...mpc.cost import DistCost, ZeroCost, FiniteDifferenceCost ,SparseReward
from ...mpc.cost.stop_cost import StopCost
from ...mpc.model.simple_model import HolonomicModel
from ...mpc.cost.circle_collision_cost import CircleCollisionCost
from ...mpc.cost.image_collision_cost import ImageCollisionCost
from ...mpc.cost.image_moveable_collision_cost import ImagemoveCollisionCost
from ...mpc.cost.bound_cost import BoundCost
from ...mpc.model.integration_utils import build_fd_matrix, tensor_linspace
from ...util_file import join_path, get_assets_path


class SimpleReacher(object):
    """
    This rollout function is for reaching a cartesian pose for a robot

    """

    def __init__(self, exp_params, tensor_args={'device':'cpu','dtype':torch.float32}):
        self.tensor_args = tensor_args
        self.exp_params = exp_params
        device = self.tensor_args['device']
        float_dtype = self.tensor_args['dtype']
        # 权重全部提取！ 很重要！
        # self.goal_state_weight = exp_params['cost']['goal_state']['weight']
        mppi_params = exp_params['mppi']

        # initialize dynamics model:
        dynamics_horizon = mppi_params['horizon'] # model_params['dt']
        #Create the dynamical system used for rollouts

        self.dynamics_model = HolonomicModel(dt=exp_params['model']['dt'],
                                             dt_traj_params=exp_params['model']['dt_traj_params'],
                                             horizon=mppi_params['horizon'],
                                             batch_size=mppi_params['num_particles'],
                                             tensor_args=self.tensor_args,
                                             control_space=exp_params['control_space'])

        self.dt = self.dynamics_model.dt
        self.n_dofs = self.dynamics_model.n_dofs
        # rollout traj_dt starts from dt->dt*(horizon+1) as tstep 0 is the current state
        self.traj_dt = self.dynamics_model._dt_h #torch.arange(self.dt, (mppi_params['horizon'] + 1) * self.dt, self.dt,**self.tensor_args)

        self.goal_state = None
        

        self.goal_cost = DistCost(**exp_params['cost']['goal_state'], # 目标限制
                                  device=device,float_dtype=float_dtype)
        
        self.sparse_reward = SparseReward(**exp_params['cost']['sparse_reward'], # 目标限制
                                  tensor_args=self.tensor_args)

        self.stop_cost = StopCost(**exp_params['cost']['stop_cost'], # 速度限制
                                  tensor_args=self.tensor_args,
                                  traj_dt=self.dynamics_model.traj_dt)
        self.stop_cost_acc = StopCost(**exp_params['cost']['stop_cost_acc'], # 加速度限制
                                      tensor_args=self.tensor_args,
                                      traj_dt=self.dynamics_model.traj_dt)

        # self.zero_vel_cost = ZeroCost(device=self.tensor_args['device'], float_dtype=self.tensor_args['dtype'], **exp_params['cost']['zero_vel'])

        # self.fd_matrix = build_fd_matrix(10 - self.exp_params['cost']['smooth']['order'], device=self.tensor_args['device'], dtype=self.tensor_args['dtype'], PREV_STATE=True, order=self.exp_params['cost']['smooth']['order'])

        # self.smooth_cost = FiniteDifferenceCost(**self.exp_params['cost']['smooth'],
        #                                         tensor_args=self.tensor_args)

        
        self.image_collision_cost = ImageCollisionCost(
            **self.exp_params['cost']['image_collision'], bounds=exp_params['model']['position_bounds'],
            tensor_args=self.tensor_args)

        self.image_move_collision_cost = ImagemoveCollisionCost(
            **self.exp_params['cost']['image_move_collision'], bounds=exp_params['model']['position_bounds'],
            tensor_args=self.tensor_args)
        
        self.bound_cost = BoundCost(**exp_params['cost']['state_bound'],
                                    tensor_args=self.tensor_args,
                                    bounds=exp_params['model']['position_bounds'])

        self.terminal_cost = ImageCollisionCost(
            **self.exp_params['cost']['terminal'],
            bounds=exp_params['model']['position_bounds'],
            collision_file=self.exp_params['cost']['image_collision']['collision_file'],
            dist_thresh=self.exp_params['cost']['image_collision']['dist_thresh'],
            tensor_args=self.tensor_args)

        multimodal_mppi_costs = exp_params['multimodal_cost']
        if multimodal_mppi_costs['switch_on']: # 启动并行 则权重均归1，隔离single_mppi
            self.multiTargetCost = multimodal_mppi_costs['target_cost']
            self.multiCollisionCost = multimodal_mppi_costs['coll_cost']
            self.multiTerminalCost = multimodal_mppi_costs['terminal_reward']
            # 权重归1 重新分配
            self.goal_cost.weight = torch.tensor(1.0, **self.tensor_args)
            self.image_move_collision_cost.weight = torch.tensor(1.0, **self.tensor_args)
            self.sparse_reward.weight = torch.tensor(1.0, **self.tensor_args)

            self.multiTargetCost_greedy_weight = self.multiTargetCost['greedy_weight']
            self.multiCollisionCost_greedy_weight = self.multiCollisionCost['greedy_weight']
            self.multiTerminalCost_greedy_weight = self.multiTerminalCost['greedy_weight']
            self.multiTargetCost_sensi_weight = self.multiTargetCost['sensi_weight']
            self.multiCollisionCost_sensi_weight = self.multiCollisionCost['sensi_weight']
            self.multiTerminalCost_sensi_weight = self.multiTerminalCost['sensi_weight']
            self.multiTargetCost_judge_weight = self.multiTargetCost['judge_weight']
            self.multiCollisionCost_judge_weight = self.multiCollisionCost['judge_weight']
            self.multiTerminalCost_judge_weight =  self.multiTerminalCost['judge_weight']

    def cost_fn(self, state_dict, action_batch, no_coll=False, horizon_cost=True, return_dist=False):
        

        state_batch = state_dict['state_seq']
        #print(action_batch)

        goal_state = self.goal_state.unsqueeze(0)
        
        cost = self.goal_cost.forward(goal_state - state_batch[:,:,:self.n_dofs]) #!

        if self.exp_params['cost']['sparse_reward']['weight'] > 0: #!
            cost += self.sparse_reward.forward(goal_state - state_batch[:,:,:self.n_dofs])
      
        if(horizon_cost):
            if self.exp_params['cost']['stop_cost']['weight'] > 0: #!
                vel_cost = self.stop_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs * 2])
                cost += vel_cost
            if self.exp_params['cost']['stop_cost_acc']['weight'] > 0:
                acc_cost = self.stop_cost_acc.forward(state_batch[:, :, self.n_dofs*2:self.n_dofs * 3])
                cost += acc_cost

        if self.exp_params['cost']['image_collision']['weight'] > 0:
            # compute collision cost:
            coll_cost = self.image_collision_cost.forward(state_batch[:,:,:self.n_dofs])
            #print (coll_cost.shape)
            cost += coll_cost
            
        if self.exp_params['cost']['image_move_collision']['weight'] > 0: #!
            # compute collision cost:
            coll_cost , judge_cost, _ = self.image_move_collision_cost.forward(state_batch[:,:,:2*self.n_dofs])
            #print (coll_cost.shape)
            cost += coll_cost


        if self.exp_params['cost']['state_bound']['weight'] > 0: #!
            # compute collision cost:
            bound_contraint= self.bound_cost.forward(state_batch[:,:,:self.n_dofs])
            cost += bound_contraint

        if self.exp_params['cost']['terminal']['weight'] > 0:
            # terminal cost:
            B, H, N = state_batch.shape
            # sample linearly from terminal position to goal:
            linear_pos_batch = torch.zeros_like(state_batch[:,:,:self.n_dofs])
            for i in range(self.n_dofs):
                data = tensor_linspace(state_batch[:,:,i], goal_state[0,0,i], H)
                linear_pos_batch[:,:,i] = data
            #print(linear_pos_batch.shape)
            term_cost = self.terminal_cost.forward(linear_pos_batch)
            #print(term_cost.shape, cost.shape)
            
            cost[:,-1] += torch.sum(term_cost, dim=-1)
        if(return_dist):
            disp_vec = goal_state - state_batch[:,:,:self.n_dofs]
            goal_dist = torch.norm(disp_vec, p=2, dim=-1,keepdim=False)
            return cost, goal_dist , coll_cost
        else:
            return cost

    def short_sighted_cost_fn(self, state_dict):

        state_batch = state_dict['state_seq']
        #print(action_batch)
        # goal_state changed to 
        goal_state = self.goal_state.unsqueeze(0)
        # goal_state = mean_traj_greedy[-1,:self.n_dofs]# 可适当调节  @测试！ 20 horizon [15 20]区间测试 增加路径弹性 需要根据测试结果讨论清楚 
        # * torch.cat((torch.zeros(10),torch.ones(10))).to(**self.tensor_args)
        # goal_state = mean_traj_greedy[ :,:self.n_dofs].unsqueeze(0)
        
        target_cost = self.goal_cost.forward(goal_state - state_batch[:,:,:self.n_dofs]) #!
        cost = target_cost  * (1.0 / self.exp_params['cost']['goal_state']['weight'] * 5.0) 
        #                 * torch.cat((torch.zeros(10),torch.ones(10))).to(**self.tensor_args)
        # if self.exp_params['cost']['sparse_reward']['weight'] > 0: #!
        # terminal_reward = self.sparse_reward.lazyforward(goal_state - state_batch[:,:,:self.n_dofs],sigma=0.03)
        # cost = terminal_reward * (1.0 / self.exp_params['cost']['sparse_reward']['weight'] * 10.0) *  torch.cat((torch.zeros(17),torch.ones(3))).to(**self.tensor_args)
        # 速度限制 禁止越界
        vel_cost = self.stop_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs * 2])
        cost += vel_cost
        
        coll_cost ,judge_cost,_ = self.image_move_collision_cost.forward(state_batch[:,:,:2*self.n_dofs])
        cost += coll_cost * 5.0

        bound_contraint= self.bound_cost.forward(state_batch[:,:,:self.n_dofs])
        cost += bound_contraint

        return cost

    def multimodal_cost_fn(self, state_dict):

        state_batch = state_dict['state_seq']
        goal_state = self.goal_state.unsqueeze(0)
        
        self.target_cost = self.goal_cost.forward(goal_state - state_batch[:,:,:self.n_dofs])
        self.coll_cost, self.judge_coll_cost, self.greedy_coll = self.image_move_collision_cost.forward(state_batch[:,:,:2*self.n_dofs])

        # 速度限制 禁止越界
        self.terminal_reward = self.sparse_reward.forward(goal_state - state_batch[:,:,:self.n_dofs])
        self.vel_cost = self.stop_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs * 2])
        self.bound_contraint= self.bound_cost.forward(state_batch[:,:,:self.n_dofs])
 


    def rollout_fn(self, start_state, act_seq):
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Args:
        
            action_seq: torch.Tensor [num_particles, horizon, d_act]
        """
        # rollout_start_time = time.time()
        #print("computing rollout")
        #print(act_seq)
        #print('step...')
        state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq) # 状态forward
        #if('act_seq' in state_dict):
        #    act_seq = state_dict['act_seq']
            #print('action')
        #states = state_dict['state_seq']
        #acc = states[:,:, self.n_dofs*2: self.n_dofs*3]
        '''
        fig, axs = plt.subplots(4)
        acc = act_seq.cpu()
        for i in range(10):
            axs[3].plot(acc[i,:,0])


        states = state_dict['state_seq']
        acc = states[:,:, self.n_dofs*2: self.n_dofs*3]
        
        for i in range(10):
            axs[2].plot(acc[i,:,0])

        acc = states[:,:, self.n_dofs*1: self.n_dofs*2]

        for i in range(10):
            axs[1].plot(acc[i,:,0])
        acc = states[:,:, : self.n_dofs]
        for i in range(10):
            axs[0].plot(acc[i,:,0])

        plt.show()
        '''
        #link_pos_seq, link_rot_seq = self.dynamics_model.get_link_poses()
        
        cost_seq = self.cost_fn(state_dict,act_seq)
        
        sim_trajs = dict(
            actions=act_seq,#.clone(),
            costs=cost_seq,#clone(),
            rollout_time=0.0,
            state_seq=state_dict['state_seq']
        )
        
        return sim_trajs
    
    def short_sighted_rollout_fn(self, start_state, act_seq):
        state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq) # 状态forward

        cost_seq = self.short_sighted_cost_fn(state_dict)
        
        sim_trajs = dict(
            actions=act_seq,#.clone(),
            costs=cost_seq,#clone(),
            rollout_time=0.0,
            state_seq=state_dict['state_seq']
        )
        return sim_trajs


    def multimodal_rollout_fn(self, start_state, act_seq):
        state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq) # 状态forward

        """
        1. greedy_policy
        2. sensitive_policy
        self.multiTargetCost = multimodal_mppi_costs['target_cost']
        self.multiCollisionCost = multimodal_mppi_costs['coll_cost']
        self.multiTerminalCost = multimodal_mppi_costs['terminal_reward']
        """
        self.multimodal_cost_fn(state_dict)
        greedy_cost_seq = self.target_cost * self.multiTargetCost_greedy_weight +\
                          self.judge_coll_cost * self.multiCollisionCost_greedy_weight +\
                          self.terminal_reward * self.multiTerminalCost_greedy_weight+\
                          self.vel_cost + self.bound_contraint 
        
        sensi_cost_seq =  self.target_cost * self.multiTargetCost_sensi_weight +\
                          self.coll_cost * self.multiCollisionCost_sensi_weight  +\
                          self.terminal_reward * self.multiTerminalCost_sensi_weight +\
                          self.vel_cost + self.bound_contraint 
        
        judge_cost_seq = self.target_cost * self.multiTargetCost_judge_weight+\
                         self.judge_coll_cost * self.multiCollisionCost_judge_weight  +\
                         self.terminal_reward * self.multiTerminalCost_judge_weight
        
        sim_trajs = dict(
            actions=act_seq,#.clone(),
            greedy_costs=greedy_cost_seq,#clone(),
            sensi_costs=sensi_cost_seq,
            judge_costs=judge_cost_seq,
            rollout_time=0.0,
            state_seq=state_dict['state_seq']
        )
        return sim_trajs        



    def single_state_forward(self,start_state,act_seq):
        single_state_seq = self.dynamics_model.single_step_fn(start_state, act_seq) # 状态forward
        return single_state_seq



    def update_params(self, goal_state=None):
        """
        Updates the goal targets for the cost functions.
        goal_state: n_dofs
        goal_ee_pos: 3
        goal_ee_rot: 3,3
        goal_ee_quat: 4

        """
        self.goal_state = torch.as_tensor(goal_state, **self.tensor_args).unsqueeze(0)
        
        return True
    def __call__(self, start_state, act_seq):
        return self.rollout_fn(start_state, act_seq)
    
    def current_cost(self, current_state):
        current_state = current_state.to(**self.tensor_args).unsqueeze(0)
        
        curr_batch_size = 1
        num_traj_points = 1
        state_dict = {'state_seq': current_state}

        cost, goal_dist, coll_cost= self.cost_fn(state_dict, None,no_coll=False, horizon_cost=False, return_dist=True)
        return cost, state_dict ,goal_dist ,coll_cost
