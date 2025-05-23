#!/usr/bin/env python
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
from abc import ABC, abstractmethod
import copy

import numpy as np
import torch
import torch.autograd.profiler as profiler
import time


class Controller(ABC):
    """Base class for sampling based controllers."""

    def __init__(self,
                 d_action,
                 action_lows,
                 action_highs,
                 horizon,
                 gamma,
                 n_iters,
                 rollout_fn=None,
                 sample_mode='mean',
                 hotstart=True,
                 seed=0,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        """
        Defines an abstract base class for 
        sampling based MPC algorithms.

        Implements the optimize method that is called to 
        generate an action sequence for a given state and
        is common across sampling based controllers

        Attributes:
        
        d_action : int
            size of action space
        action_lows : torch.Tensor 
            lower limits for each action dim
        action_highs : torch.Tensor  
            upper limits for each action dim
        horizon : int  
            horizon of rollouts
        gamma : float
            discount factor
        n_iters : int  
            number of optimization iterations per MPC call
        rollout_fn : function handle  
            rollout policy (or actions) in simulator
            and return states and costs for updating MPC
            distribution
        sample_mode : {'mean', 'sample'}  
            how to choose action to be executed
            'mean' plays the first mean action and  
            'sample' samples from the distribution
        hotstart : bool
            If true, the solution from previous step
            is used to warm start current step
        seed : int  
            seed value
        device: torch.device
            controller can run on both cpu and gpu
        float_dtype: torch.dtype
            floating point precision for calculations
        """
        self.tensor_args = tensor_args
        self.d_action = d_action
        self.action_lows = action_lows.to(**self.tensor_args)
        self.action_highs = action_highs.to(**self.tensor_args)
        self.horizon = horizon
        self.gamma = gamma
        self.n_iters = n_iters
        self.gamma_seq = torch.cumprod(torch.tensor([1.0] + [self.gamma] * (horizon - 1)),dim=0).reshape(1, horizon)
        self.gamma_seq = self.gamma_seq.to(**self.tensor_args)
        self._rollout_fn = rollout_fn
        self.sample_mode = sample_mode
        self.num_steps = 0
        self.hotstart = hotstart
        self.seed_val = seed
        self.trajectories = None


    @abstractmethod
    def _get_action_seq(self, mode='mean'):
        """
        Get action sequence to execute on the system based
        on current control distribution
        
        Args:
            mode : {'mean', 'sample'}  
                how to choose action to be executed
                'mean' plays mean action and  
                'sample' samples from the distribution
        """        
        pass


    def sample_actions(self):
        """
        Sample actions from current control distribution
        """
        raise NotImplementedError('sample_actions funtion not implemented')
    
    @abstractmethod
    def _update_distribution(self, trajectories):
        """
        Update current control distribution using 
        rollout trajectories
        
        Args:
            trajectories : dict
                Rollout trajectories. Contains the following fields
                observations : torch.tensor
                    observations along rollouts
                actions : torch.tensor 
                    actions sampled from control distribution along rollouts
                costs : torch.tensor 
                    step costs along rollouts
        """
        pass

    @abstractmethod
    def _shift(self):
        """
        Shift the current control distribution
        to hotstart the next timestep
        """
        pass

    @abstractmethod
    def reset_distribution(self):
        pass

    def reset(self):
        """
        Reset the controller
        """
        self.num_steps = 0
        self.reset_distribution()

    @abstractmethod
    def _calc_val(self, cost_seq, act_seq):
        """
        Calculate value of state given 
        rollouts from a policy
        """
        pass

    def check_convergence(self):
        """
        Checks if controller has converged
        Returns False by default
        """
        return False
        
    # @property
    # def set_sim_state_fn(self):
    #     return self._set_sim_state_fn
    
    
    # @set_sim_state_fn.setter
    # def set_sim_state_fn(self, fn):
    #     """
    #     Set function that sets the simulation 
    #     environment to a particular state
    #     """
    #     self._set_sim_state_fn = fn

    @property
    def rollout_fn(self):
        return self._rollout_fn
    
    @rollout_fn.setter
    def rollout_fn(self, fn):
        """
        Set the rollout function from 
        input function pointer
        """
        self._rollout_fn = fn
    
    @abstractmethod
    def generate_rollouts(self, state):
        pass

    def optimize(self, state, calc_val=False, shift_steps=1, n_iters=None):
        """
        Optimize for best action at current state

        Parameters
        ----------
        state : torch.Tensor
            state to calculate optimal action from
        
        calc_val : bool
            If true, calculate the optimal value estimate
            of the state along with action
                
        Returns
        -------
        action : torch.Tensor
            next action to execute
        value: float
            optimal value estimate (default: 0.)
        info: dict
            dictionary with side-information
        """

        n_iters = n_iters if n_iters is not None else self.n_iters
        # get input device:
        inp_device = state.device
        inp_dtype = state.dtype
        state.to(**self.tensor_args)

        # shift distribution to hotstart from previous timestep
        if self.hotstart:
            self._shift(shift_steps)
        else:
            self.reset_distribution()
            
        # with profiler.profile(with_stack=True, profile_memory=True) as prof:
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                for _ in range(n_iters):
                    # sample M trajectories from mean_t-1
                    # update_distribution to get mean_t1 (greedy path)

                    # generate random simulated trajectories
                    # with profiler.record_function("generate_rollouts"):
                    trajectory = self.generate_rollouts(state)
                    
                    # update distribution parameters
                    # with profiler.record_function("_update_distribution"):
                    self._update_distribution(trajectory) 
                    
                    # self.mean_traj_greedy = self.get_mean_trajectory(state)
                    # self.mean_traj_sensi =  self.mean_traj_greedy
                    """
                    1. sample N trajectories from mean_t1
                    2. update_distribution to get mean_t2 (sensitive path)
    
                    # sample N trajectories from mean_t1
                    generate_rollouts 包括sample based on mean&cov 以及 compute cost 两部分
                    compute cost 是重点修改部分，现阶段设计较为容易，进修改 target_cost 与 collision_cost的权重实现 差异化竞争
                    要实现对 权重的 修改
                    """
                    # sensitive_trajectory = self.generate_sensitive_rollouts(state)

                    # self._update_distribution(sensitive_trajectory) 
                    # self.mean_traj_sensi = self.get_mean_trajectory(state)
                    # check if converged
                    if self.check_convergence():
                        break
        # print(prof.key_averages().table(sort_by='self_cpu_time_total'))

        self.trajectories = trajectory
        curr_action_seq = self._get_action_seq(mode=self.sample_mode)

        self.num_steps += 1

        return curr_action_seq.to(inp_device, dtype=inp_dtype)




    def series_optimize(self, state, calc_val=False, shift_steps=1, n_iters=None):
        """
        Optimize for best action at current state

        Parameters
        ----------
        state : torch.Tensor
            state to calculate optimal action from
        
        calc_val : bool
            If true, calculate the optimal value estimate
            of the state along with action
                
        Returns
        -------
        action : torch.Tensor
            next action to execute
        value: float
            optimal value estimate (default: 0.)
        info: dict
            dictionary with side-information
        """

        n_iters = n_iters if n_iters is not None else self.n_iters
        # get input device:
        inp_device = state.device
        inp_dtype = state.dtype
        state.to(**self.tensor_args)

        info = dict(rollout_time=0.0, entropy=[])
        # shift distribution to hotstart from previous timestep
        if self.hotstart:
            self._shift(shift_steps)
        else:
            self.reset_distribution()
            

        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                for _ in range(n_iters):
                    # sample M trajectories from mean_t-1
                    # update_distribution to get mean_t1 (greedy path)

                    # generate random simulated trajectories
                    trajectory = self.generate_rollouts(state)
                    # update distribution parameters
                    # with profiler.record_function("mppi_update"):
                    self._update_distribution(trajectory) 
                    self.mean_traj_greedy = self.get_mean_trajectory(state)
                    # self.mean_traj_sensi =  self.mean_traj_greedy
                    """
                    1. sample N trajectories from mean_t1
                    2. update_distribution to get mean_t2 (sensitive path)
    
                    # sample N trajectories from mean_t1
                    generate_rollouts 包括sample based on mean&cov 以及 compute cost 两部分
                    compute cost 是重点修改部分，现阶段设计较为容易，进修改 target_cost 与 collision_cost的权重实现 差异化竞争
                    要实现对 权重的 修改
                    """
                    sensitive_trajectory = self.generate_sensitive_rollouts(state)

                    self._update_distribution(sensitive_trajectory) 
                    self.mean_traj_sensi = self.get_mean_trajectory(state)

                    info['rollout_time'] += trajectory['rollout_time']
                    # check if converged
                    if self.check_convergence():
                        break
        self.trajectories = trajectory
        #calculate best action
        # curr_action = self._get_next_action(state, mode=self.sample_mode)
        curr_action_seq = self._get_action_seq(mode=self.sample_mode)
        #calculate optimal value estimate if required
        # if calc_val:
        #     trajectories = self.generate_rollouts(state)
        #     value = self._calc_val(trajectories)

        # # shift distribution to hotstart next timestep
        # if self.hotstart:
        #     self._shift()
        # else:
        #     self.reset_distribution()

        info['entropy'].append(self.entropy)

        self.num_steps += 1

        return curr_action_seq.to(inp_device, dtype=inp_dtype), info
    

    def multimodal_optimize(self, state, calc_val=False, shift_steps=1, n_iters=None):

        """
        1. generate trajectories based on multi-Policies include 
            "mean_action | greedy_mean_action | sensi_mean_action & greedy_best_action | sensi_best_action"
        2. compute  each-policy costs from the same trajectories : greedy 、sensi and judge cost
        3. SoftMax each policy-cost to get each policy-action : greedy_mean_action and sensi_mean_action | greedy_covariance and sensi_covariance
        4. from each-policy's top-N trajectories to compute greedy_mean_action and sensi_mean_action's Value Function Of Judge Policy 
        5. echo policy-JudgeValueFunction is the weight proportion of the final path: 
            mean_action = (1-step_mean)*old_mean_action + step_mean( w1 * greedy_mean_action + w2 * sensi_mean_mean)
            covariance  = (1-step_covariance)*old_covariance + step_covariance( w1 * greedy_covariance + w2 * sensi_covariance)
        
        execute mean_action[0]
        shift mean_action as hotstart for next loop

        """
        n_iters = n_iters if n_iters is not None else self.n_iters
        # get input device:
        inp_device = state.device
        inp_dtype = state.dtype
        state.to(**self.tensor_args)

        info = dict(rollout_time=0.0, entropy=[])
        # shift distribution to hotstart from previous timestep
        if self.hotstart:
            self._multimodal_shift(shift_steps)
        else:
            self.multimodal_reset_distribution()
            

        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                for _ in range(n_iters):
                    # 1. samle multiPolicy_actions and compute differPolicy cost
                    trajectories = self.generate_multimodal_rollouts(state)

                    self._multimodal_update_distribution(trajectories) 
                    # TODO : 这里应该怎么写 更加完善呢 目前只适用于 2D pass ,不对Franka生效 暂行的方案就是注释
                    # self.greedy_mean_traj, self.sensi_mean_traj, self.greedy_best_traj, self.sensi_best_traj,self.mean_traj = \
                    # self.get_multimodal_mean_trajectory(state)

                    info['rollout_time'] += trajectories['rollout_time']
                    if self.check_convergence():
                        break
        self.trajectories = trajectories
        #calculate best action
        # curr_action = self._get_next_action(state, mode=self.sample_mode)
        curr_action_seq = self._get_action_seq(mode=self.sample_mode)
        #calculate optimal value estimate if required
        value = self.value_min
        # if calc_val:
        #     trajectories = self.generate_rollouts(state)
        #     value = self._calc_val(trajectories)

        # # shift distribution to hotstart next timestep
        # if self.hotstart:
        #     self._shift()
        # else:
        #     self.reset_distribution()

        info['entropy'].append(self.entropy)

        self.num_steps += 1

        return curr_action_seq.to(inp_device, dtype=inp_dtype), value, info


    def simplify_multimodal_optimize(self, state,  shift_steps=1, n_iters=None):

        """
        1. generate trajectories based on multi-Policies include 
            "mean_action | greedy_mean_action | sensi_mean_action & greedy_best_action | sensi_best_action"
        2. compute  each-policy costs from the same trajectories : greedy 、sensi and judge cost
        3. SoftMax each policy-cost to get each policy-action : greedy_mean_action and sensi_mean_action | greedy_covariance and sensi_covariance
        4. from each-policy's top-N trajectories to compute greedy_mean_action and sensi_mean_action's Value Function Of Judge Policy 
        5. echo policy-JudgeValueFunction is the weight proportion of the final path: 
            mean_action = (1-step_mean)*old_mean_action + step_mean( w1 * greedy_mean_action + w2 * sensi_mean_mean)
            covariance  = (1-step_covariance)*old_covariance + step_covariance( w1 * greedy_covariance + w2 * sensi_covariance)
        
        execute mean_action[0]
        shift mean_action as hotstart for next loop

        """
        n_iters = n_iters if n_iters is not None else self.n_iters
        # get input device:
        inp_device = state.device
        inp_dtype = state.dtype
        state.to(**self.tensor_args)

        # shift distribution to hotstart from previous timestep
        if self.hotstart:
            self._multimodal_shift(shift_steps)
        else:
            self.multimodal_reset_distribution()
            
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                for _ in range(n_iters):
                    # 1. samle multiPolicy_actions and compute differPolicy cost
                    trajectories = self.generate_multimodal_rollouts(state)
                    self._multimodal_update_distribution(trajectories) 

                    if self.check_convergence():
                        break
        curr_action_seq = self._get_action_seq(mode=self.sample_mode)
        return curr_action_seq.to(inp_device, dtype=inp_dtype)

    def get_optimal_value(self, state):
        """
        Calculate optimal value of a state, i.e 
        value under optimal policy. 

        Parameters
        ----------
        state : torch.Tensor
            state to calculate optimal value estimate for
        Returns
        -------
        value : float
            optimal value estimate of the state
        """
        self.reset() #reset the control distribution
        _, value = self.optimize(state, calc_val=True, shift_steps=0)
        return value
    
    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return seed




