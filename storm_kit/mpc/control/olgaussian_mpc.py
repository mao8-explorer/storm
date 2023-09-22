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
"""
MPC with open-loop Gaussian policies
"""
import copy

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from .control_base import Controller
from .control_utils import generate_noise, scale_ctrl, generate_gaussian_halton_samples, generate_gaussian_sobol_samples, gaussian_entropy, matrix_cholesky, batch_cholesky, get_stomp_cov
from .sample_libs import StompSampleLib, HaltonSampleLib, RandomSampleLib, HaltonStompSampleLib, MultipleSampleLib

class OLGaussianMPC(Controller):
    """
        .. inheritance-diagram:: OLGaussianMPC
           :parts: 1
    """    
    def __init__(self, 
                 d_action,                
                 action_lows,
                 action_highs,
                 horizon,
                 init_cov,
                 init_mean,
                 base_action,
                 num_particles,
                 gamma,
                 n_iters,
                 step_size_mean,
                 step_size_cov,
                 null_act_frac=0.,
                 rollout_fn=None,
                 sample_mode='mean',
                 hotstart=True,
                 squash_fn='clamp',
                 cov_type='sigma_I',
                 seed=0,
                 sample_params={'type': 'halton', 'fixed_samples': True, 'seed':0, 'filter_coeffs':None},
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32},
                 fixed_actions=False):
        """
        Parameters
        __________
        base_action : str
            Action to append at the end when shifting solution to next timestep
            'random' : appends random action
            'null' : appends zero action
            'repeat' : repeats second to last action
        num_particles : int
            Number of action sequences sampled at every iteration
        """

        super(OLGaussianMPC, self).__init__(d_action,
                                            action_lows,
                                            action_highs,
                                            horizon,
                                            gamma,
                                            n_iters,
                                            rollout_fn,
                                            sample_mode,
                                            hotstart,
                                            seed,
                                            tensor_args)
        
        self.init_cov = init_cov 
        self.init_mean = init_mean.clone().to(**self.tensor_args)
        self.cov_type = cov_type
        self.base_action = base_action
        self.num_particles = num_particles
        self.step_size_mean = step_size_mean
        self.step_size_cov = step_size_cov
        self.squash_fn = squash_fn

        self.null_act_frac = null_act_frac
        self.num_null_particles = round(int(null_act_frac * self.num_particles * 1.0))


        self.num_neg_particles = round(int(null_act_frac * self.num_particles)) - self.num_null_particles

        self.num_nonzero_particles = self.num_particles - self.num_null_particles - self.num_neg_particles

        #print(self.num_null_particles, self.num_neg_particles)

        self.sample_params = sample_params
        self.sample_type = sample_params['type']
        # initialize sampling library:
        if sample_params['type'] == 'stomp':
            self.sample_lib = StompSampleLib(self.horizon, self.d_action, tensor_args=self.tensor_args)
            self.sample_shape = torch.Size([self.num_nonzero_particles - 2], device=self.tensor_args['device'])
            self.i_ha = torch.eye(self.d_action, **self.tensor_args).repeat(1, self.horizon)

        elif sample_params['type'] == 'halton':
            self.sample_lib = HaltonSampleLib(self.horizon, self.d_action,
                                              tensor_args=self.tensor_args,
                                              **self.sample_params)
            self.sample_shape = torch.Size([self.num_nonzero_particles - 2], device=self.tensor_args['device'])
        elif sample_params['type'] == 'random':
            self.sample_lib = RandomSampleLib(self.horizon, self.d_action, tensor_args=self.tensor_args,
                                              **self.sample_params)
            self.sample_shape = torch.Size([self.num_nonzero_particles - 2], device=self.tensor_args['device'])
        elif sample_params['type'] == 'multiple':
            self.sample_lib = MultipleSampleLib(self.horizon, self.d_action, tensor_args=self.tensor_args, **self.sample_params)
            self.sample_shape = torch.Size([self.num_nonzero_particles - 2], device=self.tensor_args['device'])
            self.multimodal_sample_shape = torch.Size([self.num_nonzero_particles - 3], device=self.tensor_args['device'])
        self.stomp_matrix = None #self.sample_lib.stomp_cov_matrix
        # initialize covariance types:
        if self.cov_type == 'full_HAxHA':
            self.I = torch.eye(self.horizon * self.d_action, **self.tensor_args)
            
        else: # AxA
            self.I = torch.eye(self.d_action, **self.tensor_args)
        
        self.Z_seq = torch.zeros(1, self.horizon, self.d_action, **self.tensor_args)

        # self.reset_distribution()
        self.multimodal_reset_distribution()
        if self.num_null_particles > 0:
            self.null_act_seqs = torch.zeros(self.num_null_particles, self.horizon, self.d_action, **self.tensor_args)
            
        self.delta = None

    def _get_action_seq(self, mode='mean'):
        if mode == 'mean':
            act_seq = self.mean_action.clone()
        elif mode == 'sample':
            delta = self.generate_noise(shape=torch.Size((1, self.horizon)),
                                        base_seed=self.seed_val + 123 * self.num_steps)
            act_seq = self.mean_action + torch.matmul(delta, self.full_scale_tril)
        else:
            raise ValueError('Unidentified sampling mode in get_next_action')
        
        act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)

        return act_seq


    def generate_noise(self, shape, base_seed=None):
        """
            Generate correlated noisy samples using autoregressive process
        """
        delta = self.sample_lib.get_samples(sample_shape=shape, seed=base_seed)
        return delta
        
    def sample_enhance_actions(self, state=None):

        # chatgpt,下面更改代码，通过 不同采样均值（random_shooting的best_action 与 MPPI的mean_action）的方式增加MPPI的探索能力
        # 前面根据MPPI算法的mean_action随机采样出来一系列轨迹：act_seq = self.mean_action.unsqueeze(0) + scaled_delta
        # 现在需要根据 random_shooting算法的best_action(best_traj)随机采样出一系列轨迹；然后将两个算法采样的轨迹叠放在一起， 生成最后的批量轨迹，用于MPPI算法后续的代价计算以及softMAX部分；
        # 之所以这样更改代码，是希望能通过不同采样均值的方式增加MPPI的探索能力。 我的代码有一些错误和不足，有没有之前类似的算法值得借鉴，希望能根据我的需求补充完整
        # random_shooting_act_seq = self.best_traj.unsqueeze(0) + scaled_delta 
        # act_seq = torch.cat((act_seq, random_shooting_act_seq), dim=0)
        delta = self.sample_lib.get_samples(sample_shape=self.sample_shape, base_seed=self.seed_val + self.num_steps)
        #add zero-noise seq so mean is always a part of samples
        delta = torch.cat((delta, self.Z_seq), dim=0)
        
        # samples could be from HAxHA or AxA:
        # We reshape them based on covariance type:
        # if cov is AxA, then we don't reshape samples as samples are: N x H x A
        # if cov is HAxHA, then we reshape
        if self.cov_type == 'full_HAxHA':
            # delta: N * H * A -> N * HA
            delta = delta.view(delta.shape[0], self.horizon * self.d_action)
            
        scaled_delta = torch.matmul(delta, self.full_scale_tril).view(delta.shape[0],
                                                                      self.horizon,
                                                                      self.d_action)
       
        random_scaled_delta = scaled_delta[:60]
        mppi_scaled_delta = scaled_delta[60:]
        # debug_act = delta[:,:,0].cpu().numpy()

        mppi_act_seq = self.mean_action.unsqueeze(0) + mppi_scaled_delta
        random_act_seq = self.best_traj.unsqueeze(0) + random_scaled_delta

        act_seq = torch.cat((random_act_seq,mppi_act_seq), dim=0)
        # act_seq[-1,]==self.mean_action.unsqueeze(0)

        act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)
        

        append_acts = self.best_traj.unsqueeze(0)
        
        #append zero actions (for stopping)
        if self.num_null_particles > 0:
            # zero particles:

            # negative action particles:
            neg_action = -1.0 * self.mean_action.unsqueeze(0)
            neg_act_seqs = neg_action.expand(self.num_neg_particles,-1,-1)
            append_acts = torch.cat((append_acts, self.null_act_seqs, neg_act_seqs),dim=0)

        # mean action index_+1 : 296 = sample_shape + 1 = 295 + 1
        # best_traj_index_+1 : 297 = sample_shape + 2 = 295 + 2
        act_seq = torch.cat((act_seq, append_acts), dim=0)
        return act_seq


    def sample_multimodal_actions(self, state=None):

        delta = self.sample_lib.get_samples(sample_shape=self.multimodal_sample_shape, base_seed=self.seed_val + self.num_steps)
        #add zero-noise seq so mean is always a part of samples
        delta = torch.cat((delta, self.Z_seq), dim=0)
        
        if self.cov_type == 'full_HAxHA':
            # delta: N * H * A -> N * HA
            delta = delta.view(delta.shape[0], self.horizon * self.d_action)
            
        scaled_delta = torch.matmul(delta, self.full_scale_tril).view(delta.shape[0],
                                                                      self.horizon,
                                                                      self.d_action)
       
        sensi_random_scaled_delta = scaled_delta[:30]
        greedy_random_scaled_delta = scaled_delta[30:60]

        sensi_scaled_delta = scaled_delta[60:110]
        greedy_scaled_delta = scaled_delta[110:160]
        mppi_scaled_delta = scaled_delta[160:]

        # debug_act = delta[:,:,0].cpu().numpy()

        sensi_random_act_seq = self.sensi_best_action.unsqueeze(0) + sensi_random_scaled_delta
        greedy_random_act_seq = self.greedy_best_action.unsqueeze(0) + greedy_random_scaled_delta
        sensi_act_seq = self.sensi_mean.unsqueeze(0) + sensi_scaled_delta
        greedy_act_seq = self.greedy_mean.unsqueeze(0) + greedy_scaled_delta
        mppi_act_seq = self.mean_action.unsqueeze(0) + mppi_scaled_delta

        act_seq = torch.cat((sensi_random_act_seq,greedy_random_act_seq,sensi_act_seq,greedy_act_seq,mppi_act_seq), dim=0)
        # act_seq[-1,]==self.mean_action.unsqueeze(0)

        act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)
        

        #append zero actions (for stopping)
        if self.num_null_particles > 0:
            # zero particles:

            # negative action particles:
            neg_action = -1.0 * self.mean_action.unsqueeze(0)
            neg_act_seqs = neg_action.expand(self.num_neg_particles,-1,-1)
            append_acts = torch.cat((self.sensi_best_action.unsqueeze(0) , \
                                     self.greedy_best_action.unsqueeze(0) ,\
                                     self.null_act_seqs, neg_act_seqs),dim=0)

        # mean action index_+1 : 296 = sample_shape + 1 = 295 + 1
        # best_traj_index_+1 : 297 = sample_shape + 2 = 295 + 2
        act_seq = torch.cat((act_seq, append_acts), dim=0)
        return act_seq
       
 
    
    def generate_multimodal_rollouts(self, state):

        act_seq = self.sample_multimodal_actions(state=state)
        trajectories = self._rollout_fn.multimodal_rollout_fn(state, act_seq)
        return trajectories


    def sample_actions(self, state=None):
        delta = self.sample_lib.get_samples(sample_shape=self.sample_shape, base_seed=self.seed_val + self.num_steps)
        #add zero-noise seq so mean is always a part of samples
        delta = torch.cat((delta, self.Z_seq), dim=0)
        
        # samples could be from HAxHA or AxA:
        # We reshape them based on covariance type:
        # if cov is AxA, then we don't reshape samples as samples are: N x H x A
        # if cov is HAxHA, then we reshape
        if self.cov_type == 'full_HAxHA':
            # delta: N * H * A -> N * HA
            delta = delta.view(delta.shape[0], self.horizon * self.d_action)
            
        scaled_delta = torch.matmul(delta, self.full_scale_tril).view(delta.shape[0],
                                                                      self.horizon,
                                                                      self.d_action)
        debug_act = delta[:,:,0].cpu().numpy()

        act_seq = self.mean_action.unsqueeze(0) + scaled_delta

        act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)
        append_acts = self.best_traj.unsqueeze(0)
        
        #append zero actions (for stopping)
        if self.num_null_particles > 0:
            # zero particles:
            # negative action particles:
            neg_action = -1.0 * self.mean_action.unsqueeze(0)
            neg_act_seqs = neg_action.expand(self.num_neg_particles,-1,-1)
            append_acts = torch.cat((append_acts, self.null_act_seqs, neg_act_seqs),dim=0)

        act_seq = torch.cat((act_seq, append_acts), dim=0)
        return act_seq

    def get_mean_trajectory(self, state):
        """
            Samples a batch of actions, rolls out trajectories for each particle
            and returns the resulting observations, costs,  
            actions

            Parameters
            ----------
            state : dict or np.ndarray
                Initial state to set the simulation env to
         """
        # 200 * 20 *2 
        # act_seq = self.sample_actions(state=state) # sample noise from covariance of current control distribution
       
        mppi_act_seq = self.mean_action
        

        # act_seq -> trajectory: actions 200*20*2 | states 200*20*7 | costs 200*20
        single_trajectory = self._rollout_fn.single_state_forward(state, mppi_act_seq).squeeze(0)
        # trajectories['actions'][-5,] == act_seq[295,]
        # mean_trajectories = trajectories['state_seq'][-5,]
        # best_trajectories = trajectories['state_seq'][-4,]

        return single_trajectory

    def get_multimodal_mean_trajectory(self, state):
        """
            Samples a batch of actions, rolls out trajectories for each particle
            and returns the resulting observations, costs,  
            actions

            Parameters
            ----------
            state : dict or np.ndarray
                Initial state to set the simulation env to
         """
        # 200 * 20 *2 
        # act_seq = self.sample_actions(state=state) # sample noise from covariance of current control distribution
       
        # act_seq -> trajectory: actions 200*20*2 | states 200*20*7 | costs 200*20
        greedy_mean_traj = self._rollout_fn.single_state_forward(state, self.greedy_mean).squeeze(0)
        sensi_mean_traj =  self._rollout_fn.single_state_forward(state, self.sensi_mean).squeeze(0)
        greedy_best_traj = self._rollout_fn.single_state_forward(state, self.greedy_best_action).squeeze(0)
        sensi_best_traj =  self._rollout_fn.single_state_forward(state, self.sensi_best_action).squeeze(0)
        mean_traj =        self._rollout_fn.single_state_forward(state, self.mean_action).squeeze(0)
        # trajectories['actions'][-5,] == act_seq[295,]
        # mean_trajectories = trajectories['state_seq'][-5,]
        # best_trajectories = trajectories['state_seq'][-4,]

        return greedy_mean_traj, sensi_mean_traj, greedy_best_traj, sensi_best_traj,mean_traj
    
    def generate_rollouts(self, state):
        """
            Samples a batch of actions, rolls out trajectories for each particle
            and returns the resulting observations, costs,  
            actions

            Parameters
            ----------
            state : dict or np.ndarray
                Initial state to set the simulation env to
         """
        # 200 * 20 *2 
        # act_seq = self.sample_actions(state=state) # sample noise from covariance of current control distribution
        act_seq = self.sample_enhance_actions(state=state)
        # act_seq -> trajectory: actions 200*20*2 | states 200*20*7 | costs 200*20
        trajectories = self._rollout_fn(state, act_seq)
        # trajectories['actions'][-5,] == act_seq[295,]
        # mean_trajectories = trajectories['state_seq'][-5,]
        # best_trajectories = trajectories['state_seq'][-4,]

        return trajectories

    def generate_sensitive_rollouts(self, state):

        # 200 * 20 *2 
        # act_seq = self.sample_actions(state=state) # sample noise from covariance of current control distribution
        act_seq = self.sample_enhance_actions(state=state)
        # act_seq -> trajectory: actions 200*20*2 | states 200*20*7 | costs 200*20
        trajectories = self._rollout_fn.short_sighted_rollout_fn(state, act_seq, self.mean_traj_greedy)
        # trajectories['actions'][-5,] == act_seq[295,]
        # mean_trajectories = trajectories['state_seq'][-5,]
        # best_trajectories = trajectories['state_seq'][-4,]

        return trajectories
    
    # def generate_policy_rollouts(self, state):

    #     pi_actions = torch.empty(num_particles,self.horizon, self.action_dim, device=self.device)
    #     state = self.state.repeat(num_pi_trajs,1)
    #     # 他repeat出来N_pi个初始状态 N的数量不一定越多越好 min_std越大 N大一点来增加探索，min_std小的话，N大会降低探索程度？
    #     for t in range(self.horizon):
    #         pi_actions[t] = self.model.pi(state)  # action take from N_pi min_std: 0.05
    #         next_state = self.rollout_fn.dynamics_model.get_next_state(self.curr_state, pi_actions[t], self.sim_dt)
    #         # (next_state + observation!) did not realize ： 请问我的observation 要实时的计算吗
    #         # 需要动力学模型 state

    #     actions = torch.cat([actions,pi_actions],dim = 1)

    #     pass

    def _shift(self, shift_steps=1):
        """
            Predict mean for the next time step by
            shifting the current mean forward by one step
        """
        if(shift_steps == 0):
            return
        # self.new_mean_action = self.mean_action.clone()
        # self.new_mean_action[:-1] = #self.mean_action[1:]
        self.mean_action = self.mean_action.roll(-shift_steps,0)
        self.best_traj = self.best_traj.roll(-shift_steps,0)
        
        if self.base_action == 'random':
            self.mean_action[-1] = self.generate_noise(shape=torch.Size((1, 1)), 
                                                       base_seed=self.seed_val + 123*self.num_steps)
            self.best_traj[-1] = self.generate_noise(shape=torch.Size((1, 1)), 
                                                     base_seed=self.seed_val + 123*self.num_steps)
        elif self.base_action == 'null':
            self.mean_action[-shift_steps:].zero_() 
            self.best_traj[-shift_steps:].zero_()
        elif self.base_action == 'repeat':
            self.mean_action[-shift_steps:] = self.mean_action[-shift_steps -1].clone()
            self.best_traj[-shift_steps:] = self.best_traj[-shift_steps -1 ].clone()
            #self.mean_action[-1] = self.mean_action[-2].clone()
            #self.best_traj[-1] = self.best_traj[-2].clone()
        else:
            raise NotImplementedError("invalid option for base action during shift")
        # self.mean_action = self.new_mean_action

    def _multimodal_shift(self, shift_steps=1):
        """
            Predict mean for the next time step by
            shifting the current mean forward by one step
        """
        if(shift_steps == 0):
            return
        # self.new_mean_action = self.mean_action.clone()
        # self.new_mean_action[:-1] = #self.mean_action[1:]
        self.mean_action = self.mean_action.roll(-shift_steps,0)
        self.sensi_mean = self.sensi_mean.roll(-shift_steps,0)
        self.greedy_mean = self.greedy_mean.roll(-shift_steps,0)
        self.sensi_best_action = self.sensi_best_action.roll(-shift_steps,0)
        self.greedy_best_action = self.greedy_best_action.roll(-shift_steps,0)

        if self.base_action == 'random':
            self.mean_action[-1] = self.generate_noise(shape=torch.Size((1, 1)), 
                                                       base_seed=self.seed_val + 123*self.num_steps)
            self.sensi_mean[-1] = self.generate_noise(shape=torch.Size((1, 1)), 
                                                     base_seed=self.seed_val + 123*self.num_steps)
            self.greedy_mean[-1] = self.generate_noise(shape=torch.Size((1, 1)), 
                                                       base_seed=self.seed_val + 123*self.num_steps)
            self.sensi_best_action[-1] = self.generate_noise(shape=torch.Size((1, 1)), 
                                                     base_seed=self.seed_val + 123*self.num_steps)
            self.greedy_best_action[-1] = self.generate_noise(shape=torch.Size((1, 1)), 
                                                     base_seed=self.seed_val + 123*self.num_steps)
            
        elif self.base_action == 'null':
            self.mean_action[-shift_steps:].zero_() 
            self.sensi_mean[-shift_steps:].zero_()
            self.greedy_mean[-shift_steps:].zero_() 
            self.sensi_best_action[-shift_steps:].zero_()
            self.greedy_best_action[-shift_steps:].zero_()
        elif self.base_action == 'repeat':
            self.mean_action[-shift_steps:] = self.mean_action[-shift_steps -1].clone()
            self.sensi_mean[-shift_steps:] = self.sensi_mean[-shift_steps -1 ].clone()
            self.greedy_mean[-shift_steps:] = self.greedy_mean[-shift_steps -1].clone()
            self.sensi_best_action[-shift_steps:] = self.sensi_best_action[-shift_steps -1 ].clone()
            self.greedy_best_action[-shift_steps:] = self.greedy_best_action[-shift_steps -1].clone()
        else:
            raise NotImplementedError("invalid option for base action during shift")
        # self.mean_action = self.new_mean_action


    def reset_mean(self):
        self.mean_action = self.init_mean.clone()
        self.best_traj = self.mean_action.clone()

    def multimodal_reset_mean(self):
        self.mean_action = self.init_mean.clone()
        self.sensi_mean = self.mean_action.clone()
        self.greedy_mean = self.mean_action.clone()
        self.sensi_best_action = self.sensi_mean.clone()
        self.greedy_best_action = self.greedy_mean.clone()

    def reset_covariance(self):

        if self.cov_type == 'sigma_I':
            self.cov_action = torch.tensor(self.init_cov, **self.tensor_args)
            self.init_cov_action = self.init_cov
            self.inv_cov_action = 1.0 / self.init_cov  
            self.scale_tril = torch.sqrt(self.cov_action)
        
        elif self.cov_type == 'diag_AxA':
            self.init_cov_action = torch.tensor([self.init_cov]*self.d_action, **self.tensor_args)
            self.cov_action = self.init_cov_action
            self.inv_cov_action = 1.0 / self.cov_action
            self.scale_tril = torch.sqrt(self.cov_action)

        
        elif self.cov_type == 'full_AxA':
            self.init_cov_action = torch.diag(torch.tensor([self.init_cov]*self.d_action, **self.tensor_args))
            self.cov_action = self.init_cov_action
            self.scale_tril = matrix_cholesky(self.cov_action) #torch.cholesky(self.cov_action)
            self.inv_cov_action = torch.cholesky_inverse(self.scale_tril)

        elif self.cov_type == 'full_HAxHA':
            self.init_cov_action = torch.diag(torch.tensor([self.init_cov] * (self.horizon * self.d_action), **self.tensor_args))
                
            self.cov_action = self.init_cov_action
            self.scale_tril = torch.linalg.cholesky(self.cov_action)
            self.inv_cov_action = torch.cholesky_inverse(self.scale_tril)
        else:
            raise ValueError('Unidentified covariance type in update_distribution')

    def reset_distribution(self):
        """
            Reset control distribution
        """
        self.reset_mean()
        self.reset_covariance()


    def multimodal_reset_distribution(self):
        """
            Reset control distribution
        """
        self.multimodal_reset_mean()
        self.reset_covariance()


    def _calc_val(self, cost_seq, act_seq):
        raise NotImplementedError("_calc_val not implemented")


    @property
    def squashed_mean(self):
        return scale_ctrl(self.mean_action, self.action_lows, self.action_highs, squash_fn=self.squash_fn)

    @property
    def full_cov(self):
        if self.cov_type == 'sigma_I':
            return self.cov_action * self.I
        elif self.cov_type == 'diag_AxA':
            return torch.diag(self.cov_action)
        elif self.cov_type == 'full_AxA':
            return self.cov_action
        elif self.cov_type == 'full_HAxHA':
            return self.cov_action
    
    @property
    def full_inv_cov(self):
        if self.cov_type == 'sigma_I':
            return self.inv_cov_action * self.I
        elif self.cov_type == 'diag_AxA':
            return torch.diag(self.inv_cov_action)
        elif self.cov_type == 'full_AxA':
            return self.inv_cov_action
        elif self.cov_type == 'full_HAxHA':
            return self.inv_cov_action

            

    @property
    def full_scale_tril(self):
        if self.cov_type == 'sigma_I':
            return self.scale_tril * self.I
        elif self.cov_type == 'diag_AxA':
            return torch.diag(self.scale_tril)
        elif self.cov_type == 'full_AxA':
            return self.scale_tril
        elif self.cov_type == 'full_HAxHA':
            return self.scale_tril
            


    @property
    def entropy(self):
        # ent_cov = gaussian_entropy(cov=self.full_cov)
        ent_L = gaussian_entropy(L=self.full_scale_tril)
        return ent_L
