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
import os, sys
import traceback
import time
import copy

#import numpy as np
import torch


from ..utils.torch_utils import find_first_idx, find_last_idx
#import multiprocessing import Queue
from torch.multiprocessing import Pool, Process, set_start_method, Queue

import numpy as np

class ControlProcess(object):
    def __init__(self, controller, control_space='acc', control_dt=0.01):
        try:
            controller.rollout_fn.dynamics_model.robot_model.delete_lxml_objects()
        except Exception:
            pass
            
        # torch.jit.save(controller, 'control_instance.pt')
        
        try:
            controller.rollout_fn.dynamics_model.robot_model.load_lxml_objects()
        except Exception:
            pass
        self.command = None
        self.current_state = None
        self.done = False
        self.opt_data = 0.0
        self.n_dofs = controller.rollout_fn.dynamics_model.n_dofs
        
        self.traj_tstep = copy.deepcopy(controller.rollout_fn.dynamics_model._traj_tstep.detach().cpu())
        self.command_tstep = self.traj_tstep
        self.mpc_dt = 0.0 #None
        self.params = None
        self.top_trajs = None
        self.top_values = None
        self.top_idx = None
        self.control_space = control_space
        
        # TODO:  多进程设计 需要研究
        # self.result_queue = Queue(maxsize=1)
        # self.opt_queue = Queue(maxsize=1)        
        # self.opt_process = Process(target=optimize_process, args=('control_instance.p', self.opt_queue,self.result_queue))
        # self.opt_process.daemon = True
        # self.opt_process.start()
        self.controller = controller
        self.control_dt = control_dt
        self.prev_mpc_tstep = 0.0

        self.mpc_time_sum = 0
        self.end_time = 0

    def predict_next_state(self, t_step, curr_state):
        # predict next state
        # given current t_step, integrate to t_step+mpc_dt
        t1_idx = find_first_idx(self.command_tstep, t_step) - 1
        t2_idx = find_first_idx(self.command_tstep, t_step + self.mpc_dt) #- 1

        # integrate from t1->t2
        for i in range(t1_idx, t2_idx):
            command = self.command[0][i]
            curr_state = self.controller.rollout_fn.dynamics_model.get_next_state(curr_state, command, self.mpc_dt)
    
        return curr_state
    
    def simple_predict_next_state(self, curr_state):

        curr_state = self.controller.rollout_fn.dynamics_model.get_next_state(curr_state, self.command[0], self.mpc_dt)
    
        return curr_state
    
    def get_command_debug(self, state_tensor):
        if(self.command is not None):
            # 假设在mpc规划的dt时间段内，机械臂继续按照command执行，那么curr_state就要根据command做dt的推演 - 有这种想法比较合理 准！ 所有的参数 这里假定都是torch的
            state_tensor = self.controller.rollout_fn.dynamics_model.get_next_state(state_tensor, self.command[0], self.control_dt)
        command = self.controller.optimize(state_tensor.unsqueeze(0), shift_steps=1)
        self.command = command # mean_action_sequence
        act = self.command[0].cpu().detach().numpy()
        return act



    def get_real_multimodal_command_debug(self, state_tensor):
        if(self.command is not None):
            state_tensor = self.controller.rollout_fn.dynamics_model.get_next_state(state_tensor, self.command[0], self.control_dt)
        command = self.controller.simplify_multimodal_optimize(state_tensor.unsqueeze(0), shift_steps=1)
        self.command = command # mean_action_sequence
        act = self.command[0].cpu().detach().numpy()
        return act

    
   
    def get_multimodal_command_debug(self, t_step, curr_state, debug=False, control_dt=0.01):
        """ This function runs the controller in the same process and waits for optimization to  complete before return of a new command
        Args:
        t_step: current timestep
        curr_state: current state to give to mpc
        debug: flag to enable debug commands [not implemented]
        control_dt: dt to integrate command to acceleration space from a higher order space(jerk, snap). 
        """
        if(self.command is not None):
            curr_state = self.predict_next_state(t_step, curr_state)

        current_state = np.append(curr_state, t_step + self.mpc_dt)
        shift_steps = find_first_idx(self.command_tstep, t_step + self.mpc_dt)
        
        state_tensor = torch.as_tensor(current_state,**self.controller.tensor_args).unsqueeze(0)


        mpc_time = time.time()
        command = list(self.controller.multimodal_optimize(state_tensor, shift_steps=shift_steps))
        mpc_time = time.time() - mpc_time
        command[0] = command[0].cpu().numpy()
        self.command_tstep = self.traj_tstep + t_step
        
        self.opt_dt = mpc_time
        self.mpc_dt = t_step - self.prev_mpc_tstep
        self.prev_mpc_tstep = copy.deepcopy(t_step)

        # get mean_trajectory and best_trajectory
        # mean_trajectories = self.controller.trajectories['state_seq'][-1,]
        # best_trajectories = self.controller.trajectories['state_seq'][-0,]
        
        # get command data:
        self.command = command

        command_buffer, command_tstep_buffer = self.truncate_command(self.command[0], t_step, self.command_tstep)
        
        act = self.controller.rollout_fn.dynamics_model.integrate_action_step(command_buffer[0], self.control_dt)
        # next_command, val, info, best_action
        return act, command_tstep_buffer, self.command[1], command_buffer
    

    def get_command(self, t_step, curr_state, debug=False, control_dt=0.01):
        if(self.opt_queue.empty()):# and self.command is None):
            # integrate current state to mpc_dt:
            #
            if(self.command is not None):
                curr_state = self.predict_next_state(t_step, curr_state)
            
            curr_state = np.append(curr_state, t_step + self.mpc_dt)
            
            # planned command:
            
            shift_steps = find_first_idx(self.command_tstep, t_step + self.mpc_dt) #- 1
            if(shift_steps < 0):
                shift_steps = 0
            
            opt_data = {'state': curr_state, 't_step':t_step + self.mpc_dt, 'done':self.done, 'params':self.params, 'shift_steps':shift_steps, 'pred_mpc_dt':self.mpc_dt}
            
            self.start_time = time.time()
            self.mpc_dt = t_step - self.prev_mpc_tstep
            
            self.opt_queue.put(opt_data)
            self.params = None

        # wait for first command
        while(self.command is None and self.result_queue.empty()):
            time.sleep(0.01)

        
        if(not self.result_queue.empty()):# and self.command is None):
            command_data = self.result_queue.get()
            self.command_tstep = self.traj_tstep + command_data['t_step']

            self.command = command_data['command']
            self.opt_dt = command_data['mpc_dt']
            self.prev_mpc_tstep = copy.deepcopy(t_step)

            self.top_values = command_data['top_values']
            self.top_trajs = command_data['top_trajs']
            self.top_idx = command_data['top_idx']
            
            
            
        # send to process

        if(self.command is None):
            raise ValueError

        command_buffer, command_tstep_buffer = self.truncate_command(self.command[0], t_step, self.command_tstep)
        
        #print(command_buffer.shape)
        act = self.controller.rollout_fn.dynamics_model.integrate_action_step(command_buffer[0], self.control_dt)
        return act, command_tstep_buffer, self.command[1], command_buffer
    
    def truncate_command(self, command, trunc_tstep, command_tstep):
        #print(trunc_tstep, command_tstep[:4])
        f_idx = find_first_idx(command_tstep, trunc_tstep) #- 1
        if(f_idx == -1):
            f_idx = 0
        return command[f_idx:], command_tstep[f_idx:]

    def update_params(self, **kwargs):
        self.params = kwargs

 
    # def close(self):
    #     print("close是什么意思")
    #     self.done = True
    #     opt_data = {'state': None, 'dt':None, 'done':self.done, 'params':None}
    #     self.opt_queue.put(opt_data)
    #     self.opt_process.join()

def optimize_process(control_string, opt_queue, result_queue):
    """
    This runs mpc in a seperate process.

    Input:
    current_state: current state
    shift_steps: how much to shift from the previous optimization

    Returns:
    command: command trajectory


    """
    controller = torch.load(control_string)
    try:
        print("这个有啥用 ---")
        controller.rollout_fn.dynamics_model.robot_model.load_lxml_objects()
    except Exception:
        print("一直pass吗 ---")
        pass
    i = 0
    start_time = time.time()
    state_tensor = torch.zeros((1, 2 * controller.rollout_fn.dynamics_model.n_dofs), **controller.tensor_args)
    goal_count = 0
    while (True):
        print("这个有啥用 ---")
        opt_data = opt_queue.get()
        if(opt_data['done']):
            break
        current_state = opt_data['state']
        state_tensor = torch.as_tensor(current_state,**controller.tensor_args).unsqueeze(0)
        shift_steps = opt_data['shift_steps']

        
        # update goal pose if it's not none
        if(opt_data['params'] is not None):
            #print('updating goal...')
            controller.rollout_fn.update_params(**opt_data['params'])
            goal_count += 1
            if(goal_count == 100):
                #controller.reset_mean()
                controller.reset_covariance()

        mpc_time = time.time()
        command = list(controller.optimize(state_tensor, shift_steps=shift_steps))
        mpc_time = time.time() - mpc_time

        # get command data:
        top_idx = controller.top_idx
        top_values = controller.top_values
        top_trajs = controller.top_trajs
        
        command[0] = command[0].cpu().numpy()
        
        result = {'command':command, 't_step': opt_data['t_step'], 'mpc_dt': mpc_time,
                  'top_values':top_values, 'top_trajs':top_trajs, 'top_idx':top_idx}
        result_queue.put(result)
        i = time.time() - start_time
        
    return True
