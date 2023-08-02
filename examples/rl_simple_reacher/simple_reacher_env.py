import numpy as np
import pyglet
import copy
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import time
import yaml
import argparse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from storm_kit.gym.helpers import load_struct_from_dict
from storm_kit.geom.geom_types import tensor_circle
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path
from storm_kit.mpc.rollout.simple_reacher import SimpleReacher
from storm_kit.mpc.control import MPPI
from storm_kit.mpc.utils.state_filter import JointStateFilter, RobotStateFilter
from storm_kit.mpc.utils.mpc_process_wrapper import ControlProcess
from storm_kit.mpc.task.simple_task import SimpleTask

import torch
torch.multiprocessing.set_start_method('spawn',force=True)



traj_log = None
key_stop = False

class ArmEnv(object):


    def __init__(self):
        
        self.goal_state = [0.8,0.4]
        self.on_goal = 0
        tensor_args = {'device':'cuda','dtype':torch.float32}
        self.simple_task = SimpleTask(robot_file="simple_reacher.yml", tensor_args=tensor_args)

        self.simple_task.update_params(goal_state=self.goal_state)
        exp_params = self.simple_task.exp_params
        controller = self.simple_task.controller
        self.sim_dt = exp_params['control_dt']
        self.max_action = np.ravel(exp_params['model']['max_action'])
        self.action_bound = [-self.max_action, self.max_action] #(-0.1,0.1)
        self.action_dim = 2
        self.state_dim =11
        self.current_state = {'position':np.array([0.10, 0.40]), 'velocity':np.zeros(2) + 0.0}

        current_state = {'position':self.current_state['position'],
                         'velocity':self.current_state['velocity'],
                         'acceleration': self.current_state['position']*0.0}
        
        action = np.random.rand(2)*self.max_action*2 - self.max_action #(-0.1,0.1)

        # simple_task.controller.rollout_fn(current_state, action)
        self.curr_state = np.concatenate((current_state['position'], current_state['velocity'], current_state['acceleration'])) # 1*6

        # next_state = self.simple_task.controller.rollout_fn.dynamics_model.get_next_state(self.curr_state, action,self.sim_dt)
        # cost, goal_dist = self.simple_task.get_current_error(current_state)

        # plot _initial
        fig = plt.figure()
        self.ax = plt.subplot(1,1,1)
        # fig.canvas.mpl_connect('button_press_event', self.press_call_back)
        # dist_map
        self.image = controller.rollout_fn.image_collision_cost.world_coll.im # 获取障碍图像，im:原始图像 dist_map: 碰撞图像（会被0-1化离散表征）
        self.extents = np.ravel(exp_params['model']['position_bounds'])
        self.ax.imshow(self.image, extent=self.extents)


    def step(self, action):

        done = False
        # action = np.clip(action, *self.action_bound)
        # 动力学模型
        next_state = self.simple_task.controller.rollout_fn.dynamics_model.get_next_state(self.curr_state, action,self.sim_dt)
        current_state = {'position':self.curr_state[:self.action_dim],
                         'velocity':self.curr_state[self.action_dim:2*self.action_dim],
                         'acceleration': self.curr_state[2*self.action_dim:]}
        
        """ simple dynamic model
        curr_state[2 * self.n_dofs:3 * self.n_dofs] = act * dt
        curr_state[self.n_dofs:2*self.n_dofs] = curr_state[self.n_dofs:2*self.n_dofs] + curr_state[self.n_dofs*2:self.n_dofs*3] * dt
        curr_state[:self.n_dofs] = curr_state[:self.n_dofs] + curr_state[self.n_dofs:2*self.n_dofs] * dt
        """
        # reward 的设计

        cost, goal_dist = self.simple_task.get_current_error(current_state)
        reward = - cost
        # distance = np.linalg.norm(self.goal_state - self.curr_state[:2])
        # reward = -distance
        self.curr_state = next_state
        # done 
        # reward += np.exp(-100.0*distance)
        if goal_dist <= 0.05:
            reward += 40*np.exp(-100.0*goal_dist)
            self.on_goal+=1
            if self.on_goal >=5:
                done = True
        else:
            self.on_goal = 0
        
        reward -= 0.8 # step cost

        dist = (self.goal_state - self.curr_state[:2]) 
        s = np.concatenate((self.curr_state,self.goal_state,dist, [1. if self.on_goal else 0.]))
        # print(action,reward,done,dist)
        # print(dist,goal_dist)
        return s, reward, done

    def reset(self):
        # goal and state reset
        self.goal_state = np.random.rand(2) # [0~1,0~1] 
        self.simple_task.update_params(goal_state=self.goal_state)
        self.curr_state = np.concatenate((np.random.rand(2),np.zeros(4))) 
        # self.curr_state = np.concatenate((np.array([0.10,0.30]),np.zeros(4))) 
        # state 
        self.on_goal = 0
        dist = (self.goal_state - self.curr_state[:2]) 
        s = np.concatenate((self.curr_state ,self.goal_state ,dist, [1. if self.on_goal else 0.]))
        return s 

    def render(self):
        # draw state from self.curr_state

        self.ax.cla()
        self.ax.imshow(self.image, extent=self.extents)
        self.ax.set_xlim(self.extents[0], self.extents[1])
        self.ax.set_ylim(self.extents[2], self.extents[3])
        self.ax.plot(0.05,0.2, 'rX', linewidth=3.0, markersize=15) # 起始点
        self.ax.plot(self.goal_state[0], self.goal_state[1], 'gX', linewidth=3.0, markersize=15) # 目标点
        
        self.ax.plot(np.ravel(self.curr_state[0]),
        np.ravel(self.curr_state[1]),
        'g.', linewidth=3.0, markersize=15)

        plt.pause(1e-10)
    
    # def press_call_back(self,event):
    #     self.goal_state = [event.xdata,event.ydata]
    #     print("goal_state is go to ",self.goal_state)
    #     self.simple_task.update_params(goal_state=self.goal_state)

    def sample_action(self):
        return np.random.rand(2)*self.max_action*2 - self.max_action #(-0.1,0.1)
    
    def action_line(self):
        direct_e =   (self.goal_state - self.curr_state[:2]) / (np.linalg.norm(self.goal_state - self.curr_state[:2])+1e-6)
        len_direct_e = np.linalg.norm(self.goal_state - self.curr_state[:2])
        if np.linalg.norm(self.goal_state - self.curr_state[:2]) >=  0.1:
            len_direct_e =  0.1
        return direct_e * len_direct_e
    

if __name__ == '__main__':
    env = ArmEnv()
    env.reset()
    step = 0 
    while True:
        # env.step(env.sample_action())
        _,reward,done = env.step(env.action_line())
        step += 1
        env.render()
        if done:
            env.reset()
            print("====",step)
            step = 0
            