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



from tqdm import tqdm
from rl_utils import ReplayBuffer
from rl_alg import DQN

import torch
torch.multiprocessing.set_start_method('spawn',force=True)



    

traj_log = None
key_stop = False

goal_state = [0.8,0.4]





def press_call_back(event):
    global goal_state
    goal_state = [event.xdata,event.ydata]

def key_call_back(event):
    global key_stop
    key_stop = True

class holonomic_robot():
    # load
    tensor_args = {'device':'cuda','dtype':torch.float32}
    simple_task = SimpleTask(robot_file="simple_reacher.yml", tensor_args=tensor_args)

    global goal_state
    simple_task.update_params(goal_state=goal_state)

    curr_state_tensor = torch.zeros((1,4), **tensor_args)
    filter_coeff = {'position':1.0, 'velocity':1.0, 'acceleration':1.0}
    current_state = {'position':np.array([0.05, 0.2]), 'velocity':np.zeros(2) + 0.0}
    

    exp_params = simple_task.exp_params
    controller = simple_task.controller
    sim_dt = exp_params['control_dt']
    
    
    global traj_log # 记录轨迹数据
    global key_stop # 标志： 键盘是否有按键按下， 图像停止路径规划

    i = 0
    plan_length = 1000 # 路径规划的总steps 

    # dist_map
    image = controller.rollout_fn.image_collision_cost.world_coll.im # 获取障碍图像，im:原始图像 dist_map: 碰撞图像（会被0-1化离散表征）
    extents = np.ravel(exp_params['model']['position_bounds'])

    traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],
                'acc':[], 'world':image, 'bounds':extents}

    zero_acc = np.zeros(2)
    t_step = 0.0
    full_act = None
    # state dim
    curr_state = np.hstack((current_state['position'], current_state['velocity'], zero_acc, t_step)) 
    curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)

    update_goal = False

    filtered_state = copy.deepcopy(current_state)
    

    traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],
                'acc':[], 'world':image, 'bounds':extents}
    

    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    fig.canvas.mpl_connect('button_press_event', press_call_back)
    fig.canvas.mpl_connect('key_press_event', key_call_back)


    ax.imshow(traj_log['world'], extent=extents)

    while(i < plan_length and not key_stop):
        
        ax.cla()
        ax.imshow(traj_log['world'], extent=extents)
        ax.set_xlim(traj_log['bounds'][0], traj_log['bounds'][1])
        ax.set_ylim(traj_log['bounds'][2], traj_log['bounds'][3])
        ax.plot(0.05,0.2, 'rX', linewidth=3.0, markersize=15) # 起始点
        ax.plot(goal_state[0], goal_state[1], 'gX', linewidth=3.0, markersize=15) # 目标点
        

        simple_task.update_params(goal_state=goal_state) # 目标更变
        current_state = {'position':current_state['position'],
                         'velocity':current_state['velocity'],
                         'acceleration': current_state['position']*0.0}
        filtered_state = current_state
        curr_state = np.hstack((filtered_state['position'], filtered_state['velocity'], filtered_state['acceleration'], t_step))
            

        curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
        error, _ = simple_task.get_current_error(filtered_state)
        
        #action =  env.step(obs)
        command = simple_task.get_command(t_step, filtered_state, sim_dt, WAIT=True)



        print(i, command['position'])

        # 作图
        # img_ax.plot(np.ravel(position[:,0]), np.ravel(position[:,1]), 'k-.', linewidth=3.0)
        ax.plot(np.ravel(filtered_state['position'][0]),
                np.ravel(filtered_state['position'][1]),
                'g.', linewidth=3.0, markersize=15)
        
                # if(i == 0):
        top_trajs = simple_task.top_trajs
        traj_log['top_traj'] = top_trajs.cpu().numpy()
        for k in range(traj_log['top_traj'].shape[0]):
            d = traj_log['top_traj'][k,:,:2]
            if k == 0 :
                ax.plot(d[:,0],d[:,1], 'g-',linewidth=1,markersize=3)
            else :
                ax.plot(d[:,0],d[:,1], 'b.',linewidth=0.1,markersize=1)


        plt.pause(1e-10)


        traj_log['position'].append(filtered_state['position'])
        traj_log['error'].append(error)
        traj_log['velocity'].append(filtered_state['velocity'])
        traj_log['command'].append(command['acceleration'])
        traj_log['acc'].append(command['acceleration'])
        traj_log['des'].append(copy.deepcopy(goal_state))
        t_step += sim_dt
        i += 1
        
    # matplotlib.use('tkagg')
    plot_traj(traj_log)


def plot_traj(traj_log):

    plt.figure()

    position = np.matrix(traj_log['position'])
    vel = np.matrix(traj_log['velocity'])
    err = np.matrix(traj_log['error'])
    acc = np.matrix(traj_log['acc'])
    act = np.matrix(traj_log['command'])
    des = np.matrix(traj_log['des'])

    c_map = [x / position.shape[0] for x in range(position.shape[0])]
    #print(c_map)
    #fig, axs = plt.subplots(5)

    axs = [plt.subplot(3,1,i+1) for i in range(3)]
    #axs = [plt.subplot(1,1,i+1) for i in range(1)]

    
    if(len(axs) >= 3):
        axs[0].set_title('Position')
        axs[1].set_title('Velocity')
        axs[2].set_title('Acceleration')

        # axs[3].set_title('Trajectory Position')
        axs[0].plot(position[:,0], 'r', label='x')
        axs[0].plot(position[:,1], 'g',label='y')

        axs[0].plot(des[:,0], 'r-.', label='x_des')
        axs[0].plot(des[:,1],'g-.', label='y_des')
        axs[0].legend()

        axs[1].plot(vel[:,0], 'r',label='x')
        axs[1].plot(vel[:,1], 'g', label='y')
        axs[2].plot(acc[:,0], 'r', label='acc')
        axs[2].plot(acc[:,1], 'g', label='acc')

    plt.figure()

    extents = (traj_log['bounds'][0], traj_log['bounds'][1],
               traj_log['bounds'][2], traj_log['bounds'][3])
    
    img_ax = plt.subplot(1,1,1)

    img_ax.imshow(traj_log['world'], extent=extents, cmap='gray', alpha=0.4)
    img_ax.plot(np.ravel(position[0,0]), np.ravel(position[0,1]), 'rX', linewidth=3.0, markersize=15)
    img_ax.plot(des[:,0], des[:,1],'gX', linewidth=3.0, markersize=15)
    img_ax.plot(np.ravel(position[:,0]), np.ravel(position[:,1]), 'k-.', linewidth=3.0)
    
    


    for k in range(traj_log['top_traj'].shape[0]):
        d = traj_log['top_traj'][k,:,:2]

    # img_ax.axis('square')
    img_ax.set_xlim(traj_log['bounds'][0], traj_log['bounds'][1])
    img_ax.set_ylim(traj_log['bounds'][2], traj_log['bounds'][3])
    plt.show()


if __name__ == '__main__':
    

    
    
    env = holonomic_robot()


    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    random.seed(0)

    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
