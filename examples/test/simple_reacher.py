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
import time
import cv2
matplotlib.use('tkagg')
torch.multiprocessing.set_start_method('spawn',force=True)

traj_log = None
key_stop = False
goal_list = [
            # [0.8598484848484849, 0.0606060606060608],
            [0.8787878787878789, 0.7824675324675325], 
            [0.2240259740259739, 0.7851731601731602]]
# goal_list = [
#         [0.5368799557440532, 0.40112220436764046],
#         [0.18783751745212196, 0.41692789968652044]] # for escape min_distance

# goal_list = [
#         [0.30, 0.63],
#         [0.27, 0.17]] # for escape min_distance
goal_state = goal_list[-1]

def press_call_back(event):
    global goal_state
    goal_state = [event.xdata,event.ydata]
    print(goal_state)

def key_call_back(event):
    global key_stop
    key_stop = True


def holonomic_robot(args):
    global goal_state
    global traj_log # 记录轨迹数据
    global key_stop # 标志： 键盘是否有按键按下， 图像停止路径规划
    # load
    tensor_args = {'device':'cuda','dtype':torch.float32}
    simple_task = SimpleTask(robot_file="simple_reacher.yml", tensor_args=tensor_args)

    simple_task.update_params(goal_state=goal_state)
    current_state = {'position':np.array([0.12,0.2]), 'velocity':np.zeros(2) + 0.0}
    exp_params = simple_task.exp_params
    controller = simple_task.controller
    sim_dt = exp_params['control_dt']
    
    extents = np.ravel(exp_params['model']['position_bounds'])
    traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],'coll_cost':[],
                'acc':[], 'world':None, 'bounds':extents}

    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    fig.canvas.mpl_connect('button_press_event', press_call_back)
    fig.canvas.mpl_connect('key_press_event', key_call_back)

    goal_flagi = 0
    i = 0
    plan_length = 800 # 路径规划的总steps 800
    t_step = 0.0

    #  全局SDF_Gradient绘画
    x,y = np.linspace(0,1,30), np.linspace(0,1,30)
    X, Y = np.meshgrid(x, y)
    coordinates = np.column_stack((X.flatten(), Y.flatten()))
    coordinates = torch.as_tensor(coordinates, **tensor_args)
    
    while(i < plan_length and not key_stop):   
        ax.cla()

        # last = time.time()
        controller.rollout_fn.image_move_collision_cost.world_coll.updateSDFPotientailGradient()  # 2ms
        dist_map = controller.rollout_fn.image_move_collision_cost.world_coll.dist_map # 获取障碍图像，im:原始图像 dist_map: 碰撞图像（会被0-1化离散表征）
        # controller.rollout_fn.image_collision_cost.world_coll.update_world()
        im = controller.rollout_fn.image_move_collision_cost.world_coll.im 
        image = cv2.addWeighted(im.astype(float), 0.5, dist_map.cpu().numpy().astype(float), 0.5, 1).astype(np.uint8)

        traj_log['world'] = image
        ax.imshow(traj_log['world'], extent=extents,cmap='gray')
        ax.set_xlim(traj_log['bounds'][0], traj_log['bounds'][1])
        ax.set_ylim(traj_log['bounds'][2], traj_log['bounds'][3])
        # ax.plot(0.08,0.2, 'rX', linewidth=3.0, markersize=15) # 起始点
        ax.plot(goal_state[0], goal_state[1], 'gX', linewidth=3.0, markersize=15) # 目标点

        simple_task.update_params(goal_state=goal_state) # 目标更变
        current_state = {'position':current_state['position'],
                         'velocity':current_state['velocity'],
                         'acceleration': current_state['position']*0.0}
        filtered_state = current_state

        #action =  env.step(obs)
        command, value_function = simple_task.get_command(t_step, filtered_state, sim_dt, WAIT=True)
        _, goal_dist ,current_coll= simple_task.get_current_error(filtered_state)
        mean_trajectory = simple_task.mean_trajectory.cpu().numpy()
        
        current_state = command
        costs = simple_task.get_current_error(current_state)  
        if goal_dist[0] < 0.04:
            goal_state = goal_list[goal_flagi % 2]
            goal_flagi += 1
            print("next goal",goal_flagi)

        ax.scatter(np.ravel(filtered_state['position'][0]),
                   np.ravel(filtered_state['position'][1]),
                   c=np.ravel(current_coll),s=np.array(100))
        velocity_magnitude = np.linalg.norm(filtered_state['velocity'], axis=0)  # 计算速度大小
        ax.quiver(  np.ravel(filtered_state['position'][0]),
                    np.ravel(filtered_state['position'][1]),
                    np.ravel(filtered_state['velocity'][0]),
                    np.ravel(filtered_state['velocity'][1]),
                    velocity_magnitude, cmap=plt.cm.jet)
        pose = torch.as_tensor(current_state['position'], **tensor_args).unsqueeze(0)
        grad_y_curr,grad_x_curr = controller.rollout_fn.image_move_collision_cost.world_coll.get_pt_gradxy(pose)
        potential_curr = controller.rollout_fn.image_move_collision_cost.world_coll.get_pt_value(pose)
        ax.quiver(  np.ravel(filtered_state['position'][0]),
                    np.ravel(filtered_state['position'][1]),
                    np.ravel(grad_x_curr.cpu()),
                    np.ravel(grad_y_curr.cpu()),
                    color='red')
        
        a , b = filtered_state['velocity'] , np.array([np.ravel(grad_x_curr.cpu())[0],np.ravel(grad_y_curr.cpu())[0]])
        dot_product = np.dot(a, b)  # 计算向量 a 和 b 的点积
        norm_a = np.linalg.norm(a)  # 计算向量 a 的范数（长度）
        norm_b = np.linalg.norm(b)  # 计算向量 b 的范数（长度）
        dot_product = np.dot(a, b)  # 计算向量 a 和 b 的点积
        cos_theta = dot_product / (norm_a * norm_b + 1e-7)  # 计算余弦值
        theta = np.arccos(cos_theta)  # 计算夹角（弧度）
        degree = np.degrees(theta)

        
        #  velocity | potential | 夹角
        ax.text(0.4, 1.01, f'potential: {np.ravel(potential_curr.cpu())[0]}, collcost: {np.ravel(potential_curr.cpu())[0] * norm_a * (-cos_theta)}', 
                              fontsize=12, color='red' if potential_curr[0] > 0.3 else 'black')
        ax.text(0.6,1.04, f'Velocity Magnitude: {velocity_magnitude}', fontsize=12, color='black')
        ax.text(0.6, 1.07, f'angle: {degree}', fontsize=12, 
                color='red' if degree > 90 else 'black')
        ax.text(1.04, 0.5, f'value: {value_function.cpu().numpy()}', fontsize=12)
        
        #  全局SDF_Gradient绘画
        grad_y,grad_x = controller.rollout_fn.image_move_collision_cost.world_coll.get_pt_gradxy(coordinates)
        # 绘制箭头
        ax.quiver(  X,Y,
                    np.ravel(grad_x.view(30,-1).cpu()),
                    np.ravel(grad_y.view(30,-1).cpu()),
                    cmap=plt.cm.jet)
        
        
        top_trajs = simple_task.top_trajs
        _, _ ,coll_cost= simple_task.get_current_coll(top_trajs)
        traj_log['top_traj'] = top_trajs.cpu().numpy()
        ax.scatter(np.ravel(traj_log['top_traj'][:5,:,0].flatten()),
                   np.ravel(traj_log['top_traj'][:5,:,1].flatten()),
                   c='green',s=np.array(2))
        ax.scatter(np.ravel(traj_log['top_traj'][5:,:,0].flatten()),
                   np.ravel(traj_log['top_traj'][5:,:,1].flatten()),
                   c=np.ravel(coll_cost[0].cpu().numpy()[100:]),  s=np.array(2))
        ax.plot(np.ravel(traj_log['top_traj'][0,:,0].flatten()),
                   np.ravel(traj_log['top_traj'][0,:,1].flatten()),
                   'g-',linewidth=2,markersize=3)
        
        ax.plot(np.ravel(mean_trajectory[:,0]),
                   np.ravel(mean_trajectory[:,1]),
                   'r-',linewidth=2,markersize=3)  
        # ax.plot(np.ravel(traj_log['top_traj'][0,:,0].flatten()),
        #            np.ravel(traj_log['top_traj'][0,:,1].flatten()),
        #            'g-',linewidth=1,markersize=3)

        plt.pause(1e-10)
        traj_log['position'].append(filtered_state['position'])
        traj_log['coll_cost'].append(potential_curr.cpu()[0])
        traj_log['velocity'].append(filtered_state['velocity'])
        traj_log['command'].append(command['acceleration'])
        traj_log['acc'].append(command['acceleration'])
        traj_log['des'].append(copy.deepcopy(goal_state))
        t_step += sim_dt
        i += 1
    plt.savefig('runend.png')
    plot_traj(traj_log)


def plot_traj(traj_log):

    plt.figure()
    position = np.matrix(traj_log['position'])
    vel = np.matrix(traj_log['velocity'])
    coll = np.matrix(traj_log['coll_cost'])
    print((coll==1.0).sum())
    acc = np.matrix(traj_log['acc'])
    des = np.matrix(traj_log['des'])
    axs = [plt.subplot(3,1,i+1) for i in range(3)]
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
    plt.savefig('trajectory.png')

    plt.figure()
    extents = (traj_log['bounds'][0], traj_log['bounds'][1],
               traj_log['bounds'][2], traj_log['bounds'][3])
    img_ax = plt.subplot(1,1,1)
    img_ax.imshow(traj_log['world'], extent=extents, cmap='gray', alpha=0.4)
    img_ax.plot(np.ravel(position[0,0]), np.ravel(position[0,1]), 'rX', linewidth=3.0, markersize=15)
    img_ax.plot(des[:,0], des[:,1],'gX', linewidth=3.0, markersize=15)
    img_ax.scatter(np.ravel(position[:,0]),np.ravel(position[:,1]),c=np.ravel(coll))
    img_ax.set_xlim(traj_log['bounds'][0], traj_log['bounds'][1])
    img_ax.set_ylim(traj_log['bounds'][2], traj_log['bounds'][3])
    plt.savefig('091405_PPV_wholetheta.png')
    plt.show()

if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    holonomic_robot(args)
