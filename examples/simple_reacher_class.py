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
import time
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



# goal_list = [
#         [0.5368799557440532, 0.40112220436764046],
#         [0.18783751745212196, 0.41692789968652044]] # for escape min_distance

# goal_list = [
#         [0.30, 0.63],
#         [0.27, 0.17]] # for escape min_distance

class holonomic_robot:
    def __init__(self,args):

        self.goal_list = [
                # [0.8598484848484849, 0.0606060606060608],
                [0.8787878787878789, 0.7824675324675325], 
                [0.2240259740259739, 0.7851731601731602]]
        self.goal_state = self.goal_list[-1]
        self.pause = False # 标志： 键盘是否有按键按下， 图像停止路径规划
        # load
        self.tensor_args = {'device':'cuda','dtype':torch.float32}
        self.simple_task = SimpleTask(robot_file="simple_reacher.yml", tensor_args=self.tensor_args)
        self.simple_task.update_params(goal_state=self.goal_state)
        self.controller = self.simple_task.controller           
        
        exp_params = self.simple_task.exp_params
        
        self.sim_dt = exp_params['control_dt'] #0.1
        self.extents = np.ravel(exp_params['model']['position_bounds'])

        self.traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],'coll_cost':[],
                    'acc':[], 'world':None, 'bounds':self.extents}
        self.current_state = {'position':np.array([0.12,0.2]), 'velocity':np.zeros(2) + 0.0, 'acceleration':np.zeros(2) + 0.0 }

        self.plot_init()

    def plot_init(self):
        fig = plt.figure()
        self.ax = plt.subplot(1,1,1)
        fig.canvas.mpl_connect('button_press_event',self.press_call_back)
        fig.canvas.mpl_connect('key_press_event', self.key_call_back)
        #  全局SDF_Gradient绘画
        x,y = np.linspace(0,1,30), np.linspace(0,1,30)
        self.X, self.Y = np.meshgrid(x, y)
        coordinates = np.column_stack((self.X.flatten(), self.Y.flatten()))
        self.coordinates = torch.as_tensor(coordinates, **self.tensor_args)

    def plot_setting(self):
        self.ax.cla() #清屏
        # self.controller.rollout_fn.image_collision_cost.world_coll.update_world()

        dist_map = self.controller.rollout_fn.image_move_collision_cost.world_coll.dist_map # 获取障碍图像，im:原始图像 dist_map: 碰撞图像（会被0-1化离散表征）
        im = self.controller.rollout_fn.image_move_collision_cost.world_coll.im 
        image = cv2.addWeighted(im.astype(float), 0.5, dist_map.cpu().numpy().astype(float), 0.5, 1).astype(np.uint8)

        self.traj_log['world'] = image
        self.ax.imshow(self.traj_log['world'], extent=self.extents,cmap='gray')
        self.ax.set_xlim(self.traj_log['bounds'][0], self.traj_log['bounds'][1])
        self.ax.set_ylim(self.traj_log['bounds'][2], self.traj_log['bounds'][3])
        # ax.plot(0.08,0.2, 'rX', linewidth=3.0, markersize=15) # 起始点

        # 箭头标签 ----------------------------------------------------------------
        # 当前状态速度指向 | 当前位置SDF梯度指向 | 全局地图SDF梯度可视化
        velocity_magnitude = np.linalg.norm(self.current_state['velocity'], axis=0)  # 计算速度大小
        self.ax.quiver( np.ravel(self.current_state['position'][0]), np.ravel(self.current_state['position'][1]),
                        np.ravel(self.current_state['velocity'][0]),  np.ravel(self.current_state['velocity'][1]), 
                        velocity_magnitude, cmap=plt.cm.jet) # 当前状态 速度大小及方向
        curr_pose = torch.as_tensor(self.current_state['position'], **self.tensor_args).unsqueeze(0)
        grad_y_curr,grad_x_curr = self.controller.rollout_fn.image_move_collision_cost.world_coll.get_pt_gradxy(curr_pose) # 当前SDF梯度
        self.potential_curr = self.controller.rollout_fn.image_move_collision_cost.world_coll.get_pt_value(curr_pose) # 当前势场
        self.ax.quiver(np.ravel(self.current_state['position'][0]), np.ravel(self.current_state['position'][1]),
                    np.ravel(grad_x_curr.cpu()),np.ravel(grad_y_curr.cpu()),color='red') # 当前位置所在SDF梯度
        
        #  全局SDF_Gradient绘画 翻转x,y是坐标变化机理
        grad_y,grad_x = self.controller.rollout_fn.image_move_collision_cost.world_coll.get_pt_gradxy(self.coordinates)
        #  绘制箭头
        self.ax.quiver(self.X,self.Y, np.ravel(grad_x.view(30,-1).cpu()), np.ravel(grad_y.view(30,-1).cpu()),cmap=plt.cm.jet)
    
        # 散点标签 ----------------------------------------------------------------
        # 当前位置状态 
        self.ax.plot(self.goal_state[0], self.goal_state[1], 'gX', linewidth=3.0, markersize=15) # 目标点
        self.ax.scatter(np.ravel(self.current_state['position'][0]),np.ravel(self.current_state['position'][1]),
                        c=np.ravel(self.potential_curr.cpu()),s=np.array(200),cmap=plt.cm.jet, vmin=0, vmax=1)
        # 规划轨迹 batch_trajectories visual
        mean_trajectory = self.simple_task.mean_trajectory.cpu().numpy()
        top_trajs = self.simple_task.top_trajs
        _, _ ,coll_cost= self.simple_task.get_current_coll(top_trajs) 
        self.traj_log['top_traj'] = top_trajs.cpu().numpy()
        # 15条轨迹，前5条最优轨迹，后10条最差轨迹
        self.ax.scatter(np.ravel(self.traj_log['top_traj'][:5,:,0].flatten()), np.ravel(self.traj_log['top_traj'][:5,:,1].flatten()),
                c='green',s=np.array(2))
        self.ax.scatter(np.ravel(self.traj_log['top_traj'][5:,:,0].flatten()), np.ravel(self.traj_log['top_traj'][5:,:,1].flatten()),
                c=np.ravel(coll_cost[0].cpu().numpy()[100:]),  s=np.array(2))
        # random_shooting: best_trajectory 绿线
        self.ax.plot(np.ravel(self.traj_log['top_traj'][0,:,0].flatten()), np.ravel(self.traj_log['top_traj'][0,:,1].flatten()),
                'g-',linewidth=2,markersize=3)          
        # MPPI : mean_trajectory 红线
        self.ax.plot(np.ravel(mean_trajectory[:,0]),np.ravel(mean_trajectory[:,1]),
                'r-',linewidth=2,markersize=3)  

        #  文字标签 ----------------------------------------------------------------
        #  velocity | potential | 夹角  | MPQ Value_Function估计 
        grad_orient = np.array([np.ravel(grad_x_curr.cpu())[0],np.ravel(grad_y_curr.cpu())[0]])
        grad_magnitude = np.linalg.norm(grad_orient)  # 计算向量 b 的范数（长度）
        cos_theta = np.dot(self.current_state['velocity'], grad_orient) / (velocity_magnitude * grad_magnitude + 1e-7)  # 计算余弦值
        theta = np.arccos(cos_theta)  # 计算夹角（弧度）
        
        self.ax.text(0.6, 1.01, f'potential: {np.ravel(self.potential_curr.cpu())[0]}, collcost: {np.ravel(self.potential_curr.cpu())[0] * velocity_magnitude * (-cos_theta)}', 
                            fontsize=12, color='red' if self.potential_curr[0] > 0 else 'black')
        self.ax.text(0.6, 1.04, f'Velocity Magnitude: {velocity_magnitude}', fontsize=12, color='black')
        self.ax.text(0.6, 1.07, f'angle: {np.degrees(theta)}', fontsize=12, color='red' if theta > np.pi/2.0 else 'black')
         # MPQ value值估计 log_sum_exp
        self.ax.text(1.04, 0.5, f'value: {self.value_function.cpu().numpy()}', fontsize=12)
        
        plt.pause(1e-10)
        self.traj_append()

    def traj_append(self):

        self.traj_log['position'].append(self.current_state['position'])
        self.traj_log['velocity'].append(self.current_state['velocity'])
        self.traj_log['command'].append(self.current_state['acceleration'])
        self.traj_log['acc'].append(self.current_state['acceleration'])
        self.traj_log['coll_cost'].append(self.potential_curr.cpu()[0])
        self.traj_log['des'].append(copy.deepcopy(self.goal_state))

    def press_call_back(self,event):
        self.goal_state = [event.xdata,event.ydata]
        self.simple_task.update_params(goal_state=self.goal_state) # 目标更变
        print(self.goal_state)

    def key_call_back(self,event):
        self.pause = not self.pause

    def run(self):
        # temp parameters
        goal_flagi = 0 # 调控目标点
        i = 0   #调控运行steps
        t_step = 0.0 # 记录run_time
        goal_thresh = 0.04 # 目标点阈值
        while(i < 800):
            #  core_process
            self.controller.rollout_fn.image_move_collision_cost.world_coll.updateSDFPotientailGradient() #更新环境SDF
            command, value_function = self.simple_task.get_command(t_step, self.current_state, self.sim_dt, WAIT=True)
            self.current_state = command # or command * scale
            self.value_function = value_function

            # 这里的current_coll 反馈的不是是否发生碰撞，是forward计算的值，暂无意义
            _, goal_dist,_ = self.simple_task.get_current_error(self.current_state) 
            # goal_reacher update
            if goal_dist[0] < goal_thresh:
                self.goal_state = self.goal_list[goal_flagi % len(self.goal_list)]
                self.simple_task.update_params(goal_state=self.goal_state) # 目标更变
                goal_flagi += 1
                print("next goal",goal_flagi)

            self.plot_setting()
            if self.pause:
                while True:
                    time.sleep(1.0)
                    self.plot_setting()
                    if not self.pause: break
            t_step += self.sim_dt
            i += 1
        plt.savefig('runend.png')
        self.plot_traj()


    def plot_traj(self):
        plt.figure()
        position = np.matrix(self.traj_log['position'])
        vel = np.matrix(self.traj_log['velocity'])
        coll = np.matrix(self.traj_log['coll_cost'])
        print((coll==1.0).sum())
        acc = np.matrix(self.traj_log['acc'])
        des = np.matrix(self.traj_log['des'])
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
        extents = (self.traj_log['bounds'][0], self.traj_log['bounds'][1],
                self.traj_log['bounds'][2], self.traj_log['bounds'][3])
        img_ax = plt.subplot(1,1,1)
        img_ax.imshow(self.traj_log['world'], extent=extents, cmap='gray', alpha=0.4)
        img_ax.plot(np.ravel(position[0,0]), np.ravel(position[0,1]), 'rX', linewidth=3.0, markersize=15)
        img_ax.plot(des[:,0], des[:,1],'gX', linewidth=3.0, markersize=15)
        img_ax.scatter(np.ravel(position[:,0]),np.ravel(position[:,1]),c=np.ravel(coll))
        img_ax.set_xlim(self.traj_log['bounds'][0], self.traj_log['bounds'][1])
        img_ax.set_ylim(self.traj_log['bounds'][2], self.traj_log['bounds'][3])
        plt.savefig('091405_PPV_wholetheta.png')
        plt.show()

if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    controller = holonomic_robot(args)
    controller.run()
