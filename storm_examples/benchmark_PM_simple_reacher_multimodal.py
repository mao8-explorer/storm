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

from shutil import move
import matplotlib
matplotlib.use('tkagg')
import time
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
import csv


torch.multiprocessing.set_start_method('spawn',force=True)
from visual.plot_multimodal_simple import Plotter_MultiModal


# goal_list = [
#         [0.5368799557440532, 0.40112220436764046],
#         [0.18783751745212196, 0.41692789968652044]] # for escape min_distance

# goal_list = [
#         [0.30, 0.63],
#         [0.27, 0.17]] # for escape min_distance

# self.goal_list = [
#         # [0.9098484848484849, 0.2006060606060608],
#         [0.8787878787878789, 0.7824675324675325], 
#         [0.2240259740259739, 0.7851731601731602]]
class holonomic_robot(Plotter_MultiModal):
    def __init__(self):
        super().__init__()

        # Task parameter
        self.shift = 3
        self.up_down = True
        self.goal_list = [
        # [0.9098484848484849, 0.2006060606060608],
        [0.8787878787878789, 0.7824675324675325], 
        [0.2240259740259739, 0.7851731601731602]] 
        self.goal_state = self.goal_list[-1]
        self.pause = False # 标志： 键盘是否有按键按下， 图像停止路径规划
        # load
        self.tensor_args = {'device':'cuda','dtype':torch.float32}
        self.simple_task = SimpleTask(robot_file="simple_reacher_multimodal.yml", tensor_args=self.tensor_args)
        self.simple_task.update_params(goal_state=self.goal_state)
        self.controller = self.simple_task.controller           

        self.controller.rollout_fn.image_move_collision_cost.world_coll.Reinit(self.shift, self.up_down) # command handle

        exp_params = self.simple_task.exp_params
        
        self.sim_dt = exp_params['control_dt'] #0.1
        self.extents = np.ravel(exp_params['model']['position_bounds'])

        self.traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],'coll_cost':[],
                    'acc':[], 'world':None, 'bounds':self.extents , 'weights':[]}
        self.current_state = {'position':np.array([0.12,0.2]), 'velocity':np.zeros(2) + 0.0, 'acceleration':np.zeros(2) + 0.0 }

    def ReInit(self, current_state, goal_list):
        self.traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],'coll_cost':[],
                    'acc':[], 'world':None, 'bounds':self.extents , 'weights':[]}
        self.goal_list = goal_list
        self.goal_state = self.goal_list[-1]
        self.simple_task.update_params(goal_state=self.goal_state)
        self.current_state = {'position':current_state, 'velocity':np.zeros(2) + 0.0, 'acceleration':np.zeros(2) + 0.0 }
        # image 调整到初始位置 agent归位
        self.controller.rollout_fn.image_move_collision_cost.world_coll.Reinit(self.shift, self.up_down) # command handle

    def run(self):
        # temp parameters
        self.goal_flagi = -1 # 调控目标点
        self.loop_step = 0   #调控运行steps
        t_step = 0.0 # 记录run_time
        goal_thresh = 0.03 # 目标点阈值
        self.run_time = 0.0
        lap_count = 8 # 跑5轮次
        self.collisions_all = 0
        self.crash_rate = 0.
        collision_hanppend = False
        while(self.goal_flagi / len(self.goal_list) != lap_count):
            #  core_process
            self.controller.rollout_fn.image_move_collision_cost.world_coll.updateSDFPotientailGradient() #更新环境SDF
            last = time.time()
            command, value_function = self.simple_task.get_multimodal_command(t_step, self.current_state, self.sim_dt)
            self.run_time += time.time() - last
            self.current_state = command # or command * scale
            self.value_function = value_function
            # 这里的current_coll 反馈的不是是否发生碰撞，是forward计算的值，暂无意义
            _, goal_dist,_ = self.simple_task.get_current_error(self.current_state) 
            # goal_reacher update
            if goal_dist[0] < goal_thresh:
                if collision_hanppend : self.crash_rate += 1
                collision_hanppend = False
                self.goal_state = self.goal_list[(self.goal_flagi+1) % len(self.goal_list)]
                self.simple_task.update_params(goal_state=self.goal_state) # 目标更变
                self.goal_flagi += 1
        
            t_step += self.sim_dt
            self.loop_step += 1
            # self.plot_setting()
            curr_pose = torch.as_tensor(self.current_state['position'], **self.tensor_args).unsqueeze(0)
            self.potential_curr = self.controller.rollout_fn.image_move_collision_cost.world_coll.get_pt_value(curr_pose) # 当前势场
            if self.potential_curr[0] > 0.99 : 
                self.collisions_all += 1
                collision_hanppend = True
            self.traj_append()
        
        position = np.matrix(self.traj_log['position'])
        # 计算相邻点之间的差分
        diff = np.diff(position, axis=0)
        # 计算每个点与前一个点之间的距离
        distances = np.linalg.norm(diff, axis=1)
        # 累加得到轨迹的长度
        trajectory_length = np.sum(distances) # path-length
        velocity = np.matrix(self.traj_log['velocity'])
        speed_magnitude = np.linalg.norm(velocity, axis=1)
        # 计算速度平均值
        average_speed = np.mean(speed_magnitude)
        # 计算最大速度值
        max_speed = np.max(speed_magnitude)
        return trajectory_length, average_speed, max_speed


"""
加入avg.speed and  path_length做关键对比
期望：  
    同一个avg.speed下, PM在path-length / collision / crash_rate 三个维度 优于 SM
首先要做的 就是验证一个猜想： 同一个dt条件下, PM的速度小于SM。 以此，我们在比较的时候,可以将PM的dt适当调高
假设这个猜想成立的话,我们比较SM关键参数522,即使collision / crash-rate相同的情况下, 在path-length做出优势比较
"""

            
fieldnames = ['lap_time', 'whileloop_count', 'collision_count', 'crash_rate','Avg.Speed', 'Max.Speed','path_length', 'Note'] 

def run_experiment():

    lap_times = []
    whileloop_counts = []
    collision_counts = []
    crash_rates = []
    avg_speeds = []
    max_speeds = []
    path_lengths = []

    CarController = holonomic_robot()
    goals_list = [
        [[0.8687878787878789, 0.7824675324675325], 
         [0.2340259740259739, 0.7851731601731602]],
        [[0.8887878787878789, 0.7824675324675325], 
         [0.2140259740259739, 0.7851731601731602]],
        [[0.8687878787878789, 0.7824675324675325], 
         [0.2040259740259739, 0.7851731601731602]],
        [[0.8787878787878789, 0.7824675324675325], 
        [0.2240259740259739, 0.7851731601731602]] 
         ]    

    current_state_dict = np.array([[0.12,0.2],[0.14, 0.15],[0.12, 0.4],[0.10, 0.30]])

    with open('PM_VS_SM.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not f.tell():
            writer.writeheader()

        for j in range(goals_list.__len__()):
            for i in range(current_state_dict.shape[0]):
                first_time = time.time()
                CarController.ReInit(current_state = current_state_dict[i], goal_list = goals_list[j])
                trajectory_length, average_speed, max_speed = CarController.run() 
                lap_time = time.time() - first_time  

                lap_times.append(lap_time)
                whileloop_counts.append(CarController.loop_step)
                collision_counts.append(CarController.collisions_all)
                crash_rates.append(CarController.crash_rate)
                avg_speeds.append(average_speed)
                max_speeds.append(max_speed)
                path_lengths.append(trajectory_length)

                row = {'lap_time': lap_time, 'whileloop_count': CarController.loop_step, 'collision_count': CarController.collisions_all,
                        'crash_rate': CarController.crash_rate, 'Avg.Speed': average_speed, 'Max.Speed': max_speed, 
                        'path_length': trajectory_length}

                writer.writerow(row)
                f.flush()  # 刷新缓冲
                print(row)
        row = {'lap_time': sum(lap_times)/len(lap_times), 
            'whileloop_count': sum(whileloop_counts)/len(whileloop_counts), 
            'collision_count': sum(collision_counts)/len(collision_counts),
            'crash_rate': sum(crash_rates)/len(crash_rates), 
            'Avg.Speed': sum(avg_speeds)/len(avg_speeds),
            'Max.Speed': sum(max_speeds)/len(max_speeds), 
            'path_length': sum(path_lengths)/len(path_lengths),
            'Note': "one-footnote FOR Parallel MPPI"}
        writer.writerow(row)
        f.flush()  # 刷新缓冲
        print(row)

if __name__ == '__main__':  
  
  print("Experiment start...")
  run_experiment()
  print("Experiment finished!")