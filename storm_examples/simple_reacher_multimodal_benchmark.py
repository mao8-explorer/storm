
# 为了观测温度参数 β对 整个结果的影响，需要记录数据，再跑足够多圈数的情况下，记录碰撞总次数以及运行时长，关心collision and laptime 
# β 0.5 1 2 3 4 5 7 9 12 15 20 25 30 35 40
# image move 3 shift 

import matplotlib
matplotlib.use('tkagg')
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

torch.multiprocessing.set_start_method('spawn',force=True)
from visual.plot_multimodal_simple import Plotter_MultiModal


class holonomic_robot(Plotter_MultiModal):
    def __init__(self):
        super().__init__()

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
        
        exp_params = self.simple_task.exp_params
        self.sim_dt = exp_params['control_dt'] #0.1
        self.extents = np.ravel(exp_params['model']['position_bounds'])
        self.traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],'coll_cost':[],
                    'acc':[], 'world':None, 'bounds':self.extents , 'weights':[]}
        self.current_state = {'position':np.array([0.12,0.2]), 'velocity':np.zeros(2) + 0.0, 'acceleration':np.zeros(2) + 0.0 }

        # self.plot_init()

    def ReInit(self, current_state, goal_list):
        self.goal_list = goal_list
        self.goal_state = self.goal_list[-1]
        self.simple_task.update_params(goal_state=self.goal_state)
        self.current_state = {'position':current_state, 'velocity':np.zeros(2) + 0.0, 'acceleration':np.zeros(2) + 0.0 }
        # image 调整到初始位置 agent归位
        self.controller.rollout_fn.image_move_collision_cost.world_coll.move_ind = 10
        self.controller.rollout_fn.image_move_collision_cost.world_coll.im = self.controller.rollout_fn.image_move_collision_cost.world_coll.Start_Image


    def run(self):
        # temp parameters
        self.goal_flagi = -1 # 调控目标点
        self.loop_step = 0   #调控运行steps
        t_step = 0.0 # 记录run_time
        goal_thresh = 0.03 # 目标点阈值
        self.run_time = 0.0
        lap_count = 30 # 跑5轮次
        self.collisions_all = 0

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
                self.goal_state = self.goal_list[(self.goal_flagi+1) % len(self.goal_list)]
                self.simple_task.update_params(goal_state=self.goal_state) # 目标更变
                self.goal_flagi += 1
                print("next goal: {}%".format(self.goal_flagi / len(self.goal_list)/lap_count*100))
                
            t_step += self.sim_dt
            self.loop_step += 1
            # self.plot_setting()

            curr_pose = torch.as_tensor(self.current_state['position'], **self.tensor_args).unsqueeze(0)
            self.potential_curr = self.controller.rollout_fn.image_move_collision_cost.world_coll.get_pt_value(curr_pose) # 当前势场
            if self.potential_curr[0] > 0.99 : self.collisions_all += 1



import csv
fieldnames = ['β', 'lap_time', 'whileloop_count', 'collision_count']

def run_experiment(top_traj_select):

    lap_times = []
    whileloop_counts = []
    collision_counts = []

    CarController = holonomic_robot()
    CarController.controller.top_traj_select = top_traj_select

    goals_list = [
        [[0.8787878787878789, 0.7824675324675325], 
         [0.2240259740259739, 0.7851731601731602]],
        [[0.9087878787878789, 0.7824675324675325], 
         [0.2040259740259739, 0.7851731601731602]],
        [[0.8787878787878789, 0.7824675324675325], 
         [0.2240259740259739, 0.7851731601731602]]]    

    current_state_dict = np.array([[0.12,0.2],[0.15, 0.3],[0.10, 0.6],[0.10, 0.90]])


    with open('results.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not f.tell():
            writer.writeheader()

        # self.current_state = {'position':np.array([0.12,0.2]), 'velocity':np.zeros(2) + 0.0, 'acceleration':np.zeros(2) + 0.0 }
        for j in range(goals_list.__len__()):
            for i in range(current_state_dict.shape[0]):
                for lamda in [0.1,0.3, 0.5,0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 20, 25, 30, 35, 40, 45, 50]:

                    CarController.controller.lamda = lamda
                    first_time = time.time()
                    CarController.ReInit(current_state = current_state_dict[i], goal_list = goals_list[j])
                    CarController.run() 
                    lap_time = time.time() - first_time  

                    lap_times.append(lap_time)
                    whileloop_counts.append(CarController.loop_step)
                    collision_counts.append(CarController.collisions_all)

                    row = {'β': lamda, 'lap_time': lap_time, 'whileloop_count': CarController.loop_step, 'collision_count': CarController.collisions_all}
                            
                    writer.writerow(row)

                    print(row)

    return lap_times, whileloop_counts, collision_counts

if __name__ == '__main__':  
  
  print("Experiment start...")
  lap_times, whileloop_counts, collision_counts = run_experiment(top_traj_select=30)
  print("Experiment finished!")