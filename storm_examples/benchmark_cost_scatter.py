"""
本文件意在阐述 
不同的goal_weight / collision_weight 对 collision / crash_rate / opt_step / path length 的影响
分析具体的分布结果
意在 阐述 一般性的 single-layer MPPI 的性能局限

goal_cost : 5 -> 40 (interval = 5)   8个
coll_cost : 1 -> 4 (interval = 0.5)  7个
56个点

terminal_reward : 0 1 5 20 4种方案测试

"""
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
import csv

torch.multiprocessing.set_start_method('spawn',force=True)
from visual.plot_simple import Plotter


class holonomic_robot(Plotter):
    def __init__(self):
        super().__init__()

        # Task parameter
        self.shift = 3
        self.up_down = False
        self.goal_list = [
        # [0.9098484848484849, 0.2006060606060608],
         [0.8687878787878789, 0.7824675324675325], 
         [0.2340259740259739, 0.7851731601731602]]

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
        self.current_state = {'position':np.array([0.12,0.2]), 'velocity':np.zeros(2) + 0.0, 'acceleration':np.zeros(2) + 0.0 }
        self.traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],'coll_cost':[],
                    'acc':[], 'world':None, 'bounds':self.extents , 'weights':[]}
        self.plot_init()
        self.lap_count = 20
        self.goal_thresh = 0.03 # 目标点阈值

    def ReInit(self):
        self.traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],'coll_cost':[],
                    'acc':[], 'world':None, 'bounds':self.extents , 'weights':[]}
        self.goal_state = self.goal_list[-1]
        self.simple_task.update_params(goal_state=self.goal_state)
        self.current_state = {'position':np.array([0.12,0.2]), 'velocity':np.zeros(2) + 0.0, 'acceleration':np.zeros(2) + 0.0 }
        # image 调整到初始位置 agent归位
        self.controller.rollout_fn.image_move_collision_cost.world_coll.Reinit(self.shift, self.up_down) # command handle


    def run(self):
        # temp parameters
        self.goal_flagi = -1 # 调控目标点
        self.loop_step = 0   #调控运行steps
        t_step = 0.0 # 记录run_time

        self.run_time = 0.0
        self.collisions_all = 0
        self.crash_rate = 0.
        collision_hanppend = False
        while(self.goal_flagi / len(self.goal_list) != self.lap_count) and self.loop_step < 5000:
            #  core_process
            self.controller.rollout_fn.image_move_collision_cost.world_coll.updateSDFPotientailGradient() #更新环境SDF
            last = time.time()
            command = self.simple_task.get_command(self.current_state)
            self.run_time += time.time() - last
            self.current_state = command # or command * scale
            # 这里的current_coll 反馈的不是是否发生碰撞，是forward计算的值，暂无意义
            _, goal_dist,_ = self.simple_task.get_current_error(self.current_state) 
            # goal_reacher update
            if goal_dist[0] < self.goal_thresh:
                if collision_hanppend : self.crash_rate += 1
                collision_hanppend = False
                self.goal_state = self.goal_list[(self.goal_flagi+1) % len(self.goal_list)]
                self.simple_task.update_params(goal_state=self.goal_state) # 目标更变
                self.goal_flagi += 1
                
            t_step += self.sim_dt
            

            # self.plot_setting()
            # if self.pause:
            #     while True:
            #         time.sleep(0.5)
            #         self.plot_setting()
            #         if not self.pause: break

            curr_pose = torch.as_tensor(self.current_state['position'], **self.tensor_args).unsqueeze(0)
            self.potential_curr = self.controller.rollout_fn.image_move_collision_cost.world_coll.get_pt_value(curr_pose) # 当前势场
            if self.potential_curr[0] > 0.80 : 
                self.collisions_all += 1
                collision_hanppend = True
            if self.goal_flagi > -1 :
                self.traj_append()
                self.loop_step += 1
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

            
fieldnames =  ['whileloop_count', 'collision_count', 'crash_rate', 'path_length', 'Avg.Speed', 'Max.Speed', 
               'coll_w', 'goal_w', 'reward_w'] 

def run_experiment():

    CarController = holonomic_robot()

    with open('./SDFcostlog/benchmark_cost_scatter_LeftRight.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not f.tell():
            writer.writeheader()

        for r_w in [1.0, 5.0, 10.0]:
            CarController.controller.rollout_fn.sparse_reward.weight = torch.tensor(r_w, **CarController.tensor_args)
            for g_w in np.arange(5, 20.1, 5):
                CarController.controller.rollout_fn.goal_cost.weight = torch.tensor(g_w, **CarController.tensor_args)
                for coll_w in np.arange(1, 4.1, 1.0):
                    CarController.controller.rollout_fn.image_move_collision_cost.weight = torch.tensor(coll_w, **CarController.tensor_args)

                    CarController.ReInit()
                    trajectory_length, average_speed, max_speed = CarController.run() 

                    row = {
                        'whileloop_count': CarController.loop_step, 
                        'collision_count': CarController.collisions_all,
                        'crash_rate': round(CarController.crash_rate / (CarController.lap_count*len(CarController.goal_list)) * 100, 3),  
                        'path_length': round(trajectory_length, 3), 
                        'Avg.Speed': round(average_speed,3), 
                        'Max.Speed': round(max_speed,3),
                        'coll_w': coll_w,
                        'goal_w': g_w,
                        'reward_w': r_w
                        }
                    writer.writerow(row)
                    f.flush()  # 刷新缓冲
                    print(row)




if __name__ == '__main__':  
  
  print("Experiment start...")
  run_experiment()
  print("Experiment finished!")
  