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

        self.traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],'coll_cost':[],
                    'acc':[], 'world':None, 'bounds':self.extents}
        self.plot_init() # 这里是动态显示2D plot的初始化部分  配合plot-setting 在while循环中一起食用
        self.current_state = {'position':np.array([0.12,0.2]), 'velocity':np.zeros(2) + 0.0, 'acceleration':np.zeros(2) + 0.0 }

        self.fieldnames = ['lap_time', 'whileloop_count', 'collision_count', 'crash_rate','Avg.Speed', 'Max.Speed','path_length', 'Note'] 

    def run(self):
        # temp parameters
        self.goal_flagi = -1 # 调控目标点
        self.loop_step = 0   #调控运行steps
        t_step = 0.0 # 记录run_time
        goal_thresh = 0.03 # 目标点阈值
        self.run_time = 0.0
        lap_count = 20 # 跑5轮次
        first_time = time.time()
        self.collisions_all = 0
        self.crash_rate = 0.
        collision_hanppend = False
        try:
            while(self.goal_flagi / len(self.goal_list) != lap_count) and  self.loop_step < 3500:
                #  core_process
                self.controller.rollout_fn.image_move_collision_cost.world_coll.updateSDFPotientailGradient() #更新环境SDF
                last = time.time()
                command = self.simple_task.get_command(self.current_state)
                self.run_time += time.time() - last
                self.current_state = command # or command * scale
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
        except KeyboardInterrupt:
            print("Unexpected interruption ... ")
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


        whole_time = time.time() - first_time
        row = {'lap_time': round(whole_time,3), 
               'whileloop_count': self.loop_step, 
               'collision_count': self.collisions_all,
               'crash_rate': round(self.crash_rate / (lap_count*len(self.goal_list)) * 100, 3),  
               'path_length': round(trajectory_length, 3), 
               'Avg.Speed': round(average_speed,3), 
               'Max.Speed': round(max_speed,3),
               'Note': "P—balanced-left move=3"
               }
        print(row)
        self.plot_traj(img_name = './SDFcostlog/P-left_right.png')
        # self.save_traj_log()
        with open('./SDFcostlog/benchmark_SDFcost_2D.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not f.tell():
                writer.writeheader()
            writer.writerow(row)
  



if __name__ == '__main__':
    
    controller = holonomic_robot()
    controller.run()
