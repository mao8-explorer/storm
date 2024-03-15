
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
import pickle


torch.multiprocessing.set_start_method('spawn',force=True)
from visual.plot_multimodal_simple import Plotter_MultiModal


"""
multi modal MPPI 运动特征可视化
可视化的目的在于 更加形象立体的展示算法的 多模态特点。 所以，不仅仅要展示运动过程，还要展示运动过程的权重分配以及轨迹的速度 位置信息
数据可通过pickle 的方式 dump 存储 load加载 进行预处理
traj_append 要加入目标点切换时的step, 让后面的数据再处理更好操作

"""

class holonomic_robot(Plotter_MultiModal):
    def __init__(self):
        super().__init__()

        # Task parameter
        self.shift = 3
        self.up_down =  True
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
                    'acc':[], 'world':None, 'bounds':self.extents , 'weights':[],
                    'thresh_index':[]}
        self.plot_init()

        self.fieldnames = ['lap_time', 'whileloop_count', 
                           'collision_count', 'crash_rate',
                           'Avg.Speed', 'Max.Speed','path_length', 
                           'Note'] 


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
                    self.traj_log['thresh_index'].append(self.loop_step)
                    
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
               'Note': "PM move=3"
               }
        print(row)
        
        self.plot_traj(root_path = './SDFcostlog/' , img_name = '3points.png')


        # self.save_traj_log()
        with open('./SDFcostlog/benchmark_SDFcost_2D.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not f.tell():
                writer.writeheader()
            writer.writerow(row)
        # 轨迹数据保存 以 再处理
        # with open('./SDFcostlog/visual_traj.pkl', 'wb') as f:
        #     pickle.dump(self.traj_log, f)

if __name__ == '__main__':  

    CarController = holonomic_robot()
    CarController.run() 