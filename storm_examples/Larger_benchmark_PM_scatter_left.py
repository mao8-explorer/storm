"""
本文件意在阐述 
不同的judge_weight 对 collision / crash_rate / opt_step / path length 的影响
分析具体的分布结果
意在 阐述 一般性的 single-layer MPPI 的性能局限

multimodal_cost:         
  switch_on: True
  target_cost:
    greedy_weight: 40.0
    sensi_weight: 5.0
    judge_weight: 10.0

  coll_cost: 
    greedy_weight: 1.0
    sensi_weight: 3.0
    judge_weight: 1.6

  terminal_reward:
    greedy_weight: 20.0
    sensi_weight: 0.0
    judge_weight: 1.0

coll_cost: 1.0 - 2.0 step : 0.10

必须要备注的是: 目前由于时间的关系，我暂时采取的策略是 左右移动的障碍物, greedy_policy 采用的是 greedy_coll
对于上下移动的障碍物， greedy_policy 采用的是 judge_coll
其中的原因是： greedy_coll 是P策略, 规划路径容易贴近障碍物，在上下移动的场景中，过分贴近障碍物，导致碰撞发生较多，影响实验结果；
在左右移动的工作场景中， P策略由于其能够有一定的反应能力,所以这种情况下，可以予以规避

"""
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


class holonomic_robot(Plotter_MultiModal):
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

    def ReInit(self,current_state, goal_list):
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

        self.run_time = 0.0
        self.collisions_all = 0
        self.crash_rate = 0.
        collision_hanppend = False
        while(self.goal_flagi / len(self.goal_list) != self.lap_count) and self.loop_step < 3500:
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
        weights = np.matrix(self.traj_log['weights'])
        mean_w = np.mean(weights, axis=0)[0,0]

        return trajectory_length, average_speed, max_speed, mean_w

            
fieldnames =  ['lamda','running_time','whileloop_count', 'collision_count', 'crash_rate', 
               'path_length', 'Avg.Speed', 'Max.Speed', 'Mean_weight','judge_coll_weight','note'] 

def run_experiment():

    CarController = holonomic_robot()

    """
        self.multiTargetCost_greedy_weight = self.multiTargetCost['greedy_weight']
        self.multiCollisionCost_greedy_weight = self.multiCollisionCost['greedy_weight']
        self.multiTerminalCost_greedy_weight = self.multiTerminalCost['greedy_weight']
        self.multiTargetCost_sensi_weight = self.multiTargetCost['sensi_weight']
        self.multiCollisionCost_sensi_weight = self.multiCollisionCost['sensi_weight']
        self.multiTerminalCost_sensi_weight = self.multiTerminalCost['sensi_weight']
        self.multiTargetCost_judge_weight = self.multiTargetCost['judge_weight']
        self.multiCollisionCost_judge_weight = self.multiCollisionCost['judge_weight']
        self.multiTerminalCost_judge_weight =  self.multiTerminalCost['judge_weight']
    """

    with open('./SDFcostlog/version0101/Larger_PM_LEFT_greedy.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not f.tell():
            writer.writeheader()

        current_state_dict = np.array([[0.12,0.2],[0.13, 0.15],[0.10, 0.30]])
        # current_state_dict = np.array([[0.12,0.2],[0.14, 0.15]])
        goals_list = [
            [[0.8687878787878789, 0.7824675324675325], 
            [0.2340259740259739, 0.7851731601731602]],
            # [[0.8887878787878789, 0.7824675324675325], 
            # [0.2140259740259739, 0.7851731601731602]],
            # [[0.8687878787878789, 0.7824675324675325], 
            # [0.2040259740259739, 0.7851731601731602]],
            [[0.8787878787878789, 0.7824675324675325], 
            [0.2240259740259739, 0.7851731601731602]] 
            ]  
        with open('./SDFcostlog/version0101/Larger_PM_LEFT_greedy_meanstd.csv', 'a', newline='') as var_file:
            var_writer = csv.DictWriter(var_file, fieldnames=fieldnames)
            if not var_file.tell():
                var_writer.writeheader()

            for lamda in [0.8,1.0,2.0]:
                CarController.controller.lamda = lamda
                # for coll_w in np.arange(1.0, 2.55, 0.1):
                #     CarController.controller.rollout_fn.multiCollisionCost_judge_weight = round(coll_w , 1)
                for g_w in np.arange(5, 40.1, 5):
                    CarController.controller.rollout_fn.multiTargetCost_judge_weight = round(g_w , 1)
                    for coll_w in np.arange(1, 4.1, 0.5):
                        CarController.controller.rollout_fn.multiCollisionCost_judge_weight = round(coll_w , 1)

                        results = []
                        for j in range(goals_list.__len__()):
                            for i in range(current_state_dict.shape[0]):
                                first_time = time.time()
                                CarController.ReInit(current_state = current_state_dict[i], goal_list = goals_list[j])
                                trajectory_length, average_speed, max_speed, mean_w = CarController.run() 
                                lap_time = time.time() - first_time  

                                row = {
                                    'lamda': lamda,
                                    'running_time': round(lap_time, 3),
                                    'whileloop_count': CarController.loop_step, 
                                    'collision_count': CarController.collisions_all,
                                    'crash_rate': round(CarController.crash_rate / (CarController.lap_count*len(CarController.goal_list)) * 100, 3),  
                                    'path_length': round(trajectory_length, 3), 
                                    'Avg.Speed': round(average_speed,3), 
                                    'Max.Speed': round(max_speed,3),
                                    'Mean_weight': mean_w,
                                    'judge_coll_weight': round(coll_w,1),

                                    }
                                writer.writerow(row)
                                f.flush()  # 刷新缓冲
                                print(row)
                                # CarController.plot_traj(root_path = './SDFcostlog/' , img_name = 'Rg_Coll_w_'+str(round(coll_w,1)) + '.png')
                                results.append(row)
                            
                        params = {key: [] for key in row}
                        for result in results:
                            for key in result:
                                params[key].append(result[key])
                        averages = {key: round(np.mean(values),3) for key, values in params.items()}
                        averages['note'] = 'average'
                        for key in params:
                            q1 = np.percentile(params[key], 25)
                            q3 = np.percentile(params[key], 75)
                            iqr = q3 - q1
                            outlier_range = 1.5 * iqr
                            params[key] = [value for value in params[key] if (q1 - outlier_range) <= value <= (q3 + outlier_range)]

                        averages_no_outliers = {key: round(np.mean(values),3) if values else 'N/A' for key, values in params.items()}
                        averages_no_outliers['note'] = 'averages_no_outliers'
                        writer.writerow(averages) # 均值保存
                        writer.writerow(averages_no_outliers) # 箱线图数据
                        f.flush()  # 刷新缓冲
                        var_writer.writerow(averages_no_outliers)
                        # Calculate variance and write it to a new file
                        variances = {key: round(np.var(values), 3) if values != 'N/A' else 'N/A' for key, values in params.items()}
                        variances['note'] = 'variance'
                        var_writer.writerow(variances)
                        var_file.flush()


if __name__ == '__main__':  
  
  print("Experiment start...")
  run_experiment()
  print("Experiment finished!")
  