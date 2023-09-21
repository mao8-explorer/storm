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
matplotlib.use('tkagg')
torch.multiprocessing.set_start_method('spawn',force=True)
from visual.plot_simple import Plotter


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
class holonomic_robot(Plotter):
    def __init__(self,args):
        super().__init__()

        self.goal_list = [
                # [0.9098484848484849, 0.2006060606060608],
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


if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    controller = holonomic_robot(args)
    controller.run()
