
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import copy
matplotlib.use('tkagg')
torch.multiprocessing.set_start_method('spawn',force=True)


class Plotter_MultiModal(object):
    def __init__(self):
        self.tensor_args = {'device':'cuda','dtype':torch.float32}
        self.fig = plt.figure()
        self.ax = plt.subplot(1, 1, 1)
        self.X, self.Y = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
        coordinates = np.column_stack((self.X.flatten(), self.Y.flatten()))
        self.coordinates = torch.as_tensor(coordinates, **self.tensor_args)
        self.fig.canvas.mpl_connect('button_press_event', self.press_call_back)
        self.fig.canvas.mpl_connect('key_press_event', self.key_call_back)

    def plot_setting(self):

        self.ax.cla() #清屏
        # self.controller.rollout_fn.image_collision_cost.world_coll.update_world()

        dist_map = self.controller.rollout_fn.image_move_collision_cost.world_coll.dist_map # 获取障碍图像，im:原始图像 dist_map: 碰撞图像（会被0-1化离散表征）
        im = self.controller.rollout_fn.image_move_collision_cost.world_coll.im 
        image = cv2.addWeighted(im.astype(float), 0.5, dist_map.cpu().numpy().astype(float), 0.5, 1).astype(np.uint8)

        self.traj_log['world'] = image
        # self.ax.imshow(self.traj_log['world'], extent=self.extents,cmap='gray')
        self.ax.imshow(self.traj_log['world'], extent=self.extents)
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

        greedy_mean_traj = self.simple_task.control_process.controller.greedy_mean_traj.cpu().numpy()
        sensi_mean_traj = self.simple_task.control_process.controller.sensi_mean_traj.cpu().numpy()
        mean_traj = self.simple_task.control_process.controller.mean_traj.cpu().numpy()
        # top_trajs = self.simple_task.top_trajs
        # _, _ ,coll_cost= self.simple_task.get_current_coll(top_trajs) 
        # self.traj_log['top_traj'] = top_trajs.cpu().numpy()
        # 15条轨迹，前5条最优轨迹，后10条最差轨迹
        # self.ax.scatter(np.ravel(self.traj_log['top_traj'][:5,:,0].flatten()), np.ravel(self.traj_log['top_traj'][:5,:,1].flatten()),
        #         c='green',s=np.array(2))
        # self.ax.scatter(np.ravel(self.traj_log['top_traj'][5:,:,0].flatten()), np.ravel(self.traj_log['top_traj'][5:,:,1].flatten()),
        #         c=np.ravel(coll_cost[0].cpu().numpy()[100:]),  s=np.array(2))
        # random_shooting: best_trajectory 绿线
        # self.ax.plot(np.ravel(self.traj_log['top_traj'][0,:,0].flatten()), np.ravel(self.traj_log['top_traj'][0,:,1].flatten()),
        #         'g-',linewidth=1,markersize=3)          
        # MPPI : mean_trajectory 红线
        self.ax.plot(np.ravel(greedy_mean_traj[:,0]),np.ravel(greedy_mean_traj[:,1]),
                'r-',linewidth=2,markersize=3)  
        self.ax.plot(np.ravel(sensi_mean_traj[:,0]),np.ravel(sensi_mean_traj[:,1]),
                'b-',linewidth=2,markersize=3)  
        self.ax.plot(np.ravel(mean_traj[:,0]),np.ravel(mean_traj[:,1]),
                'g-',linewidth=4,markersize=3)  

        #  文字标签 ----------------------------------------------------------------
        #  velocity | potential | 夹角  | MPQ Value_Function估计 
        grad_orient = np.array([np.ravel(grad_x_curr.cpu())[0],np.ravel(grad_y_curr.cpu())[0]])
        grad_magnitude = np.linalg.norm(grad_orient)  # 计算向量 b 的范数（长度）
        cos_theta = np.dot(self.current_state['velocity'], grad_orient) / (velocity_magnitude * grad_magnitude + 1e-7)  # 计算余弦值
        theta = np.arccos(cos_theta)  # 计算夹角（弧度）
        
        self.ax.text(0.6, 1.01, f'potential: {np.ravel(self.potential_curr.cpu())[0]:.3f}, collcost: {np.ravel(self.potential_curr.cpu())[0] * velocity_magnitude * (-cos_theta):.3f}', 
                            fontsize=12, color='red' if self.potential_curr[0] > 0 else 'black')
        self.ax.text(0.6, 1.04, f'Velocity Magnitude: {velocity_magnitude:.3f}', fontsize=12, color='black')
        self.ax.text(0.6, 1.07, f'angle: {np.degrees(theta):.3f}', fontsize=12, color='red' if theta > np.pi/2.0 else 'black')
         # MPQ value值估计 log_sum_exp
        self.ax.text(1.04, 0.5, f'value: {self.value_function.cpu().numpy()}', fontsize=12)
        self.ax.text(1.04, 0.4, f'greedy_w1: {self.controller.weights_divide.cpu().numpy()[0]:.3f}', fontsize=12 ,
                     color='red' if self.controller.weights_divide.cpu().numpy()[0] > 0.5 else 'black')
        self.ax.text(1.04, 0.36, f'sensi_w2: {self.controller.weights_divide.cpu().numpy()[1]:.3f}', fontsize=12 ,
                     color='red' if self.controller.weights_divide.cpu().numpy()[1] > 0.5 else 'black')
        self.ax.text(1.04, 0.32, f'cov: {self.controller.cov_action.cpu().numpy()[0]:.4f} , {self.controller.cov_action.cpu().numpy()[1]:.4f}', fontsize=12)
        plt.pause(1e-10)
        self.traj_append()

    def press_call_back(self,event):
        self.goal_state = [event.xdata,event.ydata]
        self.simple_task.update_params(goal_state=self.goal_state) # 目标更变
        print(self.goal_state)

    def key_call_back(self,event):
        self.pause = not self.pause


    def traj_append(self):

        self.traj_log['position'].append(self.current_state['position'])
        self.traj_log['velocity'].append(self.current_state['velocity'])
        self.traj_log['command'].append(self.current_state['acceleration'])
        self.traj_log['acc'].append(self.current_state['acceleration'])
        self.traj_log['coll_cost'].append(self.potential_curr.cpu()[0])
        self.traj_log['des'].append(copy.deepcopy(self.goal_state))



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
