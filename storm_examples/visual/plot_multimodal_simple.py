
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import copy
import matplotlib.cm as cm
from matplotlib.colors import CSS4_COLORS
matplotlib.use('tkagg')
torch.multiprocessing.set_start_method('spawn',force=True)


class Plotter_MultiModal(object):
    def __init__(self):
        self.tensor_args = {'device':'cuda','dtype':torch.float32}
        self.X, self.Y = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
        coordinates = np.column_stack((self.X.flatten(), self.Y.flatten()))
        self.coordinates = torch.as_tensor(coordinates, **self.tensor_args)
        self.collision_count = 0

    def plot_init(self):
        self.fig = plt.figure()
        self.ax = plt.subplot(1, 1, 1)
        self.ax.axis('off')
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
        #  全局SDF_Gradient绘画 翻转x,y是坐标变化机理
        grad_y,grad_x = self.controller.rollout_fn.image_move_collision_cost.world_coll.get_pt_gradxy(self.coordinates)
        # # #  绘制箭头
        self.ax.quiver(self.X,self.Y, np.ravel(grad_x.view(30,-1).cpu()), np.ravel(grad_y.view(30,-1).cpu()), color=CSS4_COLORS['khaki'], alpha=0.8)

        # 当前状态速度指向 | 当前位置SDF梯度指向 | 全局地图SDF梯度可视化
        velocity_magnitude = np.linalg.norm(self.current_state['velocity'], axis=0)  # 计算速度大小
        self.ax.quiver( np.ravel(self.current_state['position'][0]), np.ravel(self.current_state['position'][1]),
                        np.ravel(self.current_state['velocity'][0]),  np.ravel(self.current_state['velocity'][1]), 
                        velocity_magnitude, cmap=plt.cm.jet) # 当前状态 速度大小及方向
        curr_pose = torch.as_tensor(self.current_state['position'], **self.tensor_args).unsqueeze(0)
        grad_y_curr,grad_x_curr = self.controller.rollout_fn.image_move_collision_cost.world_coll.get_pt_gradxy(curr_pose) # 当前SDF梯度
        self.potential_curr = self.controller.rollout_fn.image_move_collision_cost.world_coll.get_pt_value(curr_pose) # 当前势场
        if self.potential_curr[0] > 0.99 : self.collision_count += 1
        self.ax.quiver(np.ravel(self.current_state['position'][0]), np.ravel(self.current_state['position'][1]),
                    np.ravel(grad_x_curr.cpu()),np.ravel(grad_y_curr.cpu()),color='red') # 当前位置所在SDF梯度
        

        # 散点标签 ----------------------------------------------------------------
        # 当前位置状态 
        self.ax.plot(self.goal_state[0], self.goal_state[1], 'gX', linewidth=3.0, markersize=15) # 目标点
        self.ax.scatter(np.ravel(self.current_state['position'][0]),np.ravel(self.current_state['position'][1]),
                        c=np.ravel(self.potential_curr.cpu()),s=np.array(400),cmap=plt.cm.jet, vmin=0, vmax=1)
        # 规划轨迹 batch_trajectories visual
        # goto control_base.py check
        greedy_mean_traj = self.simple_task.control_process.controller.greedy_mean_traj.cpu().numpy()
        sensi_mean_traj = self.simple_task.control_process.controller.sensi_mean_traj.cpu().numpy()
        mean_traj = self.simple_task.control_process.controller.mean_traj.cpu().numpy()

        # multimodal mean trajectories
        # greedy_top_trajs = self.simple_task.control_process.controller.greedy_top_trajs.cpu().numpy()
        # sensi_top_trajs = self.simple_task.control_process.controller.sensi_top_trajs.cpu().numpy()
        # sample multi-mean trajs.
        # sensi_sample_trajs = self.simple_task.control_process.controller.sensi_sample_trajs.cpu().numpy()
        # greedy_sample_trajs = self.simple_task.control_process.controller.greedy_sample_trajs.cpu().numpy()
        # mean_sample_trajs = self.simple_task.control_process.controller.mean_sample_trajs.cpu().numpy()  
        # self.ax.scatter(np.ravel(sensi_sample_trajs[:,:,0].flatten()), np.ravel(sensi_sample_trajs[:,:,1].flatten()),
        #         c='green',s=np.array(2))   
        # self.ax.scatter(np.ravel(greedy_sample_trajs[:,:,0].flatten()), np.ravel(greedy_sample_trajs[:,:,1].flatten()),
        #         c='green',s=np.array(2))   
        # self.ax.scatter(np.ravel(mean_sample_trajs[:,:,0].flatten()), np.ravel(mean_sample_trajs[:,:,1].flatten()),
        #         c='green',s=np.array(2))   
        # top_trajs = self.simple_task.top_trajs
        # _, _ ,coll_cost= self.simple_task.get_current_coll(top_trajs) 
        # self.traj_log['top_traj'] = top_trajs.cpu().numpy()
        # 15条轨迹，前5条最优轨迹，后10条最差轨迹
        # self.ax.scatter(np.ravel(greedy_top_trajs[:,:,0].flatten()), np.ravel(greedy_top_trajs[:,:,1].flatten()),
        #         c='red',s=np.array(2))
        # self.ax.scatter(np.ravel(sensi_top_trajs[:,:,0].flatten()), np.ravel(sensi_top_trajs[:,:,1].flatten()),
        #         c='green',s=np.array(2))
        # self.ax.scatter(np.ravel(self.traj_log['top_traj'][5:,:,0].flatten()), np.ravel(self.traj_log['top_traj'][5:,:,1].flatten()),
        #         c=np.ravel(coll_cost[0].cpu().numpy()[100:]),  s=np.array(2))
        # random_shooting: best_trajectory 绿线
        # self.ax.plot(np.ravel(self.traj_log['top_traj'][0,:,0].flatten()), np.ravel(self.traj_log['top_traj'][0,:,1].flatten()),
        #         'g-',linewidth=1,markersize=3)          
        # MPPI : mean_trajectory 红线
        self.ax.plot(np.ravel(greedy_mean_traj[:,0]),np.ravel(greedy_mean_traj[:,1]),
                'r-',linewidth=3,markersize=3)  
        self.ax.plot(np.ravel(sensi_mean_traj[:,0]),np.ravel(sensi_mean_traj[:,1]),
                'g-',linewidth=3,markersize=3)  
        self.ax.plot(np.ravel(mean_traj[:,0]),np.ravel(mean_traj[:,1]),
                'b-',linewidth=4,markersize=3)  

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
        self.ax.text(1.04, 0.32, f'cov: {self.controller.cov_action.cpu().numpy()[0]:.4f} , {self.controller.cov_action.cpu().numpy()[1]:.4f}', fontsize=12)
        self.ax.text(1.04, 0.90, f'lap_count: {self.goal_flagi / len(self.goal_list)}', fontsize=12, color='black')
        self.ax.text(1.04, 0.87, f'whileloop_count: {self.loop_step}', fontsize=12, color='black')
        self.ax.text(1.04, 0.84, f'opt_runtime: {self.run_time}', fontsize=12, color='black')
        self.ax.text(1.04, 0.81, f'opt_hz: {self.loop_step /self.run_time}', fontsize=12, color='black')
        self.ax.text(1.04, 0.78, f'collision_count: {self.collision_count}', fontsize=12, color='black')

        plt.pause(1e-10)
        # self.traj_append()

    def press_call_back(self,event):
        self.goal_state = [event.xdata,event.ydata]
        self.simple_task.update_params(goal_state=self.goal_state) # 目标更变
        print(self.goal_state)

    def key_call_back(self,event):
        self.pause = not self.pause


    def traj_append(self):
        self.traj_log['position'].append(self.current_state['position'])
        self.traj_log['coll_cost'].append(self.potential_curr.cpu()[0])
        self.traj_log['velocity'].append(self.current_state['velocity'])
        # self.traj_log['command'].append(self.current_state['acceleration'])
        self.traj_log['acc'].append(self.current_state['acceleration'])
        self.traj_log['des'].append(copy.deepcopy(self.goal_state))
        self.traj_log['weights'].append(self.controller.weights_divide.cpu().numpy())

    def plot_traj(self, root_path = './SDFcostlog/'  , img_name = 'multimodalPPV.png'):
        plt.figure()
        position = np.matrix(self.traj_log['position'])
        vel = np.matrix(self.traj_log['velocity'])
        coll = np.matrix(self.traj_log['coll_cost'])
        acc = np.matrix(self.traj_log['acc'])
        des = np.matrix(self.traj_log['des'])
        weights = np.matrix(self.traj_log['weights'])
        axs = [plt.subplot(4,1,i+1) for i in range(4)]
        if(len(axs) >= 3):
            axs[0].set_title('Position')
            axs[1].set_title('Velocity')
            axs[2].set_title('Acceleration')
            axs[3].set_title('weights')
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
            axs[3].plot(weights[:,0], 'r', label='greedy')
            axs[3].plot(weights[:,1], 'g', label='sensi')
            axs[3].legend()

            plt.tight_layout()  # 可以加入这一句对图像进行优化布局
            plt.savefig(root_path + 'traj_'+img_name)  # 保存图片
            plt.close()  # 关闭图像避免显示

        #prepare trajectory background
        collision_map_path = "/home/zm/MotionPolicyNetworks/storm_ws/history/storm/content/assets/collision_maps/collision_map_cem.png"
        im = cv2.imread(collision_map_path,0)
        _,im = cv2.threshold(im,10,255,cv2.THRESH_BINARY)
        rows, cols = im.shape
        shift =  self.shift*10
        if self.up_down: 
            movelist = np.float32([
            [[1, 0, 0], [0, 1,  shift]],
            [[1, 0, 0], [0, 1, -shift]]])
        else:
            movelist = np.float32([
                [[1, 0, -shift], [0, 1, 0]],
                [[1, 0,  shift], [0, 1, 0]]])
        im_down = cv2.warpAffine(im, movelist[1], (cols, rows),borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        im_up = cv2.warpAffine(im, movelist[0], (cols, rows),borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        # 创建叠加后的图像
        overlay = np.zeros_like(im, dtype=np.float32)
        alpha = 0.1  # 调整透明度的值
        overlay[im_down > 0] += alpha
        overlay[im_up > 0] += alpha

        plt.figure()
        extents = (self.traj_log['bounds'][0], self.traj_log['bounds'][1],
                self.traj_log['bounds'][2], self.traj_log['bounds'][3])
        img_ax = plt.subplot(1,1,1)
        # img_ax.imshow(self.controller.rollout_fn.image_move_collision_cost.world_coll.Start_Image,cmap='gray', extent=extents)
        img_ax.imshow(im, cmap='gray',extent=extents)
        img_ax.imshow(overlay, cmap='gray', alpha=0.2,extent=extents)
        # img_ax.plot(np.ravel(position[0,0]), np.ravel(position[0,1]), 'rX', linewidth=3.0, markersize=15)
        img_ax.plot(des[:,0], des[:,1],'gX', linewidth=3.0, markersize=15)
        # img_ax.scatter(np.ravel(position[:,0]),np.ravel(position[:,1]),c=np.ravel(coll),s=np.array(2),marker='+')

        cmap = cm.get_cmap('viridis')
        coll = np.ravel(coll)
        for i in range(len(position) - 1):
            plt.plot([position[i, 0], position[i+1, 0]], [position[i, 1], position[i+1, 1]], lw=1, color=cmap(coll[i]))
        # plt.colorbar(label='Coll')
        img_ax.set_xlim(self.traj_log['bounds'][0], self.traj_log['bounds'][1])
        img_ax.set_ylim(self.traj_log['bounds'][2], self.traj_log['bounds'][3])
        plt.axis('off')
        plt.savefig(root_path + img_name)
        plt.show()
        # plt.close()  # 关闭图像避免显示
