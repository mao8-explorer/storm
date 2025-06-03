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
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
np.int = int
np.float = float
np.bool = bool


import torch
import torch.nn.functional as F
import math

from storm_kit.mpc.rollout.arm_base import ArmBase
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path, get_mpc_configs_path, get_weights_path
import yaml
from storm_kit.mpc.control.control_utils import generate_halton_samples
from storm_kit.geom.nn_model.robot_self_collision import RobotSelfCollisionNet
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 确保交互式绘图有效


def init_live_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    train_line, = ax.plot([], [], label='Train RMSE (m)', color='blue', linewidth=2)
    val_line, = ax.plot([], [], label='Validation RMSE (m)', color='orange', linewidth=2)
    coll_line, = ax.plot([], [], label='Collision RMSE (m)', color='green', linestyle='--', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('RMSE (meters)', fontsize=12)
    ax.set_title('Live Training Loss Curve', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()

    return fig, ax, train_line, val_line, coll_line


def update_live_plot(fig, ax, train_line, val_line, coll_line,
                      train_losses, val_losses, coll_losses, window=100):
    total_epochs = len(train_losses)
    start_idx = max(0, total_epochs - window)

    epochs_x = list(range(start_idx, total_epochs))

    train_y = train_losses[start_idx:]
    val_y = val_losses[start_idx:]
    coll_y = coll_losses[start_idx:]

    train_line.set_data(epochs_x, train_y)
    val_line.set_data(epochs_x, val_y)
    coll_line.set_data(epochs_x, coll_y)

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_loss_curves(train_losses, val_losses, coll_losses=None, save_path=None):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_losses, label='Train RMSE (m)', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation RMSE (m)', linewidth=2)

    if coll_losses is not None:
        plt.plot(epochs, coll_losses, label='Collision Sample RMSE (m)', linestyle='--', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RMSE (meters)', fontsize=12)
    plt.title('Training Loss Curve in Physical Units (m)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Loss plot saved to: {save_path}")

    plt.tight_layout()
    # 不调用 show()，用于非交互保存


def print_epoch_rmse(e, train_loss, val_loss, coll_loss, std_y):
    train_rmse = math.sqrt(train_loss) * std_y # 反归一化
    val_rmse = math.sqrt(val_loss) * std_y
    coll_rmse = math.sqrt(coll_loss) * std_y
    print(f"Epoch {e:03d} | Train RMSE: {train_rmse:.5f} m | Val RMSE: {val_rmse:.5f} m | Coll RMSE: {coll_rmse:.5f} m")

    return train_rmse, val_rmse, coll_rmse


class RobotDataset(torch.utils.data.Dataset):
    def __init__(self, x,y,y_gt):
        self.x = x
        self.y = y
        self.y_gt = y_gt
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, idx):
        sample = {'x': self.x[idx,:], 'y': self.y[idx,:], 'y_gt': self.y_gt[idx,:]}
        return sample
    
def create_dataset(robot_name):
    checkpoints_dir = get_weights_path()+'/robot_self/test'
    num_particles = 10000
    task_file = robot_name+'_reacher.yml'

    # load robot model:
    device = torch.device('cuda', 0) 
    tensor_args = {'device':device, 'dtype':torch.float32}
    mpc_yml_file = join_path(get_mpc_configs_path(), task_file)

    with open(mpc_yml_file) as file:
        exp_params = yaml.load(file, Loader=yaml.FullLoader)
    exp_params['robot_params'] = exp_params['model'] #robot_params
    exp_params['cost']['primitive_collision']['weight'] = 0.0
    exp_params['control_space'] = 'pos'
    exp_params['mppi']['horizon'] = 2
    exp_params['mppi']['num_particles'] = num_particles
    # 默认尝试加载 robot_name+'_self_sdf.pt'
    rollout_fn = ArmBase(exp_params, tensor_args, world_params=None)
    
    # sample joint angles
    dof = rollout_fn.dynamics_model.d_action
    q_samples = generate_halton_samples(num_particles*2, dof, use_ghalton=True, seed_val=123,
                                        device=tensor_args['device'],
                                        float_dtype=tensor_args['dtype'])


    # rollout_fn, ndof 已定义
    robot_model = rollout_fn.dynamics_model.robot_model
    name_to_idx = robot_model._name_to_idx_map
    idx_to_name = {idx: name for name, idx in name_to_idx.items()}

    controlled = robot_model._controlled_joints                   # list of link 索引
    up_bounds = rollout_fn.dynamics_model.state_upper_bounds[:dof]
    low_bounds = rollout_fn.dynamics_model.state_lower_bounds[:dof]

    # 偏置量
    offset = 0.0 #math.pi / 10
    # 放大上下限
    low_bounds = low_bounds - offset
    up_bounds  = up_bounds  + offset

    print("受控关节及其取值范围：")
    for dof_idx, link_idx in enumerate(controlled):
        name = idx_to_name[link_idx]
        low, up = low_bounds[dof_idx], up_bounds[dof_idx]
        print(f"  Joint {dof_idx+1:2d} | {name:<20s} | [{low: .3f}, {up: .3f}]")



    # # 故意放大bounds， 目的： 增加碰撞样本，不然数据差异很少，可以通过print(torch.min(y), torch.max(y))判断
    # up_bounds = torch.full_like(up_bounds, math.pi)
    # low_bounds = torch.full_like(low_bounds, -math.pi)

    range_b = up_bounds - low_bounds
    q_samples = q_samples * range_b + low_bounds
    q_samples = q_samples.view(num_particles,2,dof)

    
    start_state = torch.zeros((rollout_fn.dynamics_model.d_state), **tensor_args)

    state_dict = rollout_fn.dynamics_model.rollout_open_loop(start_state.unsqueeze(0), q_samples)
    
    link_pos_seq = state_dict['link_pos_seq']
    link_rot_seq = state_dict['link_rot_seq']
    # compute link poses
    cost = rollout_fn.robot_self_collision_cost.distance
    dist = cost(link_pos_seq, link_rot_seq)


    # dataset:
    x = q_samples.view(num_particles*2, dof)


    y = dist.view(num_particles*2,1) #* 100.0

    # x_data  = x.cpu().numpy()
    # torch.save(x, 'x_data.p')
    # torch.save(y, 'y_data.p')
    # plt.scatter(x_data[:,1], x_data[:,3], c=y.cpu().numpy(), vmin=-0.1, vmax=0.1,cmap='coolwarm')
    # plt.show()
    # print(torch.min(y), torch.max(y))
    # x = x[y > -0.02]
    # y[y < -0.02] = -0.02
    # y[y < -0.1] = 0.1
    # y[y >= -0.02] = 1.0
    # y[y < -0.02] = 0.0
    
    print(torch.min(y), torch.max(y)) 
    
    n_size = x.shape[0]
    #print(n_size)
    # 
    # 新建的new model
    # nn_model = RobotSelfCollisionNet(n_joints=dof)
    # nn_model.model.to(**tensor_args)
    # model = nn_model.model

    # rollout_fn 初始化并加载weight后的 | 如果没有weight，就不在加载，仍正常运行
    enhance_model = rollout_fn.robot_self_collision_cost.coll.robot_nn
    model = enhance_model.model  # ✨ 绑定你要训练的 model
    model.to(**tensor_args)      # ⚠️ 确保模型迁移到正确设备


    # load training set:
    x_train = x[:int((n_size)*0.7),:]
    y_train = y[:int((n_size)*0.7)]
    coll_thresh = 0.005
    x_coll = x_train[y_train[:,0]> coll_thresh]#.cpu().numpy()
    y_coll = y_train[y_train[:,0]> coll_thresh]#.cpu().numpy()
    # 计算碰撞比例（float 标量）
    collision_ratio = (y_train[:, 0] > coll_thresh).float().mean().item()

    # 以百分比格式打印，保留两位小数
    print(f"Collision ratio: {collision_ratio:.2%}")

    #x_data = x_train[y_train[:,0]>-0.01].cpu().numpy()
    #y_data = y_train[y_train[:,0]>-0.01].cpu().numpy()
    #x_train = x_train.cpu().numpy()
    #plt.scatter(x_data[:,1], x_data[:,3],c=y_data,vmin=-0.1, vmax=0.1, cmap='coolwarm')
    #plt.show()

    # scale dataset: 查看数据分布是否合理 对数据进行了缩放，即将输入数据和标签数据归一化到均值为0、标准差为1的分布
    # mean_x = torch.mean(x, dim=0)#* 0.0 #+ 1.0
    # std_x = torch.mean(x, dim=0)* 0.0 + 1.0
    # mean_y = torch.mean(y, dim=0)#* 0.0 #+ 1.0
    # std_y = torch.mean(y, dim=0)#* 0.0 + 1.0

    mean_x = torch.mean(x, dim=0)
    std_x = torch.std(x, dim=0) + 1e-6  # 避免除0
    mean_y = torch.mean(y, dim=0)
    std_y = torch.std(y, dim=0) + 1e-6

    
    x_train = torch.div((x_train - mean_x),std_x)
    #x_train[x_train!=x_train] = 0.0
    x_coll = torch.div(x_coll - mean_x, std_x).detach()
    y_coll = torch.div(y_coll - mean_y, std_y).detach()
    y_train_true = y_train.clone()
    y_train = torch.div((y_train - mean_y),std_y)
    #y_train[y_train!=y_train] = 0.0
    #d = y[int((n_size*2)*0.9):]
    #print(d[d>0.0].shape)
    #exit()
    x = torch.div(x - mean_x,std_x)
    x[x!=x] = 0.0
    y = torch.div(y - mean_y,std_y)
    y[y!=y] = 0.0
    # x_val = x[int((n_size)*0.7):int((n_size)*0.9),:]
    # y_val = y[int((n_size)*0.7):int((n_size)*0.9)]

    # 将碰撞/非碰撞样本分别抽一部分给 val
    mask_coll = (y[:,0] > coll_thresh).squeeze()
    mask_free = ~mask_coll

    x_coll_val = x[mask_coll][:int(0.1 * mask_coll.sum())]
    y_coll_val = y[mask_coll][:int(0.1 * mask_coll.sum())]

    x_free_val = x[mask_free][:int(0.2 * mask_free.sum())]
    y_free_val = y[mask_free][:int(0.2 * mask_free.sum())]

    x_val = torch.cat([x_coll_val, x_free_val], dim=0)
    y_val = torch.cat([y_coll_val, y_free_val], dim=0)

    train_dataset = RobotDataset(x_train.detach(), y_train.detach(), y_train_true.detach())
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    coll_dataset = RobotDataset(x_coll.detach(), y_coll.detach(), y_coll.detach())
    collloader = torch.utils.data.DataLoader(coll_dataset, batch_size=64, shuffle=True)


    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
        # optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)#,momentum=0.97)
    # 学习率调度器：val_loss 停滞时降低 lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-5
    )

    # EMA 模型参数缓存（指数滑动平均）
    ema_model = deepcopy(model)
    ema_decay = 0.995  # 越接近 1，变化越慢

    def update_ema(model, ema_model, decay):
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


    # model参数配置
    epochs = 500
    min_loss = 100.0
    alpha = 2.0 # collision数据的偏置

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = deepcopy(model.state_dict())


    # === 动态 Loss 曲线绘图初始化 ===
    # 初始化绘图对象
    fig, ax, train_line, val_line, coll_line = init_live_plot()
    train_losses = []
    val_losses = []
    coll_losses = []
    windows = 120 # 窗口大小的控件

    # training:
    for e in range(epochs):
        model.train()
        loss = []
        i = 0
        x_train = x_train[torch.randperm(x_train.size()[0])]

        # 初始化每轮 epoch 的碰撞迭代器
        coll_iter = iter(collloader)
        for i, data in enumerate(trainloader):
            
            optimizer.zero_grad()
            
            y = data['y'].to(device)
            y_gt = data['y_gt'].to(device)
            x = data['x'].to(device)

            # --- 获取下一个碰撞样本 batch（循环使用） ---
            if coll_iter is None:
                coll_iter = iter(collloader)

            try:
                coll_data = next(coll_iter)
            except StopIteration:
                coll_iter = iter(collloader)
                coll_data = next(coll_iter)

            x_coll_batch = coll_data['x'].to(device)
            y_coll_batch = coll_data['y'].to(device)
            
            y_pred = (model.forward(x))
            y_coll_pred = (model.forward(x_coll_batch))
            #print(y_coll_pred)#, y_coll_batch, x_coll_batch)
             #torch.where(y_gt > -0.1, 100.0, 1.0)
            # train_loss = F.binary_cross_entropy_with_logits(y_pred,y) + F.binary_cross_entropy_with_logits(y_coll_pred, y_coll_batch)
            train_loss = F.mse_loss(y_pred, y, reduction='mean') + alpha*F.mse_loss(y_coll_pred, y_coll_batch, reduction='mean')
            # train_loss = F.smooth_l1_loss(y_pred, y) + alpha * F.smooth_l1_loss(y_coll_pred, y_coll_batch)
            # === 损失函数计算 ===
            # main_loss = F.smooth_l1_loss(y_pred, y)
            # coll_loss = boundary_weighted_mse(y_coll_pred, y_coll_batch, beta=beta, boundary=boundary_val)
            # train_loss = main_loss + alpha * coll_loss
            train_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            loss.append(train_loss.item())
            #print(train_loss.item())
            #i += batch_size


        model.eval()
        
        y_pred = model.forward(x_val)
        #y_pred = torch.sigmoid(model.forward(x_val))
        #print(x_val)
        #print(y_pred[0,0])
        val_loss = F.mse_loss(y_pred, y_val, reduction='mean')
        #val_loss = F.binary_cross_entropy_with_logits(y_pred,y_val)
        train_loss = np.mean(loss)
        if(val_loss < min_loss and e>100):
            print('saving model ------------- ', val_loss.item())
            torch.save(
                {
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'norm':{'x':{'mean':mean_x, 'std':std_x},
                            'y':{'mean':mean_y, 'std':std_y}}
                },
                join_path(checkpoints_dir,
                          robot_name+'_self_sdf.pt'))
            min_loss = val_loss

        coll_loss = F.mse_loss(y_coll_pred, y_coll_batch, reduction='mean')  # 单次取样即可代表该 epoch


        # === 打印 & 存储物理单位下的 RMSE ===
        train_rmse, val_rmse, coll_rmse = print_epoch_rmse(
            e, train_loss, val_loss.item(), coll_loss.item(), std_y.item())
        
        # 改为 RMSE
        train_losses.append(train_rmse)
        val_losses.append(val_rmse)
        coll_losses.append(coll_rmse)

        # === 更新 EMA 模型 ===
        update_ema(model, ema_model, ema_decay)

        # === 调整学习率 ===
        scheduler.step(val_loss)

        # === Early Stopping 检查 ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
            print(f"[✓] New best val loss: {val_loss.item():.6f}")
        else:
            patience_counter += 1
            # print(f"[•] Patience {patience_counter}/{patience}")
            if patience_counter >= patience and e > 100:
                print("[⏹️] Early stopping triggered.")
                break


        # === 更新动态图 ===
        update_live_plot(fig, ax, train_line, val_line, coll_line,
                        train_losses, val_losses, coll_losses, window=windows)  # 默认 window=100



    # === 保存最终 loss 曲线图 ===
    plt.ioff()
    fig.savefig('loss_curve_final.png')
    print("[INFO] Final loss curve saved as 'loss_curve_final.png'")
    plt.show()

    model.load_state_dict(best_model_state)  # 恢复 best 模型
    ema_model.eval()  # 可用于部署推理
    # 手动保存 ema 模型权重
    torch.save(
        {
            'model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'norm':{'x':{'mean':mean_x, 'std':std_x},
                    'y':{'mean':mean_y, 'std':std_y}}
        },
        join_path(checkpoints_dir,
                    robot_name+'_self_ema.pt'))
    
    with torch.no_grad():
        # compare model vs ema_model
        pred_model = model(x_val)
        pred_ema = ema_model(x_val)
        loss_model = F.mse_loss(pred_model, y_val)
        loss_ema = F.mse_loss(pred_ema, y_val)
        rmse_model = torch.sqrt(loss_model) * std_y
        rmse_ema = torch.sqrt(loss_ema) * std_y
        print(f"[Compare RMSE] model: {rmse_model:.5f} m | ema: {rmse_ema:.5f} m")


    plot_loss_curves(train_losses, val_losses, coll_losses, save_path='loss_curve.png')

if __name__=='__main__':
    # create_dataset('franka_real_robot_tray')
    # create_dataset('franka_real_robot')
    create_dataset('genie')
