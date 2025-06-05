#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_self_collision_model.py

训练 Genesis 机器人自碰撞距离回归神经网络的完整脚本，
集成以下特性：
  - Halton 采样生成关节角样本
  - Isaac Gym + StormMPC 模型前向推理获取真实距离
  - 数据归一化与碰撞样本平衡
  - Residual MLP 或外部预训练模型加载（RobotSelfCollisionNet）
  - 动态绘制训练/验证/碰撞 RMSE 曲线
  - Gradient Clipping、ReduceLROnPlateau 学习率调度、Early Stopping
  - EMA（指数滑动平均）模型缓冲，用于推理阶段更稳定模型
  - 保存最优模型与 EMA 模型权重
  - 在训练结束后对比 Raw model 与 EMA model 在验证集上的 RMSE

用法示例：
    python train_self_collision_model.py --robot genie
"""

import os
import argparse
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 确保交互式绘图有效

from storm_kit.mpc.rollout.arm_base import ArmBase
from storm_kit.util_file import (
    join_path, get_assets_path, get_gym_configs_path,
    get_mpc_configs_path, get_weights_path, load_yaml
)
from storm_kit.mpc.control.control_utils import generate_halton_samples
from storm_kit.geom.nn_model.robot_self_collision import RobotSelfCollisionNet


def init_live_plot():
    """
    初始化动态绘图窗口，返回 figure，axes，以及三条空的折线：
    - train_line: 用于绘制训练集 RMSE
    - val_line: 用于绘制验证集 RMSE
    - coll_line: 用于绘制碰撞样本 RMSE
    """
    plt.ion()  # 打开交互模式
    fig, ax = plt.subplots(figsize=(10, 6))
    train_line, = ax.plot([], [], label='Train RMSE (m)', color='blue', linewidth=2)
    val_line,   = ax.plot([], [], label='Validation RMSE (m)', color='orange', linewidth=2)
    coll_line,  = ax.plot([], [], label='Collision RMSE (m)', color='green', linestyle='--', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('RMSE (meters)', fontsize=12)
    ax.set_title('Live Training Loss Curve', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return fig, ax, train_line, val_line, coll_line


def update_live_plot(fig, ax, train_line, val_line, coll_line,
                     train_losses, val_losses, coll_losses, window=100):
    """
    每个 epoch 调用一次，更新折线数据并重绘：
      - 只显示最近 `window` 个 epoch 的数据
    """
    total_epochs = len(train_losses)
    start_idx = max(0, total_epochs - window)
    epochs_x = list(range(start_idx, total_epochs))

    train_y = train_losses[start_idx:]
    val_y   = val_losses[start_idx:]
    coll_y  = coll_losses[start_idx:]

    train_line.set_data(epochs_x, train_y)
    val_line.set_data(epochs_x, val_y)
    coll_line.set_data(epochs_x, coll_y)

    ax.relim()            # 重新计算坐标轴范围
    ax.autoscale_view()   # 根据新数据自动缩放
    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_loss_curves(train_losses, val_losses, coll_losses=None, save_path=None):
    """
    训练完成后调用，绘制完整的 Train/Val/Coll RMSE 曲线并保存到文件（如果指定了 save_path）。
    """
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train RMSE (m)', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation RMSE (m)', linewidth=2)

    if coll_losses is not None:
        plt.plot(epochs, coll_losses, label='Collision RMSE (m)', linestyle='--', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RMSE (meters)', fontsize=12)
    plt.title('Training Loss Curve in Physical Units (m)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Loss plot saved to: {save_path}")

    plt.tight_layout()


def print_epoch_rmse(e, train_loss, val_loss, coll_loss, std_y):
    """
    打印当前 epoch 的物理单位 RMSE（反归一化后），并返回这些 RMSE：
    - train_loss, val_loss, coll_loss 均为标准化后的 MSE
    - std_y 需要为原始标签标准差，用来反归一化
    """
    train_rmse = math.sqrt(train_loss) * std_y
    val_rmse   = math.sqrt(val_loss) * std_y
    coll_rmse  = math.sqrt(coll_loss) * std_y
    print(f"Epoch {e:03d} | Train RMSE: {train_rmse:.5f} m | "
          f"Val RMSE: {val_rmse:.5f} m | Coll RMSE: {coll_rmse:.5f} m")
    return train_rmse, val_rmse, coll_rmse


def update_ema(model, ema_model, decay):
    """
    EMA（指数滑动平均）更新：
      ema_param = decay * ema_param + (1 - decay) * current_param
    用于在训练过程中累积一份平滑版本的权重，最后用于推理更稳定。
    """
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


class RobotDataset(torch.utils.data.Dataset):
    """
    简单封装的 Dataset，用于放置 (x, y, y_gt) 三元组。
    其中 y 是标准化后的标签，y_gt 是反归一化的真实标签。
    """
    def __init__(self, x, y, y_gt):
        self.x = x
        self.y = y
        self.y_gt = y_gt

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return {'x': self.x[idx, :], 'y': self.y[idx, :], 'y_gt': self.y_gt[idx, :]}


def compare_model_vs_ema(model, ema_model, x_val, y_val, std_y):
    """
    在验证集上对比 Raw 模型 vs EMA 模型的 RMSE，输出反归一化后的结果
    """
    model.eval()
    ema_model.eval()
    with torch.no_grad():
        pred_model = model(x_val)
        pred_ema   = ema_model(x_val)
        loss_model = F.mse_loss(pred_model, y_val)
        loss_ema   = F.mse_loss(pred_ema, y_val)
        rmse_model = torch.sqrt(loss_model) * std_y
        rmse_ema   = torch.sqrt(loss_ema) * std_y

    # 把 Tensor 转为 Python float，才能用 :.5f 格式化
    print(f"[Compare RMSE] Raw model: {rmse_model.item():.5f} m | EMA model: {rmse_ema.item():.5f} m")
    return rmse_model.item(), rmse_ema.item()

def save_model(model, optimizer, mean_x, std_x, mean_y, std_y, path):
    """
    保存模型权重及优化器状态、以及归一化统计量（mean/std）。
    """
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'norm': {
                'x': {'mean': mean_x, 'std': std_x},
                'y': {'mean': mean_y, 'std': std_y}
            }
        },
        path
    )
    print(f"[INFO] Model saved to: {path}")


def train_self_collision(robot_name, args):
    """
    主训练流程：
    1. Halton 采样 + rollout 生成 ground truth
    2. 归一化 & 训练/验证/测试集划分
    3. 加载预训练或新建模型
    4. 训练循环（带 Gradient Clipping、EMA、ReduceLROnPlateau、EarlyStopping）
    5. 保存最优 Raw 模型和 EMA 模型
    6. 在验证集上对比二者 RMSE
    """
    # 1. 基础目录与参数
    checkpoints_dir = os.path.join(get_weights_path(), 'robot_self', 'test')
    os.makedirs(checkpoints_dir, exist_ok=True)
    num_particles = 10000
    task_file = f"{robot_name}_reacher.yml"
    device = torch.device('cuda', 0)
    tensor_args = {'device': device, 'dtype': torch.float32}

    # 2. 加载 MPC 配置
    mpc_yml_file = join_path(get_mpc_configs_path(), task_file)
    with open(mpc_yml_file) as f:
        exp_params = yaml.load(f, Loader=yaml.FullLoader)
    exp_params['robot_params'] = exp_params['model']
    exp_params['cost']['primitive_collision']['weight'] = 0.0
    exp_params['cost']['robot_self_collision']['weight'] = 1.0
    exp_params['control_space'] = 'pos'
    exp_params['mppi']['horizon'] = 2
    exp_params['mppi']['num_particles'] = num_particles

    # 3. 构造 ArmBase，用于生成自碰撞 ground truth
    rollout_fn = ArmBase(exp_params, tensor_args, world_params=None)
    dof = rollout_fn.dynamics_model.d_action

    # 4. Halton 采样关节角度
    q_samples = generate_halton_samples(
        num_particles * 2, dof, use_ghalton=True, seed_val=123,
        device=tensor_args['device'], float_dtype=tensor_args['dtype']
    )

    # 5. 将 Halton 采样映射到关节限位 [low_bounds, up_bounds]
    up_bounds = rollout_fn.dynamics_model.state_upper_bounds[:dof]
    low_bounds = rollout_fn.dynamics_model.state_lower_bounds[:dof]
    offset = 0.0  # 若要人为扩大关节范围，可修改此值
    low_bounds = low_bounds - offset
    up_bounds = up_bounds + offset
    range_b = up_bounds - low_bounds
    q_samples = q_samples * range_b + low_bounds  # 映射到真是范围
    q_samples = q_samples.view(num_particles, 2, dof)

    # 6. 前向推理得到自碰撞距离 ground truth
    start_state = torch.zeros((rollout_fn.dynamics_model.d_state), **tensor_args)
    state_dict = rollout_fn.dynamics_model.rollout_open_loop(start_state.unsqueeze(0), q_samples)
    link_pos_seq = state_dict['link_pos_seq']
    link_rot_seq = state_dict['link_rot_seq']
    cost_fn = rollout_fn.robot_self_collision_cost.distance
    dist = cost_fn(link_pos_seq, link_rot_seq)  # (num_particles*2, 1)

    # 7. 准备特征 x 和标签 y
    x = q_samples.view(num_particles * 2, dof)    # (N, dof)
    y = dist.view(num_particles * 2, 1)            # (N, 1)
    print(f"[Data] Distance range: min={y.min().item():.4f}, max={y.max().item():.4f}")

    # 8. 划分训练/验证/测试集，并归一化
    n_size = x.shape[0]
    # 8.1 先计算归一化要用的统计量
    mean_x = torch.mean(x, dim=0)
    std_x  = torch.std(x, dim=0) + 1e-6
    mean_y = torch.mean(y, dim=0)
    std_y  = torch.std(y, dim=0) + 1e-6

    # 8.2 训练集：前 70%
    split_train = int(0.7 * n_size)
    x_train = x[:split_train]
    y_train = y[:split_train]

    # 8.3 从训练集中提取碰撞样本，用于加权损失
    coll_thresh = 0.005
    mask_coll_train = (y_train[:, 0] > coll_thresh)
    x_coll = x_train[mask_coll_train]
    y_coll = y_train[mask_coll_train]
    collision_ratio = mask_coll_train.float().mean().item()
    print(f"[Data] Collision ratio in train set: {collision_ratio:.2%}")

    # 8.4 验证集：从全体数据里抽取 10% 碰撞 + 20% 非碰撞
    mask_coll_all = (y[:, 0] > coll_thresh)
    mask_free_all = ~mask_coll_all
    x_coll_val = x[mask_coll_all][: int(0.2 * mask_coll_all.sum())]
    y_coll_val = y[mask_coll_all][: int(0.2 * mask_coll_all.sum())]
    x_free_val = x[mask_free_all][: int(0.1 * mask_free_all.sum())]
    y_free_val = y[mask_free_all][: int(0.1 * mask_free_all.sum())]
    x_val = torch.cat([x_coll_val, x_free_val], dim=0)
    y_val = torch.cat([y_coll_val, y_free_val], dim=0)

    # 8.5 测试集：最后 10%
    x_test = x[int(0.9 * n_size):]
    y_test = y[int(0.9 * n_size):]

    # 8.6 定义归一化函数
    def normalize(tensor, mean, std):
        return (tensor - mean) / std

    # 8.7 对各子集进行归一化
    x_train = normalize(x_train, mean_x, std_x)
    x_coll  = normalize(x_coll, mean_x, std_x)
    x_val   = normalize(x_val, mean_x, std_x)
    x_test  = normalize(x_test, mean_x, std_x)
    y_train_norm = normalize(y_train, mean_y, std_y)
    y_coll_norm  = normalize(y_coll, mean_y, std_y)
    y_val_norm   = normalize(y_val, mean_y, std_y)
    y_test_norm  = normalize(y_test, mean_y, std_y)

    # 9. 构建 DataLoader
    train_dataset = RobotDataset(x_train.clone(), y_train_norm.clone(), y_train.clone())
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=96, shuffle=True, drop_last=True
    )
    coll_dataset = RobotDataset(x_coll.clone(), y_coll_norm.clone(), y_coll.clone())
    collloader = torch.utils.data.DataLoader(
        coll_dataset, batch_size=128, shuffle=True, drop_last=True
    )

    # 10. 加载或初始化模型
    enhance_model = rollout_fn.robot_self_collision_cost.coll.robot_nn
    model = enhance_model.model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ----------------- Scheduler 的使用 -----------------
    # ReduceLROnPlateau: 当验证 loss 停滞超过 patience_lr 个 epoch 时，
    # 自动将学习率乘以 factor (默认为 0.5)；min_lr 为学习率下限。
    # 这里我们把 scheduler.step() 设置在每次 epoch 验证后，传入当前 val_loss。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-5
    )
    # ------------------------------------------------------

    # 11. EMA 模型（指数滑动平均），推理时更稳定
    ema_model = deepcopy(model).to(device)
    ema_decay = 0.995

    # 12. 训练超参数
    epochs = args.epochs
    alpha = args.collision_weight  # 碰撞损失权重
    best_val_loss = float('inf')
    best_model_state = deepcopy(model.state_dict())
    patience_counter = 0
    patience = args.patience

    # 13. 动态绘图初始化
    fig, ax, train_line, val_line, coll_line = init_live_plot()
    train_losses, val_losses, coll_losses = [], [], []

    # 14. 训练循环
    for e in range(epochs):
        model.train()
        train_loss_epoch = []
        # 每个 epoch 新建一个碰撞迭代器
        coll_iter = iter(collloader)

        for batch in trainloader:
            optimizer.zero_grad()
                
            # 1）普通样本部分
            x_batch = batch['x'].to(device)
            y_batch = batch['y'].to(device)
            y_pred = model(x_batch)


            # 2）碰撞样本部分
            try:
                coll_data = next(coll_iter)
            except StopIteration:
                coll_iter = iter(collloader)
                coll_data = next(coll_iter)

            x_coll_batch = coll_data['x'].to(device)
            y_coll_batch = coll_data['y'].to(device)
            y_coll_pred = model(x_coll_batch)

            # 3）合并 Loss
            train_loss = F.mse_loss(y_pred, y_batch) + \
                        alpha * F.mse_loss(y_coll_pred, y_coll_batch)
            train_loss.backward()


            # 梯度裁剪：防止梯度爆炸，特别在深度残差网络中很重要。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_epoch.append(train_loss.item())

            # 更新 EMA：指数滑动平均，对抗模型参数突然跳变
            update_ema(model, ema_model, ema_decay)

        # 每个 epoch 结束后，切换到 eval 模式计算验证误差
        model.eval()
        with torch.no_grad():
            # （1）验证集误差
            val_pred = model(x_val.to(device))
            val_loss = F.mse_loss(val_pred, y_val_norm.to(device))
            # （2）碰撞误差：直接复用最后一个 y_coll_pred, y_coll_batch
            coll_loss = F.mse_loss(y_coll_pred, y_coll_batch)

        # 反归一化并打印 RMSE
        train_loss_mean = np.mean(train_loss_epoch)
        train_rmse, val_rmse, coll_rmse = print_epoch_rmse(
            e, train_loss_mean, val_loss.item(), coll_loss.item(), std_y.item()
        )
        train_losses.append(train_rmse)
        val_losses.append(val_rmse)
        coll_losses.append(coll_rmse)

        # 动态绘图更新
        update_live_plot(fig, ax, train_line, val_line, coll_line,
                         train_losses, val_losses, coll_losses, window=args.plot_window)

        # ----------------- 调用 scheduler.step -----------------
        # 这里把当前验证 loss 传给 ReduceLROnPlateau，若 val_loss 连续多次未下降，则自动减小学习率。
        scheduler.step(val_loss)
        # ------------------------------------------------------

        # Early Stopping：若验证 loss 长时间不下降且超过最小训练轮次，则停止训练
        if e > 100 and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            print(f"[✓] New best val loss: {val_loss.item():.6f}")
            raw_path = join_path(checkpoints_dir, f"{robot_name}_self_sdf.pt")
            save_model(model, optimizer, mean_x, std_x, mean_y, std_y, raw_path)
        else:
            patience_counter += 1
            # print(f"[•] Patience {patience_counter}/{patience}")
            if patience_counter >= patience and e > args.min_epochs:
                print("[⏹️] Early stopping triggered.")
                break

    # 15. 训练结束后，恢复并保存最优模型
    model.load_state_dict(best_model_state)
    ema_model.eval()

    raw_path = join_path(checkpoints_dir, f"{robot_name}_self_sdf.pt")
    save_model(model, optimizer, mean_x, std_x, mean_y, std_y, raw_path)

    ema_path = join_path(checkpoints_dir, f"{robot_name}_self_ema.pt")
    save_model(ema_model, optimizer, mean_x, std_x, mean_y, std_y, ema_path)

    # 16. 在验证集上对比 Raw vs EMA
    compare_model_vs_ema(model, ema_model,
                        x_val.to(device), y_val_norm.to(device), std_y.to(device))

    # 17. 保存并显示最终 loss 曲线
    plt.ioff()
    plot_loss_curves(train_losses, val_losses, coll_losses, save_path='loss_curve.png')
    print("[INFO] Training complete. Loss curves saved to 'loss_curve.png'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train self-collision distance model for a robot")
    parser.add_argument("--robot", type=str, default="genie", help="Robot name (e.g. genie)")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for Adam optimizer")
    parser.add_argument("--collision_weight", type=float, default=2.0, help="Weight for collision sample loss")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--min_epochs", type=int, default=100, help="Minimum epochs before early stopping allowed")
    parser.add_argument("--plot_window", type=int, default=120, help="Window size for live plot")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_self_collision(args.robot, args)
