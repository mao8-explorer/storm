

from simple_reacher_env import ArmEnv

import random
import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from zmq import Flag

from sympy import false




# actor net
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound # 连续动作空间

# critic net
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)

        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).detach().cpu()[0]
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim) # W * N(0,1) + mean
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets)) # 如果q_target 通过MPPI 估算会怎样
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

    def state_dict(self):
        return {'actor':self.actor.state_dict(),
                'target_actor':self.target_actor.state_dict(),
                'critic':self.critic.state_dict(),
                'target_critic':self.target_critic.state_dict()}

    def save(self,fp):
        torch.save(self.state_dict(),fp)
    
    def load(self,fp):
        d = torch.load(fp)
        self.actor.load_state_dict(d['actor'])
        self.target_actor.load_state_dict(d['target_actor'])
        self.critic.load_state_dict(d['critic'])
        self.target_critic.load_state_dict(d['target_critic'])


    def evaluate(self,fp):
        self.load(fp)
        env.render()
        s = env.reset()
        step = 0
        while True:
            env.render()
            s = torch.tensor([s], dtype=torch.float).to(self.device)
            a = self.actor(s).detach().cpu()[0]
            s, r, done = env.step(a)
            step += 1
            if step >= 299:
                step = 0
                env.reset()
            print(r,done)
            if done: 
                env.reset()




if __name__ == '__main__':
    TRAIN_MODE = True
    actor_lr = 3e-4 # 这个设置大了 3e-3 tau = 0.01有问题 岔劈大了 
    critic_lr = 3e-3
    num_episodes = 800
    hidden_dim = 512
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.05  # 高斯噪声标准差  take action的时候用到，增加探索程度，跳出policy自身的局限
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # env_name = 'Pendulum-v1'
    # env = gym.make(env_name)
    # random.seed(0)
    # np.random.seed(0)
    # env.seed(0)
    # torch.manual_seed(0)
    # replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # action_bound = env.action_space.high[0]  # 动作最大值

    env = ArmEnv()
    random.seed(0)
    np.random.seed(0)
    # env.seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = env.state_dim
    action_dim = env.action_dim
    action_bound = torch.tensor(0.1,dtype=torch.float32)
    
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
    if TRAIN_MODE:
        # agent.load(fp = 'model.pt')
        return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
        agent.save(fp = 'model_ddpg.pt')
        
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DDPG on {}'.format('simple_reacher'))
        plt.show()

        mv_return = rl_utils.moving_average(return_list, 9)
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DDPG on {}'.format(('simple_reacher')))
        plt.show()
    else:
        agent.evaluate('model_ddpg.pt')