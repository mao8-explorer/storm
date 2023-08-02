from fastapi import FastAPI
from simple_reacher_env import ArmEnv

import random
from re import S
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils
from zmq import Flag

# critic and policy(actor)

# policy 是 根据当前状态 输出一个高斯分布的均值和标准差表示动作的分布
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self,state_dim, hidden_dim,action_dim,action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)  # 添加额外的线性层
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        self.min_std = torch.exp(torch.tensor(-20.0))  # 设置最小的标准差
        self.max_std = torch.exp(torch.tensor(2.0))  # 设置最大的标准差
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # 经过额外的线性层和激活函数
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))

        std = torch.clamp(std, self.min_std, self.max_std)

        dist = Normal(mu,std)
        normal_sample = dist.rsample() # rsample() 是重参数化采样
        log_prob= dist.log_prob(normal_sample) # normal_sample 在正态分布中的对数概率
        action = torch.tanh(normal_sample)  # 映射到[-1,1]的范围内 （对网络生成的伪action映射到可用的数值空间）
        log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2)+1e-7)  
        action = action *self.action_bound
        return action , log_prob # 返回的具体的action 和 该action 相应的对数概率
    

    def mean_test(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # 经过额外的线性层和激活函数
        mu = self.fc_mu(x)
        pi_action = torch.tanh(mu)
        pi_action = pi_action * self.action_bound

        return pi_action
    
# critic
class QvalueNetContinuous(torch.nn.Module):
    def __init__(self,state_dim, hidden_dim,action_dim):
        super(QvalueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim , hidden_dim )
        self.fc_out = torch.nn.Linear(hidden_dim,1)

    def forward(self,x,a):
        cat = torch.cat([x,a],dim = 1)
        x =  F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    

#SAC
"""
两个critic Q_w1, Q_w2是 actor的训练更稳定 -> 两个target_critic
一个policy(actor)
"""

class SACContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim,action_bound,
                 actor_lr,critic_lr,alpha_lr,target_entropy,tau,gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)
        self.critic_1 = QvalueNetContinuous(state_dim, hidden_dim,action_dim).to(device)
        self.critic_2 = QvalueNetContinuous(state_dim, hidden_dim,action_dim).to(device)

        self.target_critic_1 = QvalueNetContinuous(state_dim, hidden_dim,action_dim).to(device)
        self.target_critic_2 = QvalueNetContinuous(state_dim, hidden_dim,action_dim).to(device)

        #令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr = critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr = critic_lr)

        # 使用alpha的log值，可用使得训练效果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01),dtype = torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = alpha_lr)

        self.target_entropy = target_entropy # 目标熵大小
        self.gamma = gamma 
        self.tau = tau 
        self.device = device
    
    def take_action(self,state):
        state = torch.tensor([state],dtype=torch.float).to(self.device)
        # action = self.actor(state)[0]
        # return [action.item()]
        action,_ = self.actor(state)
        action = action.detach().cpu()[0]
        return action
    
    def test_action(self,state):
        state = torch.tensor([state],dtype=torch.float).to(self.device)
        # action = self.actor(state)[0]
        # return [action.item()]
        # action,_ = self.actor(state)
        action = self.actor.mean_test(state)
        action = action.detach().cpu()[0]
        return action
    
    
    def calc_target(self,rewards,next_states,done): #计算目标Q值
        with torch.no_grad():
            next_actions, log_prob = self.actor(next_states)
            entropy = -log_prob
            q1_values = self.target_critic_1(next_states,next_actions)
            q2_values = self.target_critic_2(next_states,next_actions)
            next_value = torch.min(q1_values,q2_values) + self.log_alpha.exp()*entropy

            td_target = rewards + self.gamma*next_value*(1-done)
        return td_target
    
    def soft_update(self,net,target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data*(1-self.tau) + param.data * self.tau)
    
    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # rewards = (rewards + 8.0) /8.0 重塑倒立摆的reward

        #更新两个Q网络
        td_target = self.calc_target(rewards,next_states,dones)

        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states,actions),td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states,actions),td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward() #求导并将梯度值保存在相应的参数的 .grad 属性中
        self.critic_1_optimizer.step() # 优化器会根据保存在参数的 .grad 属性中的梯度值，以及优化算法的策略（如梯度下降），对参数进行相应的更新
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()


        # 更新策略网络 没有freeze Q-network

        new_actions , log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states,new_actions)
        q2_value = self.critic_2(states,new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp()*entropy - torch.min(q1_value,q2_value))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1,self.target_critic_1)
        self.soft_update(self.critic_2,self.target_critic_2)

    
    def state_dict(self):
        return {'actor':self.actor.state_dict(),
                'critic_1':self.critic_1.state_dict(),
                'critic_2':self.critic_2.state_dict()}

    def save(self,fp):
        torch.save(self.state_dict(),fp)
    
    def load(self,fp):
        d = torch.load(fp)
        self.actor.load_state_dict(d['actor'])
        self.critic_1.load_state_dict(d['critic_1'])
        self.critic_2.load_state_dict(d['critic_2'])

    def evaluate(self,fp):
        self.load(fp)
        env.render()
        s = env.reset()
        step = 0
        while True:
            env.render()
            a = self.take_action(s)
            s, r, done = env.step(a)
            step += 1
            if step >= 299:
                step = 0
                env.reset()
            # print(s,r,done)
            if done: 
                env.reset()


if __name__ == '__main__':

    TRAIN_MODE = False
    env = ArmEnv()

    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 4000
    hidden_dim = 256
    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 5000
    batch_size = 200
    target_entropy = -env.action_dim
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    random.seed(1234)
    np.random.seed(1234)
    # env.seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = env.state_dim
    action_dim = env.action_dim
    action_bound = torch.tensor(0.1,dtype=torch.float32)
    
    agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device)
    if TRAIN_MODE:
        agent.load(fp = 'model_sac_vel.pt')
        try:
            return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
            agent.save(fp = 'model_sac_vel.pt')
            episodes_list = list(range(len(return_list)))
            plt.plot(episodes_list, return_list)
            plt.xlabel('Episodes')
            plt.ylabel('Returns')
            plt.title('SAC on {}'.format('simple_reacher'))
            plt.show()

            mv_return = rl_utils.moving_average(return_list, 9)
            plt.plot(episodes_list, mv_return)
            plt.xlabel('Episodes')
            plt.ylabel('Returns')
            plt.title('SAC on {}'.format(('simple_reacher')))
            plt.show()
        except KeyboardInterrupt:
            print('Terminating...')
            agent.save(fp='model_sac_vel.pt')

    else:
        agent.evaluate('model_sac_vel.pt')