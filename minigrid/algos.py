import random
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class feudalNetworkAlgo():
    
    def __init__(self, model, env, c, gamma_manager = 0.999, gamma_worker = 0.99, max_grad_norm = 0.5, alpha = 0.8, learning_rate = 0.0003):
        self.env = env
        self.feudal_net = model
        self.gamma_manager = gamma_manager
        self.gamma_worker = gamma_worker
        self.max_grad_norm = max_grad_norm
        self.c = c
        self.alpha = alpha
        self.learning_rate = learning_rate
        
        self.entropies = []
        self.rewards_record = []
        self.steps_record = []
        # self.manager_loss_record = []
        # self.worker_loss_record = []
        
        self.cosine_embedding_criterion = nn.CosineEmbeddingLoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.optimizer = optim.Adam(self.feudal_net.parameters(), lr=learning_rate)
               
    def collect_experiences(self, num_internal_steps):
        
        obs = self.env.reset()[0]
        
        self.log_probs = []
        self.value_manager_list = []
        self.value_worker_list = []
        self.external_rewards = []
        self.g_list = []
        self.s_list = []
                
        state_M, state_W = self.feudal_net.init_state() #initilializing states
        
        for step in range(0, num_internal_steps):
            # reshaping observation for model input
            obs = obs.transpose(2,0,1)

            obs = torch.from_numpy(obs).unsqueeze(0).to(torch.float32)

            value_manager, g, s, states_M, value_worker, probs, states_W = self.feudal_net(obs, state_M, state_W, self.g_list, self.c)

            self.s_list.append(s)

            m = Categorical(probs=probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            dist = probs.detach().numpy() 
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            obs_new, reward_env, done, terminate, info = self.env.step(action)

            self.entropies.append(entropy)
            self.external_rewards.append(reward_env)
            self.log_probs.append(log_prob)
            self.value_manager_list.append(value_manager)
            self.value_worker_list.append(value_worker)

            if done:
                break

            obs = obs_new
        
        self.steps = step
        
    def update_parameters(self, epoch):
        
        manager_update = 0
        worker_update = 0
        R_manager = 0
        R_worker = 0
        policy_update = 0
        worker_value_loss = 0
        # intrinsic_rewards = []
        
        for i in reversed(range(len(self.external_rewards))):
            R_manager = self.gamma_manager * R_manager + self.external_rewards[i]
            manager_adv = R_manager - self.value_manager_list[i]

            embedding_similarity = 1
            if i + self.c < len(self.external_rewards):
                embedding_similarity = self.cosine_embedding_criterion(self.s_list[i+self.c] - self.s_list[i], self.g_list[i], - torch.ones(self.g_list[i].size(0)))

            manager_update = manager_update - (manager_adv  * embedding_similarity)

            intrinsic_reward = 0
            if i-self.c >= 0:
                for j in range(1, self.c+1):
                    intrinsic_reward += self.cosine_similarity(self.s_list[i] - self.s_list[i-j], self.g_list[i-j])

            intrinsic_reward = intrinsic_reward/self.c
            # intrinsic_rewards.append(intrinsic_reward)

            R_worker = self.gamma_worker * R_worker + self.external_rewards[i] + self.alpha * intrinsic_reward
            worker_adv = R_worker - self.value_worker_list[i]
            worker_update = worker_update - (worker_adv * self.log_probs[i])

        worker_value_loss = 0.5 * worker_update.pow(2).mean()
        manager_value_loss = 0.5 * manager_update.pow(2).mean()
        entropy_loss = np.array(self.entropies).mean()

        policy_update = manager_update + worker_update + worker_value_loss + manager_value_loss + 0.001 * entropy_loss
        print('epoch: ', epoch, 'step: ', self.steps,' manager loss: ', round(manager_update.item(), 4), ' worker loss: ', round(worker_update.item(), 4), ' external_reward: ', round(sum(self.external_rewards), 4))

        self.rewards_record.append(sum(self.external_rewards))
        self.steps_record.append(self.steps)
        # manager_loss_record.append(manager_update.item())
        # worker_loss_record.append(worker_update.item())
        # intrinsic_reward_record.append(sum(intrinsic_rewards).item())

        self.optimizer.zero_grad()
        policy_update.backward()

        torch.nn.utils.clip_grad_norm_(self.feudal_net.parameters(), self.max_grad_norm)

        self.optimizer.step()
        
    def evaluate(self, max_steps):
        
        obs = self.env.reset()[0]
        state_M, state_W = self.feudal_net.init_state() #initilializing states
        
        self.g_list = []
        
        reward_high = 0
        steps = 0
        while True:

            obs = obs.transpose(2,0,1)
            obs = torch.from_numpy(obs).unsqueeze(0).to(torch.float32)

            value_manager, g, s, states_M, value_worker, probs, states_W = self.feudal_net(obs, state_M, state_W, self.g_list, self.c)

            m = Categorical(probs=probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            obs_new, reward_env, done, terminate, info = self.env.step(action)

            reward_high += reward_env
            steps += 1

            if done or steps >= max_steps:
                break

            obs = obs_new
        
        return reward_high, steps