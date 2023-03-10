"""
The script to do deep Q-learnings.

Main reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import Transition, TransitionHigh, TransitionLow
from tqdm import trange
from algos.base import BaseAlgo

class Optimizer():
    def __init__(self, high_policy_network, low_policy_network):
        super().__init__()
        self.optimizer_high = optim.RMSprop(high_policy_network.parameters())
        self.optimizer_low = optim.RMSprop(low_policy_network.parameters())
    
    def load_state_dict(self, optimizer_state):
        self.optimizer_high.load_state_dict(optimizer_state['high'])
        self.optimizer_low.load_state_dict(optimizer_state['low'])
    
    def state_dict(self):
        return {
            'high': optimizer_high.state_dict(),
            'low': optimizer_low.state_dict()
        }
        
class HDQNAlgo(BaseAlgo):
    """
    The class for hierarchical deep Q-learning networks.
    """

    def __init__(self, env, model, high_replay_buffer, low_replay_buffer, sub_goal_space, action_space, internal_steps=10,
                 gamma_high=0.99, gamma_low=0.99, lr_high=1e-4, lr_low=1e-4, epsilon_min_high=0.01, epsilon_max_high=1,
                 epsilon_decay_high=1e4, epsilon_min_low=0.01, epsilon_max_low=1, epsilon_decay_low=1e5,
                 batch_size_high=256, batch_size_low=512, target_network_update_high=10, target_network_update_low=10,
                 burn_in=10000, resize_transformer=None):
        """
        The init function.
        """

        # * environment related
        self.env = env
        self.sub_goal_space = sub_goal_space
        self.sub_goal_space_length = len(sub_goal_space)
        self.action_space = action_space
        self.gamma_high = gamma_high
        self.gamma_low = gamma_low
        self.resize_transformer = resize_transformer

        # * Q-networks related
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.high_policy_network = model.high_policy_network.to(self.device)
        self.high_target_network = model.high_target_network.to(self.device)
        self.low_policy_network = model.low_policy_network.to(self.device)
        self.low_target_network = model.low_target_network.to(self.device)

        self.high_target_network.load_state_dict(self.high_policy_network.state_dict())
        self.high_target_network.eval()

        self.low_target_network.load_state_dict(self.low_policy_network.state_dict())
        self.low_target_network.eval()

        self.target_network_update_high = target_network_update_high
        self.target_network_update_low = target_network_update_low

        # * SGD related
        self.optimizer = Optimizer(self.high_policy_network, self.low_policy_network)

        # * replay buffer
        self.high_replay_buffer = high_replay_buffer
        self.high_batch_size = batch_size_high

        self.low_replay_buffer = low_replay_buffer
        self.low_batch_size = batch_size_low

        # * learning related
        self.steps = 0

        self.lr_high = lr_high
        self.lr_low = lr_low
        self.epsilon_min_high = epsilon_min_high
        self.epsilon_max_high = epsilon_max_high
        self.epsilon_decay_high = epsilon_decay_high
        self.epsilon_min_low = epsilon_min_low
        self.epsilon_max_low = epsilon_max_low
        self.epsilon_decay_low = epsilon_decay_low

        self.burn_in = burn_in

        # * recording related
        self.steps_records = []
        self.rewards_records = []
        self.internal_rewards_records = []

    def compute_epsilon(self, level):
        """
        The function to compute the epsilon given the current steps.
            Usually, high level will decay faster than low level (usually with a larger decay factor).
        :param level: the level to compute epsilon "high" or "low".
        :return: the updated epsilon.
        """

        assert level in ['high', 'low', 'sub-goal'], "The level is not an available value."

        if level == "high":
            # * for the high-level epsilon computation
            return self.epsilon_min_high + (self.epsilon_max_high - self.epsilon_min_high) \
                   * np.exp(-1 * self.steps / self.epsilon_decay_high)
        if level == "low":
            # * for the high-level epsilon computation
            return self.epsilon_min_low + (self.epsilon_max_low - self.epsilon_min_low) \
                   * np.exp(-1 * self.steps / self.epsilon_decay_low)

        # * for the derived class to implement epsilon-greedy based SGD
        if level == "sub-goal":
            # * for the sub-goal epsilon computation
            return self.epsilon_min_sub_goal + (self.epsilon_max_sub_goal - self.epsilon_min_sub_goal) \
                   * np.exp(-1 * self.steps / self.epsilon_decay_sub_goal)

    def step_goal(self, obs):
        """
        The function to select a sub-goal
        """

        epsilon = self.compute_epsilon("high")
        if random.random() < epsilon:
            # select sub-goals randomly
            return torch.tensor([[random.randrange(self.sub_goal_space_length)]], device=self.device, dtype=torch.long)
        else:
            # select sub-goals greedily from the policy network
            with torch.no_grad():
                return self.high_policy_network(obs).max(1)[1].view(1, 1)

    def step_action(self, sub_goal, obs):
        """
        The function to select an action.
        """
        epsilon = self.compute_epsilon("low")
        if random.random() < epsilon:
            # select actions randomly
            return torch.tensor([[random.randrange(self.action_space)]], device=self.device, dtype=torch.long)
        else:
            # select actions greedily from the policy network
            with torch.no_grad():
                return self.low_policy_network(obs)[sub_goal.item()].max(1)[1].view(1, 1)

    def reward_critic(self, sub_goal, done, info):
        """
        The function to return a critic reward for low-level, reward=1 for complete the sub-goal, =0 otherwise.
        """
        if sub_goal == 0:
            # * for sub-goal to pick the key
            if info['key_picked']:
                return 1, True
            else:
                return 0, False
        if sub_goal == 1:
            # * for sub-goal to pass the door
            if info['door_passed']:
                return 1, True
            else:
                return 0, False
        if sub_goal == 2:
            # * for sub-goal to reach the final state
            if done:
                return 1, True
            else:
                return 0, False


    def optimize_high(self):
        """
        The function to optimize the high-level policy network for one step.
        """

        transitions = self.high_replay_buffer.sample(self.high_batch_size)
        samples_batch = TransitionHigh(*zip(*transitions))

        states_batch = torch.cat(samples_batch.state)
        sub_goals_batch = torch.cat(samples_batch.sub_goal)
        rewards_batch = torch.cat(samples_batch.reward_high)
        next_states_batch = torch.cat(samples_batch.next_N_state)
        dones_batch = torch.tensor(samples_batch.done, device=self.device, dtype=torch.bool)

        # * The current Q-values = Q_policy(s,a)
        current_q_values = self.high_policy_network(states_batch).gather(1, sub_goals_batch)

        # * Initialize the TD-targets Y
        td_targets = torch.zeros((self.high_batch_size, 1), device=self.device)
        # + running the Double-DQN approach
        # * select expected sub-goals with highest Q-values from policy network g+=argmax{Q_policy(s_next)}
        expected_sub_goals = self.high_policy_network(next_states_batch).max(1)[1].view(-1, 1)
        # * compute the Q-values for expected sub-goals from target network Q_expected = Q_target(s_next, g+)
        next_state_q_values = self.high_target_network(next_states_batch).gather(1, expected_sub_goals)
        td_targets[~dones_batch] = next_state_q_values[~dones_batch]
        td_targets = rewards_batch.view(-1, 1) + td_targets * self.gamma_high

        # * compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q_values, td_targets)

        # * optimize the model
        self.optimizer.optimizer_high.zero_grad()
        loss.backward()
        for param in self.high_policy_network.parameters():
            # * this is for the gradient clipping
            param.grad.data.clamp_(-1, 1)
        self.optimizer.optimizer_high.step()

    def optimize_low(self):
        """
        The function to optimize the low-level model
        """
        transitions = self.low_replay_buffer.sample(self.low_batch_size)
        samples_batch = TransitionLow(*zip(*transitions))

        states_batch = torch.cat(samples_batch.state)
        sub_goals_batch = torch.cat(samples_batch.sub_goal).view(self.low_batch_size)
        actions_batch = torch.cat(samples_batch.action)
        rewards_batch = torch.cat(samples_batch.reward_low)
        next_states_batch = torch.cat(samples_batch.next_state)
        completes_batch = torch.tensor(samples_batch.complete, device=self.device, dtype=torch.bool)

        index_range = torch.arange(self.low_batch_size, dtype=torch.long)

        # * The current Q-values = Q_policy(s,a)
        current_q_values = self.low_policy_network(states_batch)
        # * index corresponding
        current_q_values = current_q_values[sub_goals_batch, index_range, :].gather(1, actions_batch)

        # * Initialize the TD-targets Y
        td_targets = torch.zeros((self.low_batch_size, 1), device=self.device)
        # + running the Double-DQN approach
        # * select expected actions with highest Q-values from policy network a+=argmax{Q_policy(s_next)}
        expected_actions = self.low_policy_network(next_states_batch)
        expected_actions = expected_actions[sub_goals_batch, index_range, :].max(1)[1].view(-1, 1)
        # * compute the Q-values for expected actions from target network Q_expected = Q_target(s_next, a+)
        next_state_q_values = self.low_target_network(next_states_batch)
        next_state_q_values = next_state_q_values[sub_goals_batch, index_range, :].gather(1, expected_actions)
        td_targets[~completes_batch] = next_state_q_values[~completes_batch]
        td_targets = rewards_batch.view(-1, 1) + td_targets * self.gamma_low

        # * compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q_values, td_targets)

        # * optimize the model
        self.optimizer.optimizer_low.zero_grad()
        loss.backward()
        for param in self.low_policy_network.parameters():
            # * this is for the gradient clipping
            param.grad.data.clamp_(-1, 1)
        self.optimizer.optimizer_low.step()
        
    def obs_numpy_to_tensor(self, obs):
        obs_tensor = torch.from_numpy(np.transpose(obs, (2, 0, 1)))
        return self.resize_transformer(obs_tensor).unsqueeze(0).to(self.device)
    
    def collect_experiences(self, num_internal_steps):
        obs = self.env.reset()[0]
        obs = self.obs_numpy_to_tensor(obs)
        # * to remember the current state for high-level iteration
        obs_high_start = obs

        # * to record the steps for each epoch
        epoch_steps = 0
        epoch_reward = 0
        epoch_internal_reward = 0
        
        while True:
            sub_goal = self.step_goal(obs)

            reward_high = 0

            for _ in range(num_internal_steps):
                # + enter the low-level loop
                action = self.step_action(sub_goal, obs)

                self.steps += 1
                epoch_steps += 1

                obs_new, reward_env, done, terminate, info = self.env.step(action.item())

                epoch_reward += reward_env
                reward_high += reward_env 

                obs_new = self.obs_numpy_to_tensor(obs_new)
                reward_low, complete = self.reward_critic(sub_goal, done, info)
                reward_low = torch.tensor([reward_low], device=self.device)
                epoch_internal_reward += reward_low.item()

                self.low_replay_buffer.append(
                    TransitionLow(obs, sub_goal, action, obs_new, reward_low, complete))

                if len(self.low_replay_buffer) > self.low_batch_size and self.steps > self.burn_in:
                    self.optimize_low()
            
                obs = obs_new

                if done or complete:
                    break


            reward_high = torch.tensor([reward_high], device=self.device)

            self.high_replay_buffer.append(TransitionHigh(obs_high_start, sub_goal, obs, reward_high, done))
            
            if done:
                self.steps_records.append(epoch_steps)
                self.rewards_records.append(epoch_reward)
                self.internal_rewards_records.append(epoch_internal_reward)
                break
            
            obs_high_start = obs
            
        logs = {
            "return_per_episode": epoch_reward,
            "num_steps_per_episode": epoch_steps,
            "num_episodes": 1
        }
        return logs

    def update_parameters(self, epoch): 
        if len(self.high_replay_buffer) > self.high_batch_size and self.steps > self.burn_in:
            print('optimizing high-level controller') 
            self.optimize_high()
            
        if (epoch + 1) % self.target_network_update_high == 0:
            self.high_target_network.load_state_dict(self.high_policy_network.state_dict())

        if (epoch + 1) % self.target_network_update_low == 0:
            self.low_target_network.load_state_dict(self.low_policy_network.state_dict())
            
        return
