import gymnasium as gym
import envs.envs
import random 
import matplotlib.pyplot as plt

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from envs.wrappers import MovetoFiveDirectionsWrapper

env_name = 'MiniGrid-Exp-V2-10x10'

env = gym.make(env_name)
env = RGBImgObsWrapper(env)
env = ImgObsWrapper(env)
env = MovetoFiveDirectionsWrapper(env)

max_steps = 1000
episodes = 1

for ep in range(episodes):
    steps = 0
    total_reward = 0
    obs = env.reset()
    # plt.imshow(obs)
    print(obs[0].shape)
    while True:
        action = random.randint(0,3)
        obs_new, reward, done, terminate, info = env.step(action)
        total_reward = total_reward + reward

        if done == True or steps >= max_steps:
            break

        steps = steps + 1
    
    print('Episode: ', ep, ' Total Reward: ', total_reward)

