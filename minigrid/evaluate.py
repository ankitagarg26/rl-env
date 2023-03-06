import argparse
import numpy as np
import gymnasium as gym
import envs.envs
import torch
import os

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from envs.wrappers import MovetoFiveDirectionsWrapper
from models.feudalNet import FeudalNet
from algos import feudalNetworkAlgo

parser = argparse.ArgumentParser()

parser.add_argument("--num-episodes", type=int, default=1000, help="Number of episodes")
parser.add_argument("--max-steps", type=int, default=1000, help="Number of maximum steps")
parser.add_argument("--algorithm", default='feudalNet', help="Algorithm for training agent")
parser.add_argument("--model-name", required=True, help="Model file name")

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    env_name = 'MiniGrid-Exp-V2-10x10'

    env = gym.make(env_name)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = MovetoFiveDirectionsWrapper(env)
    
    observation_space = env.reset()[0].shape
    action_space = 5 # 0:left, 1:right, 2:forward, 3:pickup the key, 4:open the door
    
    if args.algorithm == 'feudalNet':
    
        d = 256 # perception output dimension
        k = 16  # dimensionality of embedding vectors w 
        c = 4   # horizon defining temporal resolution of the manager

        model = FeudalNet(observation_space, d, k, c, action_space)
        
        if os.path.exists('./Results/' + args.model_name):
            model.load_state_dict(torch.load('./Results/' + args.model_name))
        else:
            raise Exception("Model file does not exist")

        algo = feudalNetworkAlgo(model, env, c)
    else:
        raise Exception("Algorithm not supported")
    
    results = []
    steps = []
    for i in range(0, args.num_episodes):
        reward, step_count = algo.evaluate(max_steps = 1000)
        results.append(reward)
        steps.append(step_count)
        
    np.save('Results/' + args.model_name + '_test_results.npy', results)
    np.save('Results/' + args.model_name + '_test_steps.npy', results)
            
    