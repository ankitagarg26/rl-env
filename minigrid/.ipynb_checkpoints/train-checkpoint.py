import gymnasium as gym
import envs.envs
import random 
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch 

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from envs.wrappers import MovetoFiveDirectionsWrapper
from models.feudalNet import FeudalNet
from algos import feudalNetworkAlgo


parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
parser.add_argument("--gamma-worker", type=float, default=0.99, help="Discount factor for worker")
parser.add_argument("--gamma-manager", type=float, default=0.99, help="Discount factor for manager")
parser.add_argument("--learning-rate", type=float, default=0.0003, help="Learning rate")
parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Maximum norm of gradient")
parser.add_argument("--alpha", type=float, default=0.8, help="Hyperparamter to regulate the influence of intrinsic reward")
parser.add_argument("--num-internal-steps", type=int, default=400, help="Maximum no of environment steps for training worker")
parser.add_argument("--algorithm", default='feudalNet', help="Algorithm for training agent")
parser.add_argument("--model-name", default=None, help="Model file name")

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
        
        if args.model_name == None:
            model = FeudalNet(observation_space, d, k, c, action_space)
        else:
            if os.path.exists('./Results/' + args.model_name):
                model.load_state_dict(torch.load('./Results/' + args.model_name))
            else:
                raise Exception("Model file does not exist")
                
        algo = feudalNetworkAlgo(model, env, c, args.gamma_manager, args.gamma_worker, 
                                 args.max_grad_norm, args.alpha, args.learning_rate)
    else:
        raise Exception("Algorithm not supported")
    
    for epoch in range(args.epochs):
        algo.collect_experiences(args.num_internal_steps)
        algo.update_parameters(epoch)
    
    #save results
    torch.save(model.state_dict(), 'Results/' + args.algorithm)
    np.save('Results/' + args.algorithm + '_external_rewards.npy', algo.rewards_record)
    np.save('Results/' + args.algorithm + '_steps.npy', algo.steps_record)