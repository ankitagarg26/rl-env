import random
import numpy
import torch
import collections
import envs.envs

import os
import torch
import logging
import sys
import csv
import gymnasium as gym

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"

def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)

def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")

def get_status(model_dir):
    path = get_status_path(model_dir)
    return torch.load(path, map_location=device)

def save_status(status, model_dir):
    path = get_status_path(model_dir)
    create_folders_if_necessary(path)
    torch.save(status, path)

def get_vocab(model_dir):
    return get_status(model_dir)["vocab"]

def get_model_state(model_dir):
    return get_status(model_dir)["model_state"]

def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()

def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env.reset(seed=seed)
    return env

def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d


"""
utility functions for hDQN
"""
import torchvision.transforms as T

from collections import namedtuple, deque
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from envs.wrappers import SubGoalsIndicator

Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward', 'done'))

TransitionHigh = namedtuple("Transition", ('state', 'sub_goal', 'next_N_state', 'reward_high', 'done'))
TransitionLow = namedtuple("Transition", ('state', 'sub_goal', 'action', 'next_state', 'reward_low', 'complete'))

resize_transform = T.Compose([T.ToPILImage(),
                              T.Resize(40, interpolation=InterpolationMode.BICUBIC),
                              T.ToTensor()])


class ReplayBuffer:
    """
    The class for replay buffer.
    """

    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def env_with_subgoals(env):
    return SubGoalsIndicator(env)