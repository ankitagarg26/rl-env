from abc import ABC, abstractmethod, abstractproperty
import torch.nn as nn
import torch.nn.functional as F

class HRLModel(ABC):

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs):
        pass