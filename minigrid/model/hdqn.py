"""
The script for deep neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import HRLModel


class SimpleNeuralNet(nn.Module):
    """
    A very simple network with three convolutional layers with activate layers.
    """

    def __init__(self, width, height, outputs):
        super(SimpleNeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - kernel_size) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = conv_height * conv_width * 32

        self.head = nn.Linear(linear_input_size, outputs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return self.head(x.view(x.size(0), -1))


class ThreeHeadsSimpleNeuralNet(nn.Module):
    """
    A very simple network with THREE heads for three sub-goals respectively.
    """

    def __init__(self, width, height, outputs):
        super(ThreeHeadsSimpleNeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - kernel_size) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = conv_height * conv_width * 32

        self.head1 = nn.Linear(linear_input_size, outputs)
        self.head2 = nn.Linear(linear_input_size, outputs)
        self.head3 = nn.Linear(linear_input_size, outputs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        features = F.relu(self.bn3(self.conv3(x)))

        output1 = self.head1(features.view(features.size(0), -1))
        output2 = self.head2(features.view(features.size(0), -1))
        output3 = self.head3(features.view(features.size(0), -1))

        output = torch.stack((output1, output2, output3))

        return output

class HDQN(HRLModel):
    
    def __init__(self, observation_shape, num_sub_goals, num_outputs):
        super().__init__(observation_shape, num_outputs)
        
        self.high_policy_network = SimpleNeuralNet(40, 40, num_sub_goals)
        self.high_target_network = SimpleNeuralNet(40, 40, num_sub_goals)

        self.low_policy_network = ThreeHeadsSimpleNeuralNet(40, 40, num_outputs)
        self.low_target_network = ThreeHeadsSimpleNeuralNet(40, 40, num_outputs)
        
    def load_state_dict(self, model_state):
        self.high_policy_network.load_state_dict(model_state['high'])
        self.low_policy_network.load_state_dict(model_state['low'])
        
    def state_dict(self):
        return {
            'high': high_policy_network.state_dict(),
            'low': low_policy_network.state_dict()
        }
    
    def forward(self, obs):
        
        g = self.high_policy_network(obs).max(1)[1].view(1, 1)
        
        prob = self.low_policy_network(obs)[g.item()]
        
        return prob
        