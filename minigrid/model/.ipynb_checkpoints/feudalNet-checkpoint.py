import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import HRLModel

class Perception(nn.Module):
    
    def __init__(self, observation_shape, d):
        super(Perception, self).__init__()

        height, width, channels = observation_shape

        percept_linear_in = 32 * int((int((height - 4) / 4) - 2) / 2) * int((int((width - 4) / 4) - 2) / 2)
        self.f_percept = nn.Sequential(
            nn.Conv2d(channels, 16, (8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(percept_linear_in, d),
            nn.ReLU()
        )

    def forward(self, x):
        return self.f_percept(x)
    
class Worker(nn.Module):
    def __init__(self, num_outputs, d, k):
        super(Worker, self).__init__()

        self.k = k
        self.num_outputs = num_outputs
        self.phi = nn.Sequential(
            nn.Linear(d, k, bias=False)
        )
        self.f_Wrnn = nn.LSTMCell(d, num_outputs * k)

        self.value_function = nn.Linear(num_outputs * k, 1)

    def reset_states_grad(self, states):
        h, c = states
        return h.detach(), c.detach()
    
    def init_state(self, batch_size=1):
        return (
            torch.rand(batch_size, self.f_Wrnn.hidden_size, requires_grad=self.training),
            torch.rand(batch_size, self.f_Wrnn.hidden_size, requires_grad=self.training)
        )

    def forward(self, z, sum_g_W, states_W, reset_value_grad):
        w = self.phi(sum_g_W)  # w : [1 x k] 
        
        U_flat, c_x = states_W = self.f_Wrnn(z, states_W)
        U = U_flat.reshape((self.k, self.num_outputs))

        a = (w @ U)  # [1 x a]

        probs = F.softmax(a, dim=1)
        
        value = self.value_function(U_flat)

        return value, probs, states_W
    
class dLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, r):
        super(dLSTM, self).__init__()
        
        self.lstm = nn.LSTMCell(input_size, hidden_size * r)
        self.tick = 0
        self.r = r
        self.hidden_size = hidden_size
        
    def init_state(self, batch_size=1):
        self.tick = 0
        return (torch.rand(batch_size, self.lstm.hidden_size, requires_grad=self.training),
                torch.rand(batch_size, self.lstm.hidden_size, requires_grad=self.training))
        
    def forward(self, x, states_M, batch_size=1):
        mask = torch.zeros(batch_size, self.lstm.hidden_size)
        left = (self.tick % self.r) * self.hidden_size
        right = ((self.tick % self.r) * self.hidden_size) + self.hidden_size
        mask[:, left : right] = 1
        
        h_t = states_M[0]
        c_t = states_M[1]
            
        out, states = self.lstm(x, (h_t * mask, c_t * mask))
        
        self.tick = self.tick + 1
        
        return out[:, left: right], states
    
class Manager_dLSTM(nn.Module):
    def __init__(self, d, r):
        super(Manager_dLSTM, self).__init__()
        
        self.f_Mspace = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU()
        )

        self.f_Mrnn = dLSTM(d, d, r)

        self.value_function = nn.Linear(d, 1)

    def forward(self, z, states_M, reset_value_grad):
        s = self.f_Mspace(z)
        g_hat, states_M = self.f_Mrnn(s, states_M)

        g = F.normalize(g_hat)

        value = self.value_function(g_hat)

        return value, g, s, states_M

    def init_state(self, batch_size=1):
        return self.f_Mrnn.init_state(batch_size)

class Manager(nn.Module):
    def __init__(self, d):
        super(Manager, self).__init__()
        
        self.f_Mspace = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU()
        )

        self.f_Mrnn = nn.LSTMCell(d, d)

        self.value_function = nn.Linear(d, 1)

    def forward(self, z, states_M, reset_value_grad):
        s = self.f_Mspace(z)
        g_hat, states_M = self.f_Mrnn(s, states_M)

        g = F.normalize(g_hat)

        value = self.value_function(g_hat)

        return value, g, s, states_M

    def init_state(self, batch_size=1):
        return (
            torch.rand(batch_size, self.f_Mrnn.hidden_size, requires_grad=self.training),
            torch.rand(batch_size, self.f_Mrnn.hidden_size, requires_grad=self.training)
        )
    
class FeudalNet(nn.Module):
    
    def __init__(self, observation_shape, num_outputs, c, d=256, k=16):
        super().__init__()
        
        self.percept = Perception(observation_shape, d)
        self.worker = Worker(num_outputs, d, k)
        self.manager = Manager_dLSTM(d, c)
    
    def init_state(self):
        return self.manager.init_state(), self.worker.init_state()

    def forward(self, obs, states_M, states_W, g_list, c, reset_value_grad=False):
        z = self.percept(obs)
        
        # manager
        value_manager, g, s, states_M = self.manager(z, states_M, False)
        g_list.append(g)
        
        # worker
        sum_g_W = sum(g_list[-c:])
        value_worker, probs, states_W = self.worker(z, sum_g_W, states_W, False)
        
        return probs, value_manager, g, s, states_M, value_worker, states_W