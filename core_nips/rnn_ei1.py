import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn.init as init
import pdb
import math

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
import utils

from scipy.stats import ortho_group


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EIRecLinear(nn.Module):
    r"""Recurrent E-I Linear transformation.

    Args:
        hidden_size: int, layer size
        e_prop: float between 0 and 1, proportion of excitatory units
    """
    __constants__ = ['bias', 'hidden_size', 'e_prop']

    def __init__(self, hp, hidden_size, e_prop=0.8, bias=True):
        super().__init__()

        is_cuda = hp['is_cuda']
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.hp = hp

        self.hidden_size = hidden_size
        self.e_prop = self.hp['e_prop']
        self.e_size = int(self.e_prop * hidden_size)
        self.i_size = hidden_size - self.e_size
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        mask = np.tile([1]*self.e_size+[-1]*self.i_size, (hidden_size, 1))
        np.fill_diagonal(mask, 0)
        self.mask = torch.tensor(mask, device=self.device,dtype=torch.float32)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()#=================================================================

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Scale E weight by E-I ratio
        self.weight.data[:, :self.e_size] /= (self.e_size/self.i_size)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def effective_weight(self):
        effective_weights = torch.abs(self.weight) * self.mask

        return effective_weights

    def forward(self, input):
        # weight is non-negative
        return F.linear(input, self.effective_weight(), self.bias)





class EIRNN(nn.Module):


    def __init__(self, hp,input_size, hidden_size, dt=None,
                 e_prop=0.8, sigma_rec=0, **kwargs):
        super().__init__()

        is_cuda = hp['is_cuda']
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.hp = hp

        self.e_prop = hp['e_prop']
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.e_size = int(hidden_size * self.e_prop)
        self.i_size = hidden_size - self.e_size
        self.num_layers = 1
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        # Recurrent noise
        self._sigma_rec = np.sqrt(2*alpha) * sigma_rec
        print('self._sigma_rec',self._sigma_rec)

        # self.input2h = PosWLinear(input_size, hidden_size)
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = EIRecLinear(hp, hidden_size, e_prop=self.e_prop)

        self.softplus = lambda x: nn.functional.softplus(x)
        self.tanh = lambda x: nn.functional.tanh(x)
        self.sigmoid = lambda x: nn.functional.sigmoid(x)


    def init_hidden(self, input):
        batch_size = input.shape[1]
        return (torch.zeros(batch_size, self.hidden_size).to(input.device),
                torch.zeros(batch_size, self.hidden_size).to(input.device))

    def recurrence(self, input, state):
        """Recurrence helper."""

        if self.hp['act_func']=='relu':
            output = torch.relu(state)
        elif self.hp['act_func']=='reluno':
            output = state
        elif self.hp['act_func']=='softmus':
            output = -self.softplus(state)
        elif self.hp['act_func']=='softplus':
            output = self.softplus(state)
        elif self.hp['act_func']=='tanh':
            output = self.tanh(state)
        elif self.hp['act_func']=='sigmoid':
            output = self.sigmoid(state)


        #output = torch.relu(state)
        print('input',input.shape)
        print('output',output.shape)

        state_new = self.input2h(input) + \
                    self.h2h(output) + \
                    self._sigma_rec * torch.randn_like(state)#=========================================

        state = self.alpha * state_new + self.oneminusalpha * state
        output = state#self.softplus(state)
        #output = torch.relu(state)

        return state, output

    def forward_rnn(self, input, init_state):
        """Propogate input through the network."""


        state = init_state

        state_collector = []
        steps = range(input.size(0))
        for i in steps:
            input_per_step = input[i]

            state, output = self.recurrence(input_per_step, state)
            state_collector.append(output)

        state_collector = torch.cat(state_collector, 0)

        return state_collector, state

    def out_weight_clipper(self):
        self.weight_out.data.clamp_(0.)



    def self_weight_clipper(self):
        diag_element = self.h2h.weight.diag().data.clamp_(0., 1.)
        self.h2h.weight.data[range(self.hidden_size), range(self.hidden_size)] = diag_element


    def orthogonal_matrix(self):
        x = ortho_group.rvs(self.hidden_size)
        x = torch.tensor(x, device=self.device,dtype=torch.float32)
        print('x',x.shape)

        return x


class RNNNet(nn.Module):

    def __init__(self, hp, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.rnn = EIRNN(hp, input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(self.rnn.e_size, output_size)

    def forward_rnn(self, x,hidden):
        rnn_activity, _ = self.rnn.forward_rnn(x,hidden)

        rnn_e = rnn_activity[:, :, :self.rnn.e_size]
        out = self.fc(rnn_e)
        return out, rnn_activity





#
#
# if "__main__" == __name__:
#     gru = GRUCell(3,100)
#     input_ =  torch.randn(1,3)
#     h_0    = torch.randn(1,100)
#
#     h_1 = gru.forward(input_, h_0)
#     print(input_.shape, h_0.shape)
#
