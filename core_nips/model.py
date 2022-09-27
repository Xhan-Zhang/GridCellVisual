# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import utils
import sys
import pdb
#from gru_ei import EIRNN
#from rnn_ei import EIRNN
from rnn_ei1 import EIRNN
class Network(nn.Module):
    def __init__(self, hp):
        super(Network, self).__init__()

        self.Ng = hp['Ng']
        self.Np = hp['Np']
        self.sequence_length = hp['sequence_length']
        self.hp = hp



        self.input_size = hp['input_size']
        self.hidden_size = hp['Ng']
        self.alpha = hp['alpha']
        self.sigma_rec = hp['sigma_rec']

        self.is_cuda = hp['is_cuda']




        self.encoder_layer = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.RNN_layer = EIRNN(hp=self.hp,
                               input_size=hp['input_size'],
                               hidden_size=self.Ng)

        self.decoder_layer = torch.nn.Linear(self.Ng, self.Np, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.act_fcn = lambda x: nn.functional.relu(x)




        for name, param in self.named_parameters():
            print(f"Layer: {name} | Size: {param.size()}\n")


    def grid_hidden(self, inputs):

        v, p0 = inputs



        init_state = self.encoder_layer(p0)[None]



        g, _ = self.RNN_layer.forward_rnn(v, init_state)
        if self.hp['use_relu_grid']:
            g = self.act_fcn(g)


        return g
    

    def forward_predict(self, inputs):

        v, init_actv = inputs

        init_state = self.encoder_layer(init_actv)[None]

        g, h_n = self.RNN_layer.forward_rnn(v, init_state)

        place_preds = self.decoder_layer(g)
        sys.exit(0)
        return place_preds


    def save(self,model_dir):
        if not os.path.isdir(model_dir):
            utils.mkdir_p(model_dir)
        save_path = os.path.join(model_dir, 'most_recent_model.pth')
        torch.save(self.state_dict(), save_path)


    def load_model(self,model_dir):
        if model_dir is not None:

            save_path = os.path.join(model_dir, 'most_recent_model.pth')
            if os.path.isfile(save_path):
                self.load_state_dict(torch.load(save_path,torch.device('cpu')))
            else:
                sys.exit(0)
















