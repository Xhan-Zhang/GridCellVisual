# -*- coding: utf-8 -*-
import numpy as np
import torch
from scipy import interpolate


class PlaceCells(object):
    def __init__(self, hp, us=None, is_cuda=False):
        self.Np = hp['Np']
        self.sigma = hp['place_cell_rf']
        self.surround_scale = hp['surround_scale']
        self.box_width = hp['box_width']
        self.box_height = hp['box_height']
        self.is_periodic = hp['periodic']
        self.DoG = hp['DoG']
        self.softmax = torch.nn.Softmax(dim=-1)
        

        np.random.seed(0)
        self.usx = np.random.uniform(-self.box_width/2, self.box_width/2, (self.Np,))
        self.usy = np.random.uniform(-self.box_width/2, self.box_width/2, (self.Np,))

        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.c_recep_field = torch.tensor(np.vstack([self.usx, self.usy]).T).to(self.device)


        
    def get_activation(self, x_pos):

        d = torch.abs(x_pos[:, :, None, :] - self.c_recep_field[None, None, ...]).float()
        norm2 = (d**2).sum(-1)

        outputs = self.softmax(-norm2/(2*self.sigma**2))

        if self.DoG:

            outputs -= self.softmax(-norm2/(2*self.surround_scale*self.sigma**2))


            min_output,_ = outputs.min(-1,keepdims=True)
            outputs += torch.abs(min_output)
            outputs /= outputs.sum(-1, keepdims=True)


        return outputs

    
    def get_nearest_cell_pos(self, activation, k=3):

        value_max_k, idxs = torch.topk(activation, k=k)

        pred_pos = self.c_recep_field[idxs].mean(-2)

        us_x = self.c_recep_field[:,0]


        return pred_pos
        

    def grid_pc(self, pc_outputs, res=32):

        coordsx = np.linspace(-self.box_width/2, self.box_width/2, res)
        coordsy = np.linspace(-self.box_height/2, self.box_height/2, res)
        grid_x, grid_y = np.meshgrid(coordsx, coordsy)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        pc_outputs = pc_outputs.reshape(-1, self.Np)

        
        T = pc_outputs.shape[0]

        pc = np.zeros([T, res, res])
        for i in range(len(pc_outputs)):

            gridval = interpolate.griddata(self.c_recep_field, pc_outputs[i], grid,method='linear', fill_value=np.nan, rescale=False)
            pc[i] = gridval.reshape([res, res])
        
        return pc

    def compute_covariance(self, res=30):
        pos = np.array(np.meshgrid(np.linspace(-self.box_width/2, self.box_width/2, res),
                         np.linspace(-self.box_height/2, self.box_height/2, res))).T


        pos = torch.tensor(pos, device=self.device)



        pc_outputs = self.get_activation(pos).reshape(-1,self.Np)
        self.pc_outputs_for_plot = pc_outputs

        C = np.matmul(pc_outputs,pc_outputs.T)

        Csquare = C.reshape(res,res,res,res)

        Cmean = np.zeros([res,res])
        for i in range(res):
            for j in range(res):
                Cmean += np.roll(np.roll(Csquare[i,j], -i, axis=0), -j, axis=1)
                
        Cmean = np.roll(np.roll(Cmean, res//2, axis=0), res//2, axis=1)


        '''
        fig = plt.figure()
        for i in range(9):
            plt.plot(pc_outputs[i,:])
        fig = plt.figure()
        plt.imshow(self.pc_outputs_for_plot, cmap='jet', interpolation='gaussian')
        
        fig = plt.figure()
        plt.imshow(C, cmap='jet', interpolation='gaussian')
        
        fig = plt.figure()
        plt.imshow(Cmean, cmap='jet', interpolation='gaussian')
        '''
        return Cmean

if __name__ == '__main__':
    from core_nips.defaults import get_default_hp
    from matplotlib import pyplot as plt

    #load parames
    hp = get_default_hp()


    res=3
    place_cells = PlaceCells(hp)
    x_pos = np.array(np.meshgrid(np.linspace(-hp['box_width']/2, hp['box_width']/2, res),
                                 np.linspace(-hp['box_height']/2, hp['box_height']/2, res))).T
    x_pos = torch.tensor(x_pos)



    pc_outputs = place_cells.get_activation(x_pos).reshape(-1,hp['Np'])


    pred_pos = place_cells.get_nearest_cell_pos(pc_outputs, k=3)


    x_pos = x_pos.reshape(res*res, 2)
    plt.plot(x_pos[:,0],x_pos[:,1],'o',c='b',markersize=4)
    plt.plot(place_cells.c_recep_field[:,0],place_cells.c_recep_field[:,1],'o',c='r',markersize=3)
    plt.plot(pred_pos[:,0],pred_pos[:,1],'o',c='green',markersize=5)
    plt.xticks([-hp['box_width']/2, hp['box_width']/2])
    plt.yticks([-hp['box_width']/2, hp['box_width']/2])

    plt.show()

    #


