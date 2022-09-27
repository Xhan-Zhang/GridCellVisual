# -*- coding: utf-8 -*-
import torch
import os,sys
import numpy as np
from core_nips.defaults import get_default_hp
from core_nips.place_cells import PlaceCells
from matplotlib import pyplot as plt

#from core_nips.patch_get import generate_patch,pca_image
from core_nips import patch_get
import cv2



import pdb

class TrajectoryGenerator(object):
    def __init__(self, hp, place_cells, img, new_pca_matrix,is_cuda=False):
        self.hp = hp
        self.place_cells = place_cells
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.img=img
        self.new_pca_matrix = new_pca_matrix



    def avoid_wall(self, position, hd, box_width, box_height):
        '''
        Compute distance and angle to nearest wall
        '''
        x = position[:,0]
        y = position[:,1]
        dists = [box_width/2-x, box_height/2-y, box_width/2+x, box_height/2+y]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4)*np.pi/2
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2*np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2*np.pi) - np.pi


        is_near_wall = (d_wall < self.border_region)*(np.abs(a_wall) < np.pi/2)
        turn_angle = np.zeros_like(hd)
        a=np.sign(a_wall[is_near_wall])*(np.pi/2 - np.abs(a_wall[is_near_wall]))
        turn_angle[is_near_wall] = a

        return is_near_wall, turn_angle


    def generate_trajectory(self, box_width=None, box_height=None, batch_size=None):
        '''Generate a random walk in a rectangular box'''

        rng = self.hp['rng']

        if not batch_size:
            batch_size = self.hp['batch_size']
        if not box_width:
            box_width = self.hp['box_width']
        if not box_height:
            box_height = self.hp['box_height']


        samples = self.hp['sequence_length']
        dt = 0.02
        sigma = 5.76 * 2
        b = 0.13 * 2 * np.pi
        mu = 0
        self.border_region = 0.03


        position = np.zeros([batch_size, samples+2, 2])
        head_dir = np.zeros([batch_size, samples+2])

        position[:,0,0] = rng.uniform(-box_width/2, box_width/2, batch_size)
        position[:,0,1] = rng.uniform(-box_height/2, box_height/2, batch_size)
        head_dir[:,0] = rng.uniform(0, 2*np.pi, batch_size)
        speed = np.zeros([batch_size, samples+2])

        random_turn = rng.normal(mu, sigma, [batch_size, samples+1])
        random_speed = rng.normal(b, 0.1,[batch_size, samples+1])


        for t in range(samples+1):

            speed_step = random_speed[:,t]
            turn_angle = np.zeros(batch_size)

            if not self.hp['periodic']:

                is_near_wall, turn_angle = self.avoid_wall(position[:,t], head_dir[:,t], box_width, box_height)
                speed_step[is_near_wall] *= 0.25

            turn_angle += dt*random_turn[:,t]


            speed[:,t] = speed_step*dt


            update_pos = speed[:,t,None]*np.stack([np.cos(head_dir[:,t]), np.sin(head_dir[:,t])], axis=-1)



            position[:,t+1] = position[:,t] + update_pos


            head_dir[:,t+1] = head_dir[:,t] + turn_angle


        if self.hp['periodic']:
            position[:,:,0] = np.mod(position[:,:,0] + box_width/2, box_width) - box_width/2
            position[:,:,1] = np.mod(position[:,:,1] + box_height/2, box_height) - box_height/2

        head_dir = np.mod(head_dir + np.pi, 2*np.pi) - np.pi


        traj = {}
        # Input variables
        traj['init_hd'] = head_dir[:,0,None]
        traj['init_x'] = position[:,1,0,None]
        traj['init_y'] = position[:,1,1,None]

        traj['ego_v'] = speed[:,1:-1]
        ang_v = np.diff(head_dir, axis=-1)
        traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:,:-1], np.sin(ang_v)[:,:-1]

        # Target variables
        traj['target_hd'] = head_dir[:,1:-1]
        traj['target_x'] = position[:,2:,0]
        traj['target_y'] = position[:,2:,1]
        traj['position'] = position[:,2:,:]

        return traj


    def get_batch_for_test(self, speed_scale=None,batch_size=None, box_width=None, box_height=None):
        #print('speed_scale:',speed_scale)
        ''' For testing performance, returns a batch of smample trajectories'''
        if not batch_size:
            batch_size = self.hp['batch_size_test']
        if not box_width:
            box_width = self.hp['box_width']
        if not box_height:
            box_height = self.hp['box_height']

        speed_scale = self.hp['speed_scale']


        traj = self.generate_trajectory(box_width, box_height,batch_size)

        pos = np.stack([traj['target_x'], traj['target_y']],axis=-1)
        pos = torch.tensor(pos,dtype=torch.float32, device=self.device).transpose(0,1)
        #print('pos',pos.shape)
        place_outputs = self.place_cells.get_activation(pos)

        init_pos = np.stack([traj['init_x'], traj['init_y']],axis=-1)
        init_pos = torch.tensor(init_pos,dtype=torch.float32, device=self.device)
        init_actv = 1*self.place_cells.get_activation(init_pos).squeeze()



        if self.hp['vis_input']:
            channel_visual = self.visual_input_img(batch_size=batch_size,position=traj['position'],
                                                   target_hd=traj['target_hd'],box_width=box_width)

        else:

            channel_visual = [np.zeros([batch_size, self.hp['sequence_length']]) for i in range(self.hp['n_components'])]


        channel_1 = speed_scale*traj['ego_v']*np.sin(traj['target_hd'])
        channel_2 = speed_scale*traj['ego_v']*np.cos(traj['target_hd'])


        channel_visual.extend([channel_1, channel_2])


        v = np.stack([channel_visual[i] for i in range(len(channel_visual))],axis=-1)
        v = torch.tensor(v, dtype=torch.float32).transpose(0,1)
        inputs = (v, init_actv)


        return (inputs, pos, place_outputs)




    def get_generator_for_training_visual_img(self, batch_size=None, box_width=None, box_height=None,visual_input=True):

        if not batch_size:
            batch_size = self.hp['batch_size']
        if not box_width:
            box_width = self.hp['box_width']
        if not box_height:
            box_height = self.hp['box_height']

        i=0
        while True:
            i+=1



            traj = self.generate_trajectory(box_width, box_height, batch_size)

            pos = np.stack([traj['target_x'], traj['target_y']],axis=-1)
            pos = torch.tensor(pos, dtype=torch.float32, device=self.device).transpose(0,1)

            init_pos = np.stack([traj['init_x'], traj['init_y']],axis=-1)
            init_pos = torch.tensor(init_pos, dtype=torch.float32, device=self.device)

            init_actv = self.place_cells.get_activation(init_pos).squeeze()
            place_outputs = self.place_cells.get_activation(pos)

            if self.hp['vis_input']:
                channel_visual= self.visual_input_img(batch_size=batch_size,position=traj['position'],
                                                  target_hd=traj['target_hd'],box_width=box_width)
            else:
                channel_visual = [np.zeros([batch_size, self.hp['sequence_length']]) for i in range(self.hp['n_components'])]



            channel_1 = 1*traj['ego_v']*np.sin(traj['target_hd'])
            channel_2 = 1*traj['ego_v']*np.cos(traj['target_hd'])
            channel_visual.extend([channel_1, channel_2])


            v = np.stack([channel_visual[i] for i in range(len(channel_visual))],axis=-1)

            v = torch.tensor(v, dtype=torch.float32).transpose(0,1)


            inputs = (v, init_actv)
            yield (inputs, pos, place_outputs)



    def visual_input_img(self, batch_size=None,position=None,target_hd=None,box_width=None):



        b = 0.13 * 2 * np.pi
        dt = 0.02
        samples = self.hp['sequence_length']
        position_visual_cue = np.array([0.0, box_width/2])
        position = position.reshape(position.shape[0]*position.shape[1],2)



        pixel_lenght  = box_width/550
        step_length = b*dt
        number_grid_for_step = int(step_length/pixel_lenght)
        number_forward_visual = 1
        length_radius = number_forward_visual * step_length



        visual_inputs = []


        for i in range(batch_size*samples):
            target_hd = target_hd.flatten()



            delta_x = length_radius * np.cos(target_hd[i])
            delta_y = length_radius * np.sin(target_hd[i])
            radius_end_x = position[i,0] + delta_x
            radius_end_y = position[i,1] + delta_y

            if (np.abs(radius_end_x)<1.1 and np.abs(radius_end_y)<1.1) and self.hp['vis_input']:


                radius_end = np.array([radius_end_x,radius_end_y])
                radius_center = radius_end/2
                axis_hd_center = self.trans_axis(radius_center)


                x_grid = int(550*axis_hd_center[0]/2.2)
                y_grid = int(550*axis_hd_center[1]/2.2)


                img_goal = patch_get.generate_patch(img=self.img, box_left=x_grid-4,box_right=x_grid+4,
                                                    box_up=y_grid-4,box_below=y_grid+4)

                visual_input = [np.dot(img_goal,self.new_pca_matrix[:,i]) for i in range(self.hp['n_components'])]

                visual_input = np.array(visual_input)

            else:
                visual_input = np.zeros(shape=(self.hp['n_components'],))

            visual_inputs.append(visual_input)

        visual_inputs = np.array(visual_inputs)
        nonzero = visual_inputs.nonzero()
        min_num =visual_inputs[nonzero].min()
        visual_inputs = (visual_inputs - min_num)/(np.max(visual_inputs)-min_num)
        visual_inputs[visual_inputs<0] = 0

        visual_inputs = visual_inputs.transpose()

        visual_channel = [0.05*visual_inputs[i][np.newaxis,:].reshape(batch_size,self.hp['sequence_length']) for i in range(self.hp['n_components'])]


        return visual_channel

    def trans_axis(self,axis_hd_center):

        axis_hd_center[0] = 1.1 + axis_hd_center[0]
        axis_hd_center[1] = 1.1 - axis_hd_center[1]
        return axis_hd_center



if __name__ == '__main__':
    import cv2
    root_path = os.path.abspath(os.path.join(os.getcwd(),"../"))
    figure_dir = os.path.join(root_path, 'z_figure')


    hp = get_default_hp()
    hp['sequence_length'] = hp['seq_length_analysis']
    place_cells = PlaceCells(hp)
    img = cv2.imread('0.png')
    new_pca_matrix = np.load("pca_matrix_img0.npy")

    trajectory_generator = TrajectoryGenerator(hp=hp, place_cells=place_cells, new_pca_matrix =new_pca_matrix,img=img)


    def plot_trajectory_generator():
        traj = trajectory_generator.generate_trajectory(batch_size=5)

        target_x =traj['target_x'][0,:]
        target_y = traj['target_y'][0,:]

        plt.figure(figsize=(5,5))
        plt.plot(target_x,target_y, 'black','-',label='Simulated trajectory')
        plt.plot(target_x,target_y, 'o',markersize=3,label='Simulated trajectory')
        plt.scatter(target_x[0],target_y[0],color='r',marker='o',label='start')
        plt.scatter(target_x[-1],target_y[-1],color='g',marker='^')


        c_recep_field = place_cells.c_recep_field.cpu()
        plt.scatter(c_recep_field[:,0], c_recep_field[:,1], c='lightgrey', label='Place cell centers')
        plt.show()


    def plot_get_generator_for_training():
        batch_size = hp['batch_size_test']
        inputs, pos, place_outputs = trajectory_generator.get_batch_for_test()

        v, _ = inputs


        fig = plt.figure(figsize=(5,5))
        us = place_cells.c_recep_field.cpu()
        pos = pos.cpu()
        plt.scatter(us[:,0], us[:,1], c='C1', label='Place cell centers',s=15,alpha=0.6)
        for i in range(batch_size):
            plt.plot(pos[:,i,0],pos[:,i,1], label='Simulated trajectory', c='grey',linewidth=1)
            # if i==0:
            #     plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])


        plt.savefig(figure_dir+'/trajectory')
        fig.savefig(figure_dir+'/trajectory.eps', format='eps', dpi=1000)


        #"""
        plt.show()
    plot_get_generator_for_training()