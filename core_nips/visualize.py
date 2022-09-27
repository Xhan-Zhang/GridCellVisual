# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import os,sys
import scipy
import scipy.stats
from imageio import imsave
import cv2
from scipy import signal

from core_nips.utils import mkdir_p

import pdb
def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret



def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret



def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps_high_score(activations, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb(im, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig


def rgb_orgin(im, cmap='jet', smooth=True):

    im = (im - np.min(im)) / (np.max(im) - np.min(im))

    cmap = plt.cm.get_cmap(cmap)

    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, smooth=True):

    im = (im - np.min(im)) / (np.max(im) - np.min(im))

    cmap = plt.cm.get_cmap('jet')
    np.seterr(invalid='ignore')
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im

def rgb_correlation(im, smooth=True):
    in1 = im
    in2 = im
    im = signal.correlate(in1, in2, mode='same', method='direct')

    im = (im - np.min(im)) / (np.max(im) - np.min(im))

    cmap = plt.cm.get_cmap('jet')
    np.seterr(invalid='ignore')
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(hp, activations, smooth=True,fig_save=None):

    for i in range(hp['Ng']):
        if i>=5:
            sys.exit(0)


        im = activations[i,:,:]
        image = rgb(im, smooth)

        fig = plt.figure(figsize=(1, 1))
        plt.matshow(image[:,:,0])
        plt.title('unit_'+str(i))
        plt.show()

def plot_ratemaps_correlation(hp, activations, smooth=True,fig_save=None):

    for i in range(hp['Ng']):
        if i>=5:
            sys.exit(0)


        im = activations[i,:,:]
        image = rgb_correlation(im, smooth)


        fig = plt.figure(figsize=(1, 1))
        plt.matshow(image[:,:,0])
        plt.title('unit_'+str(i))
        plt.show()

def plot_ratemaps_0(hp, activations, smooth=True,fig_save=None):


    for i in np.array([17]):

        im = activations[i,:,:]



        im = (im - np.min(im)) / (np.max(im) - np.min(im))# normalization
        cmap = plt.cm.get_cmap('jet')#

        np.seterr(invalid='ignore')
        if smooth:
            im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)

        im = cmap(im)
        im = np.uint8(im * 255)
        im = im[:,:,0]

        fig = plt.figure(figsize=(1, 1))
        plt.matshow(im)

        plt.title('e_prop=0.9'+'/'+'speed_1_sl_'+str(hp['seq_length_analysis']))
        plt.show()



def compute_ratemaps(model, trajectory_generator, hp, speed_scale=0,res=20,  Ng=512, n_avg=None,idxs=None):

    if not n_avg:
        n_avg = 1

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]


    g = np.zeros([n_avg, hp['batch_size_test'] * hp['seq_length_analysis'], Ng])
    pos = np.zeros([n_avg, hp['batch_size_test'] * hp['seq_length_analysis'], 2])


    activations = np.zeros([Ng, res, res]) 
    counts  = np.zeros([res, res])

    for index in range(n_avg):

        inputs, pos_batch, _ = trajectory_generator.get_batch_for_test(speed_scale=speed_scale,batch_size=hp['batch_size_test'])
        print('pos_batch',pos_batch.shape)
        g_batch = model.grid_hidden(inputs).detach().cpu().numpy()
        print('*g_batch',g_batch.shape)
        pdb.set_trace()

        g_batch = g_batch[:,:,idxs].reshape(-1, Ng)
        print('**g_batch',g_batch.shape)



        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])


        x_batch = (pos_batch[:,0] + hp['box_width']/2) / (hp['box_width']) * res
        y_batch = (pos_batch[:,1] + hp['box_height']/2) / (hp['box_height']) * res


        x_pos_list = []
        y_pos_list = []


        for i in range(hp['batch_size_test']*hp['seq_length_analysis']):
            x_pos = x_batch[i]
            y_pos = y_batch[i]

            if x_pos >=0 and x_pos < res and y_pos >=0 and y_pos < res:
                x_pos_list.append(x_pos)
                y_pos_list.append(y_pos)

                counts[int(x_pos), int(y_pos)] += 1
                activations[:, int(x_pos), int(y_pos)] += g_batch[i, :]


    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]


    rate_map = activations.reshape(Ng, -1)

    return activations, rate_map, g, pos



def compute_ratemaps_inh(model, trajectory_generator, hp, speed_scale=0,res=20,  Ng=512, n_avg=None,idxs=None):
    '''Compute spatial firing fields'''
    print('seq_length_analysis',hp['seq_length_analysis'])
    if not n_avg:
        n_avg = 1

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]


    g = np.zeros([n_avg, hp['batch_size_test'] * hp['seq_length_analysis'], Ng])
    pos = np.zeros([n_avg, hp['batch_size_test'] * hp['seq_length_analysis'], 2])


    activations = np.zeros([Ng, res, res])
    counts  = np.zeros([res, res])

    for index in range(n_avg):

        inputs, pos_batch, _ = trajectory_generator.get_batch_for_test(speed_scale=speed_scale,batch_size=hp['batch_size_test'])

        g_batch = model.grid_hidden(inputs).detach().cpu().numpy()

        g_batch = g_batch[:,:,idxs].reshape(-1, Ng)
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])


        x_batch = (pos_batch[:,0] + hp['box_width']/2) / (hp['box_width']) * res
        y_batch = (pos_batch[:,1] + hp['box_height']/2) / (hp['box_height']) * res


        x_pos_list = []
        y_pos_list = []


        for i in range(hp['batch_size_test']*hp['seq_length_analysis']):



            x_pos = x_batch[i]
            y_pos = y_batch[i]

            if x_pos >=0 and x_pos < res and y_pos >=0 and y_pos < res:
                x_pos_list.append(x_pos)
                y_pos_list.append(y_pos)

                counts[int(x_pos), int(y_pos)] += 1
                activations[:, int(x_pos), int(y_pos)] += g_batch[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]


    rate_map = activations.reshape(Ng, -1)

    return activations, rate_map, g, pos




def save_ratemaps(model, trajectory_generator, hp, step, res=20, n_avg=None):
    if not n_avg:
        n_avg = 1000 // hp['sequence_length']
    activations, rate_map, g, pos = compute_ratemaps(model, trajectory_generator,
                                                     hp, res=res, n_avg=n_avg)
    rm_fig = plot_ratemaps(activations, n_plots=len(activations))
    imdir = hp['save_dir'] + "/" + hp['run_ID']
    imsave(imdir + "/" + str(step) + ".png", rm_fig)



def save_autocorr(sess, model, save_name, trajectory_generator, step, flags):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range=((-1.1, 1.1), (-1.1, 1.1))
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, coord_range, masks_parameters)
    
    res = dict()
    index_size = 100
    for _ in range(index_size):
      feed_dict = trajectory_generator.feed_dict(flags.box_width, flags.box_height)
      mb_res = sess.run({
          'pos_xy': model.target_pos,
          'bottleneck': model.g,
      }, feed_dict=feed_dict)
      res = utils.concat_dict(res, mb_res)
        
    filename = save_name + '/autocorrs_' + str(step) + '.pdf'
    imdir = flags.save_dir + '/'
    out = utils.get_scores_and_plot(
                latest_epoch_scorer, res['pos_xy'], res['bottleneck'],
                imdir, filename)



def compute_ratemaps_perturb_input(model, trajectory_generator, hp, res=20, n_avg=None, Ng=512, idxs=None):
    '''Compute spatial firing fields'''

    if not n_avg:
        n_avg = 1000 // hp['sequence_length']

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    g = np.zeros([n_avg, hp['batch_size_test'] * hp['sequence_length'], Ng])
    pos = np.zeros([n_avg, hp['batch_size_test'] * hp['sequence_length'], 2])

    activations = np.zeros([Ng, res, res])
    counts  = np.zeros([res, res])

    for index in range(n_avg):
        print('index',index)
        inputs, pos_batch, _ = trajectory_generator.get_generator_be_perturbed()
        g_batch = model.grid_hidden(inputs).detach().cpu().numpy()
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])


        g_batch = g_batch[:,:,idxs].reshape(-1, Ng)


        g[index] = g_batch
        pos[index] = pos_batch


        x_batch = (pos_batch[:,0] + hp['box_width']/2) / (hp['box_width']) * res
        y_batch = (pos_batch[:,1] + hp['box_height']/2) / (hp['box_height']) * res

        x_pos_list = []
        y_pos_list = []

        for i in range(hp['batch_size_test']*hp['sequence_length']):

            x_pos = x_batch[i]
            y_pos = y_batch[i]

            if x_pos >=0 and x_pos < res and y_pos >=0 and y_pos < res:
                x_pos_list.append(x_pos)
                y_pos_list.append(y_pos)


                counts[int(x_pos), int(y_pos)] += 1
                activations[:, int(x_pos), int(y_pos)] += g_batch[i, :]


    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]

    g = g.reshape([-1, Ng])
    pos = pos.reshape([-1, 2])

    rate_map = activations.reshape(Ng, -1)

    return activations, rate_map, g, pos


def plot_ratemaps_panel(activations, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb(im, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig

def plot_ratemaps_panel_correlation(activations, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb_correlation(im, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig


def rgb_me(im, smooth=True):

    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    np.seterr(invalid='ignore')
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)

    im = np.uint8(im * 255)
    return im
def plot_ratemaps_panel_me(activations, smooth=True):


    images = np.array([rgb_me(im, smooth) for im in activations])



    return images

