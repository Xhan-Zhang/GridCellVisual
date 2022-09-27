from sklearn.decomposition import PCA

import sys, os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import scipy
from scipy import io,interpolate
from scipy.interpolate import make_interp_spline

from scipy import signal
import math


sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../")))


from core_nips.utils import generate_run_ID, load_trained_weights,mkdir_p
from core_nips.defaults import get_default_hp,Key_to_value
from core_nips.place_cells import PlaceCells
from core_nips.trajectory_generator import TrajectoryGenerator
from core_nips.model import Network
from core_nips.visualize import compute_ratemaps, plot_ratemaps,plot_ratemaps_high_score
from core_nips.scores import GridScorer
from core_nips import visualize
from core_nips import plot_utility



#load parames
hp = get_default_hp(random_seed=10000)
arg = Key_to_value(hp)

hp['Np'] = 1024
hp['rng'] = np.random.RandomState(1)
hp['run_ID'] = generate_run_ID(arg)
hp['sequence_length'] = hp['seq_length_analysis']



# img
place_cells = PlaceCells(hp)
img=cv2.imread('../img/0.png')
new_pca_matrix = np.load("../img/pca_matrix_img0.npy")
trajectory_generator = TrajectoryGenerator(hp=hp, place_cells=place_cells, img=img,
                                           new_pca_matrix =new_pca_matrix)

def calculate_score(hp,trajectory_generator,model,fig_path):
    res = 50
    n_avg = 1

    low_res = 20
    starts = [0.2] * 10
    ends = np.linspace(0.4, 0.8, num=10)
    box_width=hp['box_width']
    box_height=hp['box_height']
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(low_res, coord_range, masks_parameters)



    _, rate_map_lores, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                         res=low_res,
                                                         Ng=hp['Ng'],
                                                         n_avg=n_avg)
    score_60, score_90, sac, max_60_ind = zip(*[scorer.get_scores(rm.reshape(low_res, low_res)) for rm in tqdm(rate_map_lores)])
    score_type = score_60
    io.savemat(fig_path+'/'+"score_model1.mat",{'score':score_type})

    return score_type


def Evaluate_performance(hp,trajectory_generator,model):
    figure_path = os.path.join(fig_path, 'Evaluate_performance/'+str(hp['seq_length_analysis']))
    mkdir_p(figure_path)


    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    pred_activity = model.forward_predict(inputs)
    pred_pos = place_cells.get_nearest_cell_pos(pred_activity)
    err = np.sqrt(((pos - pred_pos)**2).sum(-1)).mean()*100

    us = place_cells.c_recep_field


    ss=30


    for j in np.array([10]):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        i=j
        plt.plot(pos[:,i,0], pos[:,i,1], label='True position', c='black',linewidth=2.5)
        ax.scatter(pos[0,i,0], pos[0,i,1],   s=ss,marker='o', color='black')
        ax.scatter(pos[-1,i,0], pos[-1,i,1],  s=ss, marker='*', color='black')

        ax.plot(pred_pos[:-2,i,0], pred_pos[:-2,i,1], '-',c='red', label='Decoded position',linewidth=2,zorder=1)
        ax.plot(pred_pos[:-3,i,0], pred_pos[:-3,i,1], 'o',c='tab:blue', markersize=4, label='Decoded position')
        ax.scatter(pred_pos[0,i,0], pred_pos[0,i,1], s=ss,  marker='o', color='tab:blue')
        ax.scatter(pred_pos[-3,i,0], pred_pos[-3,i,1],  s=ss, marker='*', color='tab:blue',zorder=3)




    for k1 in range(100):
        fig1 = plt.figure(figsize=(5,5))
        ax1 = fig1.add_subplot(111)

        plt.plot(pos[:,k1,0], pos[:,k1,1], c='black', label='True position', linewidth=1)
        ax1.scatter(pos[0,k1,0], pos[0,k1,1],   marker='o', color='red')
        ax1.scatter(pos[-1,k1,0], pos[-1,k1,1],   marker='*', color='red')

        plt.plot(pred_pos[:-1,k1,0], pred_pos[:-1,k1,1], '.-', c='tab:orange', label='Decoded position',linewidth=1)
        ax1.scatter(pred_pos[0,k1,0], pred_pos[0,k1,1],   marker='o', color='b')
        ax1.scatter(pred_pos[-2,k1,0], pred_pos[-2,k1,1],   marker='*', color='b')
        plt.xlim([-hp['box_width']/2,hp['box_width']/2])
        plt.ylim([-hp['box_height']/2,hp['box_height']/2])
        plt.title(k1)
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
        plt.savefig(figure_path+'/'+str(k1)+'.png')



    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-0.9,0.1])
    plt.ylim([-0.2,0.7])
    plt.title('batch_size_test='+ str(hp['batch_size_test'])+' '+
              'seq_length_analysis='+ str(hp['seq_length_analysis'])+
              '\n'+str(hp['act_func'])+'+'+model_idx)


    plt.savefig(fig_path+'/'+'Evaluate_performance_'+str(hp['seq_length_analysis'])+'.pdf')
    plt.show()
#Evaluate_performance()

def ratemap_one_units(hp,trajectory_generator,model,unit_idx,speed_scale=0):
    figure_path = os.path.join(fig_path, 'fig1/')
    mkdir_p(figure_path)

    res = 20#
    n_avg = 1#

    activations, rate_map_lores, _, _ = compute_ratemaps(model, trajectory_generator, hp,
                                                         speed_scale=speed_scale,
                                                         res=res,
                                                         Ng=hp['Ng'],
                                                         n_avg=n_avg
                                                         )


    for i in unit_idx:


        im = activations[i,:,:]

        print('im',im.shape)
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        cmap = plt.cm.get_cmap('jet')

        np.seterr(invalid='ignore')
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
        im = cmap(im)
        im = im[:,:,0]


        fig = plt.figure(figsize=(1, 1))
        plt.matshow(im,vmin=0.1, vmax=0.9)
        plt.title('unit_'+str(i)+';'+'model_'+str(hp['seq_length_model'])+';'+
                  'speed_'+str(speed_scale)+';'+'sl_'+str(hp['seq_length_analysis'])+';'+
                  'visual_'+str(hp['vis_input']))
        plt.axis('off')
        plt.savefig(figure_path+'/'+'unit_'+str(i)+'.png')
        plt.show()





def Calculate_grid_score(hp,trajectory_generator,model,unit_idx):
    figure_path = os.path.join(fig_path, 'Calculate_grid_score/'+'score_60')
    mkdir_p(figure_path)
    res = 50
    n_avg = 1

    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)


    low_res = 20
    _, rate_map_lores, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                         res=low_res,
                                                         Ng=hp['Ng'],
                                                         n_avg=n_avg)


    rate_map_lores_ext = rate_map_lores[unit_idx]

    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    box_width=hp['box_width']
    box_height=hp['box_height']
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(low_res, coord_range, masks_parameters)

    score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
        *[scorer.get_scores(rm.reshape(low_res, low_res)) for rm in tqdm(rate_map_lores_ext)])


    score_type = score_60
    io.savemat(figure_path+'/'+"score_ext.mat",{'score':score_type})


    k = -1
    for idx in unit_idx:
        k += 1
        im = activations[idx,:,:]
        image = visualize.rgb(im, smooth=True)

        fig = plt.figure(figsize=(1, 1))
        plt.matshow(image[:,:,0])
        plt.title('unit_'+str(idx)+'; score:' +str(np.round(score_type[k],2)))
        plt.axis('off')
        plt.savefig(figure_path+'/'+str(k)+'.png')
        plt.show()


def Calculate_percent_grid_ext_inh(hp,trajectory_generator,model,fig_path):
    figure_path = os.path.join(fig_path, 'Calculate_percent_grid_ext_inh/'+'score_60')
    mkdir_p(figure_path)
    res = 50
    n_avg = 1

    low_res = 20
    starts = [0.2] * 10
    ends = np.linspace(0.4, 0.8, num=10)
    box_width=hp['box_width']
    box_height=hp['box_height']
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(low_res, coord_range, masks_parameters)


    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)


    _, rate_map_lores, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                         res=low_res,
                                                         Ng=hp['Ng'],
                                                         n_avg=n_avg)
    score_60, score_90, sac, max_60_ind = zip(*[scorer.get_scores(rm.reshape(low_res, low_res)) for rm in tqdm(rate_map_lores)])

    score_type = score_60
    io.savemat(figure_path+'/'+"score_ext.mat",{'score':score_type})


    load_data = io.loadmat(figure_path+'/'+"score_ext.mat")
    score_type = load_data['score'][0,:]

    score_exc = score_type[:410]
    idxs_exc = np.flip(np.argsort(score_exc))
    score_inh = score_type[410:]
    idxs_inh = np.flip(np.argsort(score_inh))


    k_exc=-1
    for i in idxs_exc:
        k_exc+=1
        if np.abs(score_exc[i])<0.3:
            break
    idx_exc = idxs_exc[:k_exc]
    k_inh=-1

    for j in idxs_inh:
        k_inh+=1
        if np.abs(score_inh[j])<0.3:
            break
    idx_inh = idxs_inh[:k_inh]

    percentage_E_grid = np.round(idx_exc.shape[0]/410,6)
    percentage_I_grid = np.round(idx_inh.shape[0]/102,6)


    n_plot = 25
    fig0 = plt.figure(figsize=(10,10))
    ax = fig0.add_axes([0.05, 0.05, 0.8, 0.8])
    rm_fig = visualize.plot_ratemaps_panel(activations[idx_exc], n_plot, smooth=True,width=5)
    plt.imshow(rm_fig)
    plt.title('percentage of E-grid: '+str(idx_exc.shape[0]/410),fontsize=16)
    plt.axis('off')
    plt.savefig(figure_path+'/high_grid_scores_E.png')
    #plt.show()


    n_plot = 25
    fig1 = plt.figure(figsize=(10,10))
    ax = fig1.add_axes([0.05, 0.05, 0.9, 0.9])

    rm_fig = visualize.plot_ratemaps_panel(activations[idx_inh+410], n_plot, smooth=True,width=5)
    plt.imshow(rm_fig)
    plt.suptitle('percentage of I-grid: '+str(idx_inh.shape[0]/102),fontsize=16)
    plt.axis('off')
    plt.savefig(figure_path+'/fig1B_high_score_I.png')
    #plt.show()


    return percentage_E_grid, percentage_I_grid

def Plot_score_distribute_ext_inh(hp,trajectory_generator,model,fig_path):
    figure_path = os.path.join(fig_path, 'Plot_score_distribute/'+'score_60')
    mkdir_p(figure_path)
    res = 50
    n_avg = 1

    low_res = 20
    starts = [0.2] * 10
    ends = np.linspace(0.4, 0.8, num=10)
    box_width=hp['box_width']
    box_height=hp['box_height']
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(low_res, coord_range, masks_parameters)


    load_data = io.loadmat(figure_path+'/'+"score_all.mat")
    score_type = load_data['score'][0,:]

    score_exc = score_type[:410]
    idxs_exc = np.flip(np.argsort(score_exc))
    score_inh = score_type[410:]
    idxs_inh = np.flip(np.argsort(score_inh))


    hist_E = np.histogram(score_exc,bins = 20)
    hist_I = np.histogram(score_inh,bins = 20)
    prop_E = hist_E[0]/410
    prop_I = hist_I[0]/102

    number_dot_interp = 200

    model_E=make_interp_spline(hist_E[1][1:], prop_E)
    model_I=make_interp_spline(hist_I[1][1:], prop_I)
    xs_E=np.linspace(hist_E[1][1],hist_E[1][-1],number_dot_interp)#500
    ys_E=model_E(xs_E)
    xs_I=np.linspace(hist_I[1][1],hist_I[1][-1],number_dot_interp)#
    ys_I=model_I(xs_I)



    return xs_E,ys_E,xs_I,ys_I



def Plot_high_score_grid_ext_inh_panel(hp,trajectory_generator,model,fig_path):
    figure_path = os.path.join(fig_path, 'Plot_score_ext_inh/'+'score_60')
    mkdir_p(figure_path)
    res = 50
    n_avg = 1

    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)

    data_path = os.path.join(fig_path, 'Calculate_percent_grid_ext_inh/'+'score_60')
    load_data = io.loadmat(data_path+'/'+"score_ext.mat")
    score_type = load_data['score'][0,:]

    score_exc = score_type[:410]
    idxs_exc = np.flip(np.argsort(score_exc))
    score_inh = score_type[410:]
    idxs_inh = np.flip(np.argsort(score_inh))

    n_plot = 100
    fig0 = plt.figure(figsize=(10,10))
    ax = fig0.add_axes([0.05, 0.05, 0.9, 0.9])
    rm_fig = visualize.plot_ratemaps_panel(activations[idxs_exc], n_plot, smooth=True,width=10)
    plt.imshow(rm_fig)
    plt.title('percentage of E-grid: '+str(idxs_exc.shape[0]/410),fontsize=16);plt.axis('off')
    plt.savefig(figure_path+'/high_grid_scores_E.png')
    plt.show()

    n_plot = 100
    fig1 = plt.figure(figsize=(10,10))
    ax = fig1.add_axes([0.05, 0.05, 0.9, 0.9])

    rm_fig = visualize.plot_ratemaps_panel(activations[idxs_inh+410], n_plot, smooth=True,width=10)
    plt.imshow(rm_fig)
    plt.suptitle('percentage of I-grid: '+str(idxs_inh.shape[0]/102),fontsize=16);plt.axis('off')
    plt.savefig(figure_path+'/fig1B_high_score_I.png')
    plt.show()
    #"""


def Plot_grid_score_example_unit(hp,trajectory_generator,model,unit_idx,fig_path):
    figure_path = os.path.join(fig_path, 'Plot_grid_score_example_unit/'+'score_60')
    mkdir_p(figure_path)
    res = 50
    n_avg = 1

    low_res = 30
    starts = [0.2] * 10
    ends = np.linspace(0.4, 0.8, num=10)
    box_width=hp['box_width']
    box_height=hp['box_height']
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(low_res, coord_range, masks_parameters)



    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)

    data_path = os.path.join(fig_path, 'Plot_grid_score_high_unit/'+'score_60')
    load_score = io.loadmat(data_path+'/'+"score_ext.mat")
    load_sacs = io.loadmat(data_path+'/'+"sacs_ext.mat")
    score_type = load_score['score'][0,:]
    sacs = load_sacs['sac']

    for idx in unit_idx:

        fig,axs = plt.subplots(1,2,figsize=(6,3))
        im = activations[idx,:,:]
        im = (im - np.min(im)) / (np.max(im) - np.min(im))# normalization
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
        axs[0].imshow(im, interpolation='none', cmap='jet');axs[0].axis('off')

        im_correlation = sacs[idx,:,:]
        scorer.plot_sac(im_correlation,ax=axs[1])
        fig.suptitle('unit_'+str(idx)+'; score:' +str(np.round(score_type[idx],2)))
        plt.savefig(figure_path+'/'+'unit_'+str(idx)+'_'+str(np.round(score_type[idx],2))+'.png')
        plt.show()



def Plot_ratemap_example10_unit(hp,trajectory_generator,model,fig_path):
    figure_path = os.path.join(fig_path, 'Plot_ratemap_example10_unit/'+'score_60')
    mkdir_p(figure_path)
    res = 50
    n_avg = 1
    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)


    data_path = os.path.join(fig_path, 'Plot_ratemap_autocorrelation/'+'score_60')
    load_score = io.loadmat(data_path +'/'+"score_ext.mat")
    score_type = load_score['score'][0,:]

    idxs = np.array([297,184,422,125,438,349,57,481,342,109])

    fig = plt.figure(figsize=(9,1))
    ax = fig.add_axes([0.0, 0.0, 0.8, 0.8])
    for k in range(10):
        plt.subplot(1,10,k+1)
        im = activations[idxs[k],:,:]
        if hp['rgb']:
            im=visualize.rgb(im)
            plt.imshow(im[:,:,0], interpolation='none', cmap='jet');plt.axis('off')
        else:
            im = (im - np.min(im)) / (np.max(im) - np.min(im))# normalization
            im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
            plt.imshow(im, interpolation='none', cmap='jet');plt.axis('off')
    plt.title('setup#2')
    plt.savefig(figure_path+'/'+'setup#2'+'.png')

    plt.show()


def Plot_grid_score_high_1(hp,trajectory_generator,model):
    figure_path = os.path.join(fig_path, 'Grid_scores_high/'+str(hp['seq_length_analysis']))
    mkdir_p(figure_path)
    res = 50
    n_avg = 1

    activations, _, _, _ = compute_ratemaps(model, trajectory_generator, hp,
                                            res=res,
                                            Ng=hp['Ng'],
                                            n_avg=n_avg)

    lo_res = 20
    _, rate_map_lores, _, _ = compute_ratemaps(model, trajectory_generator, hp,
                                               res=lo_res,
                                               Ng=hp['Ng'],
                                               n_avg=n_avg)



    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    box_width=hp['box_width']
    box_height=hp['box_height']
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(lo_res, coord_range, masks_parameters)

    score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
        *[scorer.get_scores(rm.reshape(lo_res, lo_res)) for rm in tqdm(rate_map_lores)])



    idxs = np.flip(np.argsort(score_60))
    Ng = hp['Ng']

    n_plot = 128
    plt.figure(figsize=(16,4*n_plot//8**2))
    rm_fig = plot_ratemaps_high_score(activations[idxs], n_plot, smooth=True)
    plt.imshow(rm_fig)
    plt.suptitle('Grid scores '+str(np.round(score_60[idxs[0]], 2))
                 +' -- '+ str(np.round(score_60[idxs[n_plot]], 2)),
                 fontsize=16)
    plt.title(str(model_idx))
    plt.savefig(figure_path+'/high_grid_scores.png')
    plt.axis('off')
    plt.show()


def Plot_ratemap_autocorrelation(hp,trajectory_generator,model,fig_path):

    figure_path = os.path.join(fig_path, 'Plot_ratemap_autocorrelation/'+'score_60_order1')
    mkdir_p(figure_path)
    res = 50
    n_avg = 1

    low_res = 20
    starts = [0.2] * 10
    ends = np.linspace(0.4, 0.8, num=10)
    box_width=hp['box_width']
    box_height=hp['box_height']
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(low_res, coord_range, masks_parameters)



    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)



    _, rate_map_lores, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,res=low_res,Ng=hp['Ng'],n_avg=n_avg)
    score_60s = []
    sacs = []
    for rate_map_lo in rate_map_lores:
        rate_map = rate_map_lo.reshape(low_res, low_res)
        score_60, score_90, sac, max_60_ind = scorer.get_scores(rate_map)
        score_60s.append(score_90)
        sacs.append(sac)
    score_type = np.array(score_60s)
    sacs = np.array(sacs)
    io.savemat(figure_path+'/'+"score_ext.mat",{'score':score_type})
    io.savemat(figure_path+'/'+"sacs_ext.mat",{'sac':sacs })



    data_path = figure_path
    load_score = io.loadmat(data_path+'/'+"score_ext.mat")
    load_sacs = io.loadmat(data_path+'/'+"sacs_ext.mat")
    score_type = load_score['score'][0,:]
    sacs = load_sacs['sac']


    idxs = np.flip(np.argsort(score_type))

    k=0
    for idx in np.array(idxs):
        k+=1


        fig,axs = plt.subplots(1,2,figsize=(6,3))

        im = activations[idx,:,:]
        if hp['rgb']:
            im=visualize.rgb(im)
            axs[0].imshow(im[:,:,0], interpolation='none', cmap='jet');axs[0].axis('off')

            im_correlation = sacs[idx,:,:]
            im_correlation = visualize.rgb(im_correlation)
            scorer.plot_sac(im_correlation[:,:,0],ax=axs[1])
        else:
            im = (im - np.min(im)) / (np.max(im) - np.min(im))# normalization
            im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
            axs[0].imshow(im, interpolation='none', cmap='jet');axs[0].axis('off')

            im_correlation = sacs[idx,:,:]
            scorer.plot_sac(im_correlation,ax=axs[1])

        fig.suptitle('unit_'+str(idx)+'; score:' +str(np.round(score_type[idx],2)))
        plt.savefig(figure_path+'/'+str(k)+'unit_'+str(idx)+'_'+str(np.round(score_type[idx],2))+'.png')

        plt.show()




def ratemap_units(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'ratemap_units/')
    mkdir_p(figure_path)
    res = 50
    n_avg = 1

    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)
    unit_idx = range(512)
    for idx in unit_idx:
        fig0 = plt.figure(figsize=(5,5))
        im = activations[idx,:,:]
        im=visualize.rgb(im)
        plt.imshow(im[:,:,0], interpolation='none', cmap='jet');plt.axis('off')

        plt.title(str(idx)+'_'+str(hp['vis_input'])+';vscale_'+str(hp['speed_scale']))
        plt.savefig(figure_path+str(idx)+'.png')




def speed_tuning(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'speed_tuning_only/')
    mkdir_p(figure_path)

    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    v = inputs[0].cpu().numpy()
    hidden_firing_batch = model.grid_hidden(inputs).detach().cpu().numpy()

    vx = v[:,:,-1]
    vy = v[:,:,-2]

    v = np.sqrt(vx**2 + vy**2)
    vs = np.stack(v).ravel()
    hidden_firing = np.reshape(hidden_firing_batch, (-1,hp['Ng']))


    v_curves = []
    v_scores = []
    bin_number=9
    for i in range(hp['Ng']):
        stat,bins,_ = scipy.stats.binned_statistic(vs,hidden_firing[:,i], statistic='mean',bins=bin_number)
        v_curves.append(stat)
        v_scores.append(np.corrcoef(vs, hidden_firing[:,i])[0,1])
    v_curves = np.stack(v_curves)


    for i in unit_idx:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])

        plt.plot(v_curves[i],'-',color='black',linewidth=2)
        ax.set_ylim([0.0,0.04])


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        plt.title('unit_'+str(i),fontsize=5)
        plt.xticks([])
        plt.yticks([])

        plt.xlabel('Speed',fontsize=16)
        plt.ylabel('Firing rate',fontsize=16)

        plt.savefig(fig_path+'/'+str(i)+'speed_tune'+'.png')
        fig.savefig(figure_path+'/speed_tune_'+str(i)+'.eps', format='eps', dpi=1000)
        plt.show()


def hd_tuning(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'hd_tuning_only/')
    mkdir_p(figure_path)

    Ng=hp['Ng']

    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    v = inputs[0].cpu().numpy()
    hidden_firing_batch = model.grid_hidden(inputs).detach().cpu().numpy()

    vx = v[:,:,-1]
    vy = v[:,:,-2]

    hd = np.arctan2(vy,vx)*180/np.pi
    hidden_firing_batch = np.reshape(hidden_firing_batch, (-1,Ng))

    hidden_firing = hidden_firing_batch
    hds = hd.ravel()

    hd_curves = []
    hd_scores = []
    bin_number=10
    for i in range(hp['Ng']):
        stat,bins,_ = scipy.stats.binned_statistic(hds,hidden_firing[:,i], statistic='mean',bins=bin_number, range=(-np.pi*180/np.pi,np.pi*180/np.pi))
        hd_curves.append(stat)
        hd_scores.append(np.corrcoef(hds, hidden_firing[:,i])[0,1])
    hd_curves = np.stack(hd_curves)
    print('hd_curves',hd_curves.shape,hd_curves)

    ###############################plot panel
    """
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    x_axis = np.linspace(-np.pi*180/np.pi,np.pi*180/np.pi,bin_number)
    for i in range(20,40,1):
        #print(i,'hd_curves[i]',hd_curves[i].shape,hd_curves[i])
        #plt.subplot(4,4,i+1, projection='polar')
        plt.subplot(4,5,(i+1)-20)
        plt.plot(x_axis,hd_curves[i],'-',linewidth=2)
        #plt.hist(hd_curves[i],bins=20)
        #plt.xlabel([-180,0,180])
        if i == 22:
            plt.title('hd tuning')
        if i == 20:
            plt.ylabel('firing of unit [0,1]')
        if i ==37:
            plt.xlabel('Direction input')

        plt.xticks([])
        plt.yticks([])
        plt.ylim([-0.001,0.063])
        plt.savefig(figure_path+'/'+'hd tune')
        #plt.axis('off')
    """


    for i in unit_idx:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])

        print(i,'hd_curves[i]',hd_curves[i].shape,hd_curves[i])
        #plt.subplot(projection='polar')
        plt.plot(hd_curves[i],'-',linewidth=2)#, linewidth=1


        plt.title('unit_'+str(i),fontsize=5)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        plt.xlabel('direction',fontsize=16)
        plt.ylabel('Firing rate',fontsize=16)
        plt.ylim([-0.0,0.062])


        plt.savefig(fig_path+'/'+str(i)+'hd_tune_'+'.png')
        fig.savefig(figure_path+'/hd_tune_'+str(i)+'.eps', format='eps', dpi=1000)
        plt.show()

def Speed_hd_ratemap(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'Plot_speed_hd_ratemap/')
    mkdir_p(figure_path)
    v_curves = cal_speed_tuning(hp,trajectory_generator,model,fig_path)
    hd_curves = cal_hd_tuning(hp,trajectory_generator,model,fig_path)


    res = 50
    n_avg = 1

    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)
    for idx in unit_idx:
        fig,axs = plt.subplots(1,3,figsize=(9,3))
        im = activations[idx,:,:]
        im=visualize.rgb(im)
        axs[0].imshow(im[:,:,0], interpolation='none', cmap='jet');axs[0].axis('off')
        axs[1].plot(hd_curves[idx],'-',linewidth=2,color='tab:blue');axs[1].set_ylim([0,0.06])
        axs[2].plot(v_curves[idx],'-',linewidth=2,color='black');axs[2].set_ylim([0.01,0.035])

        plt.title(str(idx)+'_'+str(hp['vis_input']))
        plt.savefig(figure_path+str(idx)+str(hp['vis_input'])+'.png')
        plt.show()


def speed_tuning_S1(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'speed_tuning_S1/')
    mkdir_p(figure_path)

    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    v = inputs[0].cpu().numpy()
    hidden_firing_batch = model.grid_hidden(inputs).detach().cpu().numpy()

    vx = v[:,:,-1]
    vy = v[:,:,-2]

    v = np.sqrt(vx**2 + vy**2)
    vs = np.stack(v).ravel()
    hidden_firing = np.reshape(hidden_firing_batch, (-1,hp['Ng']))

    # Construct head direction tuning curves
    v_curves = []
    v_scores = []
    bin_number=9
    for i in range(hp['Ng']):
        stat,bins,_ = scipy.stats.binned_statistic(vs,hidden_firing[:,i], statistic='mean',bins=bin_number)
        v_curves.append(stat)
        v_scores.append(np.corrcoef(vs, hidden_firing[:,i])[0,1])
    v_curves = np.stack(v_curves)


    for i in unit_idx:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])

        plt.plot(v_curves[i],'-',color='black',linewidth=2)#, linewidth=1
        ax.set_ylim([-0.0,0.03])
        #ax.scatter(hd_curves[i],color='r')

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        # plt.title('unit_'+str(i),fontsize=5)
        # plt.xticks([])
        # plt.yticks([])

        plt.xlabel('Speed',fontsize=16)
        plt.ylabel('Firing rate',fontsize=16)
        #plt.show()
        plt.savefig(figure_path+str(i)+'speed_tune'+'.png')
        #fig.savefig(figure_path+'/speed_tune_'+str(i)+'.eps', format='eps', dpi=1000)
        #plt.show()
    #"""

def hd_tuning_S1(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'hd_tuning_S1/')
    mkdir_p(figure_path)

    Ng=hp['Ng']

    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    v = inputs[0].cpu().numpy()
    hidden_firing_batch = model.grid_hidden(inputs).detach().cpu().numpy()

    vx = v[:,:,-1]
    vy = v[:,:,-2]

    hd = np.arctan2(vy,vx)*180/np.pi
    hidden_firing_batch = np.reshape(hidden_firing_batch, (-1,Ng))

    hidden_firing = hidden_firing_batch
    hds = hd.ravel()



    hd_curves = []
    hd_scores = []
    bin_number=10
    for i in range(hp['Ng']):
        stat,bins,_ = scipy.stats.binned_statistic(hds,hidden_firing[:,i], statistic='mean',bins=bin_number, range=(-np.pi*180/np.pi,np.pi*180/np.pi))
        hd_curves.append(stat)
        hd_scores.append(np.corrcoef(hds, hidden_firing[:,i])[0,1])
    hd_curves = np.stack(hd_curves)

    """
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    x_axis = np.linspace(-np.pi*180/np.pi,np.pi*180/np.pi,bin_number)
    for i in range(20,40,1):
        #print(i,'hd_curves[i]',hd_curves[i].shape,hd_curves[i])
        #plt.subplot(4,4,i+1, projection='polar')
        plt.subplot(4,5,(i+1)-20)
        plt.plot(x_axis,hd_curves[i],'-',linewidth=2)
        #plt.hist(hd_curves[i],bins=20)
        #plt.xlabel([-180,0,180])
        if i == 22:
            plt.title('hd tuning')
        if i == 20:
            plt.ylabel('firing of unit [0,1]')
        if i ==37:
            plt.xlabel('Direction input')

        plt.xticks([])
        plt.yticks([])
        plt.ylim([-0.001,0.063])
        plt.savefig(figure_path+'/'+'hd tune')
        #plt.axis('off')
    """


    for i in unit_idx:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])

        print(i,'hd_curves[i]',hd_curves[i].shape,hd_curves[i])
        #plt.subplot(projection='polar')
        plt.plot(hd_curves[i],'-',linewidth=2)#, linewidth=1
        ax.set_ylim([0,0.06])


        plt.title('unit_'+str(i),fontsize=5)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        plt.xlabel('direction',fontsize=16)
        plt.ylabel('Firing rate',fontsize=16)



        plt.savefig(figure_path+str(i)+'hd_tune_'+'.png')
        #fig.savefig(figure_path+'/hd_tune_'+str(i)+'.eps', format='eps', dpi=1000)
        #plt.show()


def hd_tuning1(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'fig1/')
    mkdir_p(figure_path)

    Ng=hp['Ng']

    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    v = inputs[0].cpu().numpy()
    hidden_firing_batch = model.grid_hidden(inputs).detach().cpu().numpy()

    vx = v[:,:,-1]
    vy = v[:,:,-2]

    hd = np.arctan2(vy,vx)*180/np.pi
    hidden_firing_batch = np.reshape(hidden_firing_batch, (-1,Ng))

    hidden_firing = hidden_firing_batch
    hds = hd.ravel()
    print('@hds',hds.shape)
    print('@hidden_firing',hidden_firing.shape)

    hd_curves = []
    hd_scores = []
    bin_number=10
    for i in range(hp['Ng']):
        stat,bins,_ = scipy.stats.binned_statistic(hds,hidden_firing[:,i], statistic='mean',bins=bin_number, range=(-np.pi*180/np.pi,np.pi*180/np.pi))
        hd_curves.append(stat)
        hd_scores.append(np.corrcoef(hds, hidden_firing[:,i])[0,1])
    hd_curves = np.stack(hd_curves)

    ###############################plot panel
    """
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    x_axis = np.linspace(-np.pi*180/np.pi,np.pi*180/np.pi,bin_number)
    for i in range(20,40,1):
        #print(i,'hd_curves[i]',hd_curves[i].shape,hd_curves[i])
        #plt.subplot(4,4,i+1, projection='polar')
        plt.subplot(4,5,(i+1)-20)
        plt.plot(x_axis,hd_curves[i],'-',linewidth=2)
        #plt.hist(hd_curves[i],bins=20)
        #plt.xlabel([-180,0,180])
        if i == 22:
            plt.title('hd tuning')
        if i == 20:
            plt.ylabel('firing of unit [0,1]')
        if i ==37:
            plt.xlabel('Direction input')

        plt.xticks([])
        plt.yticks([])
        plt.ylim([-0.001,0.063])
        plt.savefig(figure_path+'/'+'hd tune')
        #plt.axis('off')
    """


    for i in unit_idx:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])

        print(i,'hd_curves[i]',hd_curves[i].shape,hd_curves[i])
        plt.plot(hd_curves[i],'-',linewidth=2)#, linewidth=1


        plt.title('unit_'+str(i),fontsize=5)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        plt.xlabel('direction',fontsize=16)
        plt.ylabel('Firing rate',fontsize=16)


        plt.savefig(figure_path+'/hd_tune_'+str(i)+'.png')
        fig.savefig(figure_path+'/hd_tune_'+str(i)+'.eps', format='eps', dpi=1000)
        plt.show()


def visual_tuning2(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'visual_tuning2/')
    mkdir_p(figure_path)

    Ng=hp['Ng']

    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    v = inputs[0].cpu().numpy()
    hidden_firing_batch = model.grid_hidden(inputs).detach().cpu().numpy()
    hidden_firing = np.reshape(hidden_firing_batch, (-1,Ng))


    #visual
    vis_PC1 = v[:,:,0]
    vis_PC1s = np.stack(vis_PC1).ravel()
    hidden_firing = np.reshape(hidden_firing_batch, (-1,hp['Ng']))


    # Construct head direction tuning curves
    vis_curves = []
    vis_scores = []
    bin_number=10
    unit_idx = unit_idx
    for i in range(hp['Ng']):
        stat,bins,_ = scipy.stats.binned_statistic(vis_PC1s,hidden_firing[:,i], statistic='mean',bins=bin_number)
        vis_curves.append(stat)
        vis_scores.append(np.corrcoef(vis_PC1s, hidden_firing[:,i])[0,1])
    print('vis_scores',vis_scores)
    vis_curves = np.stack(vis_curves)
    print('vis_curves',vis_curves.shape)

    #plot
    for i in unit_idx:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])

        plt.plot(vis_curves[i],'-',color='tab:green',linewidth=2)#, linewidth=1
        #ax.scatter(hd_curves[i],color='r')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title('unit_'+str(i),fontsize=5)
        # plt.xticks([])
        # plt.yticks([])
        plt.xlabel('Visual_PC1',fontsize=16)
        plt.ylabel('Firing rate',fontsize=16)
        #plt.show()
        plt.savefig(figure_path+'/vis_tune_'+str(i)+'.png')
        #fig.savefig(figure_path+'/vis_tune_'+str(i)+'.eps', format='eps', dpi=1000)

        plt.show()




def visual_tuning1(hp,trajectory_generator,model,unit_idx):
    figure_path = os.path.join(fig_path, 'hd_tuning')
    mkdir_p(figure_path)

    Ng=hp['Ng']

    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    v = inputs[0].cpu().numpy()
    hidden_firing_batch = model.grid_hidden(inputs).detach().cpu().numpy()
    hidden_firing = np.reshape(hidden_firing_batch, (-1,Ng))

    # Construct head direction tuning curves
    vis_curves = []
    vis_scores = []
    bin_number=10
    unit_idx = unit_idx
    for i in range(3):
        vis = v[:,:,i]
        vis = vis.ravel()
        stat,bins,_ = scipy.stats.binned_statistic(vis,hidden_firing[:,unit_idx], statistic='mean',bins=bin_number)
        vis_curves.append(stat)
        vis_scores.append(np.corrcoef(vis, hidden_firing[:,unit_idx])[0,1])
    vis_curves = np.stack(vis_curves)

    #plot
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    plt.title('unit_'+str(unit_idx))
    for i in range(3):
        plt.subplot(4,5,i+1)
        plt.plot(vis_curves[i],'-')
        #plt.axis('off')
        if i == 0:
            plt.title('unit_'+str(unit_idx))


    """

    fig1 = plt.figure(figsize=(8,8))
    ax = fig1.add_axes([0.05, 0.05, 0.9, 0.9])
    for i in range(512):
        if i>=2:
            sys.exit(0)
        print(i,'hd_curves[i]',hd_curves[i].shape,hd_curves[i])
        plt.plot(hd_curves[i],'-')#, linewidth=1
        #ax.scatter(hd_curves[i],color='r')

        plt.title('unit_'+str(i))
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        #plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        # plt.ylim([-0.05,0.05])
        # plt.xlim([55,70])

        #plt.axis('off')
        plt.show()
        plt.savefig(figure_path+'/'+str(i))
    """

    plt.show()

def hd_tuning2(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'fig1/'+str(hp['seq_length_analysis']))
    mkdir_p(figure_path)

    Ng=hp['Ng']

    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    v = inputs[0].cpu().numpy()
    hidden_firing_batch = model.grid_hidden(inputs).detach().cpu().numpy()

    vx = v[:,:,-1]
    vy = v[:,:,-2]

    hd = np.arctan2(vy,vx)*180/np.pi
    hidden_firing_batch = np.reshape(hidden_firing_batch, (-1,Ng))

    hidden_firing = hidden_firing_batch
    hds = hd.ravel()


    hd_curves = []
    hd_scores = []
    bin_number=100
    for i in range(hp['Ng']):
        stat,bins,_ = scipy.stats.binned_statistic(hds,hidden_firing[:,i], statistic='mean',bins=bin_number, range=(-np.pi*180/np.pi,np.pi*180/np.pi))
        hd_curves.append(stat)
        hd_scores.append(np.corrcoef(hds, hidden_firing[:,i])[0,1])
    hd_curves = np.stack(hd_curves)


    color1='lime'
    for i in unit_idx:
        hd_polar_fig = plt.figure()
        hd_polar_fig.set_size_inches(5, 5, forward=True)
        ax = hd_polar_fig.add_subplot(1, 1, 1)
        theta = np.linspace(0, 2*np.pi, 361)
        ax = plt.subplot(1, 1, 1, polar=True)
        ax = plot_utility.style_polar_plot(ax)

        hist_1 = plt.hist(hd_curves[i])#, linewidth=1

        plt.xticks([])
        plt.yticks([])
        plt.xticks([math.radians(0), math.radians(90), math.radians(180), math.radians(270)])
        ax.plot(hist_1, color='navy', linewidth=2)
        plt.tight_layout()




        # fig = plt.figure(figsize=(3,3))
        # ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        #
        # plt.subplot(projection='polar')
        # plt.plot(hd_curves[i],'-',linewidth=2)#, linewidth=1
        #
        #
        # plt.title('unit_'+str(i),fontsize=5)
        # # ax.set_xticks([])
        # # ax.set_yticks([])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        # plt.xlabel('direction',fontsize=16)
        # plt.ylabel('Firing rate',fontsize=16)
        #
        #
        # plt.savefig(figure_path+'/hd_tune_'+str(i)+'.png')
        # fig.savefig(figure_path+'/hd_tune_'+str(i)+'.eps', format='eps', dpi=1000)
        plt.show()

def ratemap_all_hidden_units_0():
    figure_path = os.path.join(fig_path, 'ratemap_all_hidden_units/'+str(hp['seq_length_analysis']))
    mkdir_p(figure_path)

    res = 500#
    n_avg = 1#
    activations, _, _, _ = compute_ratemaps(model, trajectory_generator, hp,
                                            res=res,
                                            Ng=hp['Ng'],
                                            n_avg=n_avg)

    visualize.plot_ratemaps_0(hp, activations)

def ratemap_all_hidden_units():
    figure_path = os.path.join(fig_path, 'ratemap_all_hidden_units/'+str(hp['seq_length_analysis']))
    mkdir_p(figure_path)

    res = 200#
    n_avg = 1
    activations, _, _, _ = compute_ratemaps(model, trajectory_generator, hp,
                                            res=res,
                                            Ng=hp['Ng'],
                                            n_avg=n_avg)

    for i in range(hp['Ng']):
        if i>=3:
            sys.exit(0)
        im = activations[i,:,:]
        image = visualize.rgb(im, smooth=True)#(50,50,4)
        ############## plot hidden
        fig = plt.figure(figsize=(1, 1))
        plt.matshow(image[:,:,0])
        plt.title('unit_'+str(i))
        plt.savefig(figure_path+'/'+str(i)+'.png')
        plt.show()

def ratemap_all_hidden_units_panel():
    res = 50
    n_avg = 1

    activations, _, _, _ = compute_ratemaps(model, trajectory_generator, hp, res=res,
                                            Ng=hp['Ng'],
                                            n_avg=n_avg)

    idxs_ext_0 = np.arange(0,100,1)



    n_plot = 100
    fig0 = plt.figure(figsize=(10,10))
    ax = fig0.add_axes([0.05, 0.05, 0.9, 0.9])

    rm_fig = visualize.plot_ratemaps_panel(activations[idxs_ext_0], n_plot, smooth=True,width=10)
    plt.imshow(rm_fig)
    mkdir_p(fig_path)
    plt.title('seq_length_analysis='+ str(hp['seq_length_analysis'])+'\n'+str(hp['act_func'])+'_'+str(model_idx))
    plt.savefig(fig_path+'/hidden_unit_0_400.pdf')
    plt.axis('off')


    idxs_inh = np.arange(410,510,1)
    fig3 = plt.figure(figsize=(10,10))
    ax = fig3.add_axes([0.05, 0.05, 0.9, 0.9])
    n_plot = 100
    rm_fig = visualize.plot_ratemaps_panel(activations[idxs_inh],n_plot, smooth=True,width=10)
    plt.imshow(rm_fig)
    mkdir_p(fig_path)
    plt.title(str(model_idx)+'\n'+str('inhibitory'))
    plt.savefig(fig_path+'/hidden_unit_412_512.pdf')
    plt.axis('off')

    plt.show()

def ratemap_all_hidden_units_panel_1():
    res = 50#
    n_avg = 1

    activations, _, _, _ = compute_ratemaps(model, trajectory_generator, hp, res=res,
                                            Ng=hp['Ng'],
                                            n_avg=n_avg)

    n_plot=25
    images = visualize.plot_ratemaps_panel_me(activations=activations, smooth=True)
    fig0 = plt.figure(figsize=(10,10))
    ax = fig0.add_axes([0.1, 0.1, 0.8, 0.8])
    for i in range(25):

        plt.subplot(5,5,i+1)
        plt.axis('off')
        plt.imshow(images[i,:,:],cmap='jet')
        if i==5:
            plt.title(str(model_idx)+'\n'+'seq_length_analysis='+ str(hp['seq_length_analysis'])+'\n'+str('excitatory'),fontsize=15)

    fig0.savefig(fig_path+'/exc_unit_100'+'seq_length_analysis='+ str(hp['seq_length_analysis'])+'.pdf')


    fig1 = plt.figure(figsize=(10,10))
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    for i in range(100):
        j = i+410
        plt.subplot(10,10,i+1)
        plt.axis('off')
        plt.imshow(images[j,:,:],cmap='jet')
        if i==5:
            plt.title(str(model_idx)+'\n'+'seq_length_analysis='+ str(hp['seq_length_analysis'])+'\n'+str('inhibitory'),fontsize=15)


    plt.show()
    fig1.savefig(fig_path+'/inh_unit_100'+'seq_length_analysis='+ str(hp['seq_length_analysis'])+'.pdf')


def speed_tuning_panel_S1(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'tuning_panel_S1/')
    mkdir_p(figure_path)

    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    v = inputs[0].cpu().numpy()
    hidden_firing_batch = model.grid_hidden(inputs).detach().cpu().numpy()

    vx = v[:,:,-1]
    vy = v[:,:,-2]

    v = np.sqrt(vx**2 + vy**2)
    vs = np.stack(v).ravel()
    hidden_firing = np.reshape(hidden_firing_batch, (-1,hp['Ng']))


    v_curves = []
    v_scores = []
    bin_number=9
    for i in range(hp['Ng']):
        stat,bins,_ = scipy.stats.binned_statistic(vs,hidden_firing[:,i], statistic='mean',bins=bin_number)
        v_curves.append(stat)
        v_scores.append(np.corrcoef(vs, hidden_firing[:,i])[0,1])
    v_curves = np.stack(v_curves)


    fig = plt.figure(figsize=(5,3.8))
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    j=0
    unit_idx = np.array([1,3,5,12,20,33,38,39,42,203,224,293,342,422,464,468,476,479,483,491])
    for i in unit_idx:
        j+=1
        plt.subplot(4,5,j)
        plt.plot(v_curves[i],'-',color='black',linewidth=2)

        plt.xticks([])
        plt.yticks([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylim([0,0.06])
    fig.savefig(figure_path+'/speed_tune_panel.eps', format='eps', dpi=1000)
    plt.savefig(figure_path+'/'+'speed_tune_panel')
    plt.show()

def hd_tuning_panel_S1(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'tuning_panel_S1/')
    mkdir_p(figure_path)

    Ng=hp['Ng']

    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    v = inputs[0].cpu().numpy()
    hidden_firing_batch = model.grid_hidden(inputs).detach().cpu().numpy()

    vx = v[:,:,-1]
    vy = v[:,:,-2]

    hd = np.arctan2(vy,vx)*180/np.pi
    hidden_firing_batch = np.reshape(hidden_firing_batch, (-1,Ng))

    hidden_firing = hidden_firing_batch
    hds = hd.ravel()

    hd_curves = []
    hd_scores = []
    bin_number=10
    for i in range(hp['Ng']):
        stat,bins,_ = scipy.stats.binned_statistic(hds,hidden_firing[:,i], statistic='mean',bins=bin_number, range=(-np.pi*180/np.pi,np.pi*180/np.pi))
        hd_curves.append(stat)
        hd_scores.append(np.corrcoef(hds, hidden_firing[:,i])[0,1])
    hd_curves = np.stack(hd_curves)

    ###############################plot panel
    #"""
    fig = plt.figure(figsize=(5,3.8))
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    j=0
    for i in unit_idx:
        j+=1
        plt.subplot(4,5,j)
        plt.plot(hd_curves[i],'-',color='tab:blue',linewidth=2)

        plt.xticks([])
        plt.yticks([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylim([0,0.06])
    fig.savefig(figure_path+'/hd_tune_panel.eps', format='eps', dpi=1000)
    plt.savefig(figure_path+'/'+'hd_tune_panel')
    plt.show()

def visual_tuning_panel_S1(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'tuning_panel_S1/')
    mkdir_p(figure_path)

    Ng=hp['Ng']

    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    v = inputs[0].cpu().numpy()
    hidden_firing_batch = model.grid_hidden(inputs).detach().cpu().numpy()
    hidden_firing = np.reshape(hidden_firing_batch, (-1,Ng))


    #visual
    vis_PC1 = v[:,:,0]
    vis_PC1s = np.stack(vis_PC1).ravel()
    hidden_firing = np.reshape(hidden_firing_batch, (-1,hp['Ng']))


    # Construct head direction tuning curves
    vis_curves = []
    vis_scores = []
    bin_number=10
    unit_idx = unit_idx
    for i in range(hp['Ng']):
        stat,bins,_ = scipy.stats.binned_statistic(vis_PC1s,hidden_firing[:,i], statistic='mean',bins=bin_number)
        vis_curves.append(stat)
        vis_scores.append(np.corrcoef(vis_PC1s, hidden_firing[:,i])[0,1])
    vis_curves = np.stack(vis_curves)

    fig = plt.figure(figsize=(5,3.8))
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    j=0
    unit_idx = np.array([49,57,58,59,63,125,126,184,202,203,224,12,342,422,464,468,476,479,483,491])
    for i in unit_idx:
        j+=1
        plt.subplot(4,5,j)
        plt.plot(vis_curves[i],'-',color='tab:green',linewidth=2)

        plt.xticks([])
        plt.yticks([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylim([0,0.03])
    fig.savefig(figure_path+'/vis_tune_panel.eps', format='eps', dpi=1000)
    plt.savefig(figure_path+'/'+'vis_tune_panel')
    plt.show()

def ratemap_units_panel_S1(hp,trajectory_generator,model,fig_path,unit_idx):
    figure_path = os.path.join(fig_path, 'tuning_panel_S1/')
    mkdir_p(figure_path)
    res = 50
    n_avg = 1

    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)
    ######################################################################################################
    fig = plt.figure(figsize=(5,3.8))
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    j=0
    for idx in unit_idx:
        j+=1
        plt.subplot(4,5,j)
        im = activations[idx,:,:]
        im=visualize.rgb(im)
        plt.imshow(im[:,:,0], interpolation='none', cmap='jet');plt.axis('off')

    plt.savefig(figure_path+'/'+'ratemap_units_panel_S1')
    plt.show()

def get_high_score_idx(fig_path):
    load_data = io.loadmat(fig_path+'/'+"score_model1.mat")
    score_type = load_data['score'][0,:]

    score_exc = score_type[:410]
    idxs_exc = np.flip(np.argsort(score_exc))
    score_inh = score_type[410:]
    idxs_inh = np.flip(np.argsort(score_inh))
    idxs_all = np.flip(np.argsort(score_type))
    idxs_nozero = np.where(score_type != 0)[0]


    idxs_select = idxs_nozero

    return idxs_select

def cal_percent_speed_tuning(hp,trajectory_generator,model,fig_path,model_idx):
    figure_path = os.path.join(fig_path, 'cal_percent/')
    mkdir_p(figure_path)
    #'''

    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    v = inputs[0].cpu().numpy()
    hidden_firing_batch = model.grid_hidden(inputs).detach().cpu().numpy()

    vx = v[:,:,-1]
    vy = v[:,:,-2]

    v = np.sqrt(vx**2 + vy**2)
    vs = np.stack(v).ravel()

    hd = np.arctan2(vy,vx)*180/np.pi
    hds = hd.ravel()

    hidden_firing = np.reshape(hidden_firing_batch, (-1,hp['Ng']))


    v_curves = []
    v_scores = []
    bin_number=9
    for i in range(hp['Ng']):
        stat,bins,_ = scipy.stats.binned_statistic(vs,hidden_firing[:,i], statistic='mean',bins=bin_number)
        v_curves.append(stat)
        v_scores.append(np.corrcoef(vs, hidden_firing[:,i])[0,1])
    v_curves = np.stack(v_curves)

    hd_curves = []
    hd_scores = []
    bin_number=100
    for i in range(hp['Ng']):
        stat,bins,_ = scipy.stats.binned_statistic(hds,hidden_firing[:,i], statistic='mean',bins=bin_number, range=(-np.pi*180/np.pi,np.pi*180/np.pi))
        hd_curves.append(stat)
        hd_scores.append(np.corrcoef(hds, hidden_firing[:,i])[0,1])
    hd_curves = np.stack(hd_curves)

    np.save(figure_path+'v_curves'+str(model_idx)+'.npy',v_curves)
    np.save(figure_path+'hd_curves'+str(model_idx)+'.npy',hd_curves)

    idxs_select = get_high_score_idx(fig_path)
    v_curves = np.load(figure_path+'v_curves'+str(model_idx)+'.npy')
    hd_curves = np.load(figure_path+'hd_curves'+str(model_idx)+'.npy')

    number_both=0
    for i in idxs_select:
        if np.max(v_curves[i,:])>0.02 and np.max(hd_curves[i,:])>0.03:
            number_both+=1

    number_v=0
    for i in idxs_select:
        if np.max(v_curves[i,:])>0.02:
            number_v+=1

    number_hd=0
    for i in idxs_select:
        if np.max(hd_curves[i,:])>0.03:
            number_hd+=1
    num = idxs_select.shape[0]
    return number_both/num, number_v/num, number_hd/num
