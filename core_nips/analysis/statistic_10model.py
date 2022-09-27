import sys, os
import numpy as np
import cv2
import ratemaps
from scipy import io
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../")))

from core_nips.utils import generate_run_ID
from core_nips.defaults import get_default_hp,Key_to_value
from core_nips.place_cells import PlaceCells
from core_nips.trajectory_generator import TrajectoryGenerator
from core_nips.model import Network
import matplotlib.pyplot as plt




#load parames
hp = get_default_hp(random_seed=10000)
arg = Key_to_value(hp)

hp['Np'] = 1024
hp['rng'] = np.random.RandomState(1)
hp['run_ID'] = generate_run_ID(arg)
hp['sequence_length'] = hp['seq_length_analysis']


root_path = os.path.abspath(os.path.join(os.getcwd(),"../.."))
save_dir = os.path.join(root_path, hp['act_func']+'_'+hp['visual'])


place_cells = PlaceCells(hp)
img=cv2.imread('../img/0.png')
new_pca_matrix = np.load("../img/pca_matrix_img0.npy")

figure_dir = os.path.join(root_path, 'z_figure')
fig_dir = os.path.join(figure_dir, hp['act_func']+'_'+hp['visual'])


def plot_percent_grid():

    percentage_E_grids = []
    percentage_I_grids = []
    for idx in range(1,11,1):
        hp['model_idx'] = idx

        model_idx = str(512)+'_'+str(1024)+'_sl_'+str(hp['seq_length_model'])+'_neg_'+str(hp['neg']+'_model_'+str(hp['model_idx']))
        fig_path = os.path.join(fig_dir, model_idx)
        model_dir = os.path.join(save_dir, model_idx)

        trajectory_generator = TrajectoryGenerator(hp=hp, place_cells=place_cells, img=img,new_pca_matrix =new_pca_matrix)

        model = Network(hp)
        model.load_model(model_dir)

        percentage_E_grid, percentage_I_grid = ratemaps.Calculate_percent_grid_ext_inh(hp,trajectory_generator,model,fig_path=fig_path)
        percentage_E_grids.append(percentage_E_grid)
        percentage_I_grids.append(percentage_I_grid)

    print('percentage_E_grids:',percentage_E_grids)
    print('percentage_I_grids:',percentage_I_grids)
    io.savemat(fig_dir+'/'+"percentage_grids.mat",{'percentage_E_grids':percentage_E_grids,
                                                   'percentage_I_grids':percentage_I_grids})

def plot_score_distribute():
    xs_Es_list=[];ys_Es_list=[];xs_Is_list=[];ys_Is_list=[]



    for idx in range(1,11,1):
        hp['model_idx'] = idx

        model_idx = str(512)+'_'+str(1024)+'_sl_'+str(hp['seq_length_model'])+'_model_'+str(hp['model_idx'])
        fig_path = os.path.join(fig_dir, model_idx)
        model_dir = os.path.join(save_dir, model_idx)

        trajectory_generator = TrajectoryGenerator(hp=hp, place_cells=place_cells, img=img,new_pca_matrix =new_pca_matrix)

        model = Network(hp)
        model.load_model(model_dir)

        xs_E,ys_E,xs_I,ys_I = ratemaps.Plot_score_distribute_ext_inh(hp,trajectory_generator,model,fig_path)
        xs_Es_list.append(xs_E)
        ys_Es_list.append(ys_E)
        xs_Is_list.append(xs_I)
        ys_Is_list.append(ys_I)
    #xs_Es_list = np.concatenate(xs_Es, axis=0)
    #print('xs_Es_list',xs_Es_list.shape)
    ys_Es_mean = np.mean(ys_Es_list, axis=0)
    ys_Es_sem = np.std(ys_Es_list, axis=0)/np.sqrt(len(ys_Es_list))

    ys_Is_mean = np.mean(ys_Is_list, axis=0)
    ys_Is_sem = np.std(ys_Is_list, axis=0)/np.sqrt(len(ys_Is_list))

    # fig = plt.figure(figsize=(4,3))
    # ax = fig0.add_axes([0.2, 0.2, 0.7, 0.7])
    # fig = plt.figure(figsize=(3, 2.5))
    # ax = fig.add_axes([0.25, 0.25, 0.6, 0.65])
    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_axes([0.25, 0.2, 0.7, 0.68])

    number_dot = ys_Es_mean.shape[0]
    fs = 12

    plt.plot(np.linspace(-0.3,1.2,number_dot), ys_Es_mean, color='red',label="E-unit")
    plt.fill_between(np.linspace(-0.3,1.2,number_dot), ys_Es_mean-ys_Es_sem, ys_Es_mean+ys_Es_sem, color='red', alpha=0.2)

    plt.plot(np.linspace(-0.3,1.2,number_dot), ys_Is_mean, color='blue',label="I-unit")
    plt.fill_between(np.linspace(-0.3,1.2,number_dot), ys_Is_mean-ys_Es_sem, ys_Is_mean+ys_Is_sem, color='blue', alpha=0.2)
    plt.xlabel('Grid Score',fontsize=fs)
    plt.ylabel('Percent',fontsize=fs)
    plt.legend()
    plt.title('#2')
    fig.savefig(fig_dir+'/plot_score_distribute#2.eps', format='eps', dpi=1000)
    fig.savefig(fig_dir+'/plot_score_distribute#2.png')

    plt.show()



plot_score_distribute()


