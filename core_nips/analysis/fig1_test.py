import sys, os
import numpy as np
import cv2
import fig1_lib
from scipy import io
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../")))

from core_nips.utils import generate_run_ID
from core_nips.defaults import get_default_hp,Key_to_value
from core_nips.place_cells import PlaceCells
from core_nips.trajectory_generator import TrajectoryGenerator
from core_nips.model import Network




#load parames
hp = get_default_hp(random_seed=10000)
hp['use_act_fcn'] = False
arg = Key_to_value(hp)

hp['rng'] = np.random.RandomState(1)
hp['run_ID'] = generate_run_ID(arg)

root_path = os.path.abspath(os.path.join(os.getcwd(),"../.."))
figure_dir = os.path.join(root_path, 'z_figure')


# img

img=cv2.imread('../img/0.png')
new_pca_matrix = np.load("../img/pca_matrix_img0.npy")

def get_model(model_sl=10,sl=10,model_idx=1,act='relu'):
    hp['act_func'] = act
    hp['seq_length_analysis']=sl
    hp['sequence_length'] = hp['seq_length_analysis']
    hp['batch_size_test']=int(80000/sl)
    hp['model_idx'] = model_idx
    hp['seq_length_model']=model_sl
    save_dir = os.path.join(root_path, hp['act_func']+'_'+hp['visual'])
    fig_dir = os.path.join(figure_dir, hp['act_func']+'_'+hp['visual'])

    place_cells = PlaceCells(hp)
    trajectory_generator = TrajectoryGenerator(hp=hp, place_cells=place_cells, img=img,
                                               new_pca_matrix =new_pca_matrix)

    model_idx = str(512)+'_'+str(1024)+'_sl_'+str(hp['seq_length_model'])+'_model_'+str(hp['model_idx'])
    model_dir = os.path.join(save_dir, model_idx)
    fig_path = os.path.join(fig_dir, model_idx+'/fig1')

    model = Network(hp)
    model.load_model(model_dir)
    get_score = fig1_lib.calculate_score(hp,trajectory_generator,model,fig_path)
    return hp,trajectory_generator,model,fig_path

#============================= tuning =============================
def Plot_speed_tuning():
    unit_idx = np.array([91,150,11,111])#range(512)
    hp,trajectory_generator,model,fig_path = get_model(sl=10,model_idx=1,act='relu')
    #fig3_lib.ratemap_units(hp,trajectory_generator,model,fig_path,unit_idx)

    fig1_lib.speed_tuning(hp,trajectory_generator,model,fig_path,unit_idx)

#Plot_speed_tuning()

def Plot_hd_tuning():
    unit_idx = np.array([91,150,11,111])
    hp,trajectory_generator,model,fig_path = get_model(sl=10,model_idx=1,act='relu')
    #fig3_lib.ratemap_units(hp,trajectory_generator,model,fig_path,unit_idx)
    fig1_lib.hd_tuning(hp,trajectory_generator,model,fig_path,unit_idx)

#Plot_hd_tuning()

def Plot_speed_hd_ratemap():
    unit_idx = np.array([91,150,11,111])
    hp,trajectory_generator,model,fig_path = get_model(sl=10,model_idx=1,act='relu')
    fig1_lib.Speed_hd_ratemap(hp,trajectory_generator,model,fig_path,unit_idx)

#Plot_speed_hd_ratemap()



def plot_ratemap(speed_scale=1,model_idx = 1):
    speed_scale=speed_scale
    model_idx = model_idx
    hp,trajectory_generator,model,fig_path = get_model(sl=10,model_idx=model_idx,
                                                       act='relu')

    #fig_path = '/Users/xiaohanzhang/PycharmProjects/EIRNN-grid_visual_9.22 /z_figure/relu_visual/figure3/'

    fig1_lib.ratemap_units(hp,trajectory_generator,model,fig_path,unit_idx=[91])#,184
#plot_ratemap()


#============================= S1 =============================
#============================= S1 =============================
#============================= S1 =============================
#============================= S1 =============================

def S1_speed_tuning():
    unit_idx = range(512)#np.array(unit_idx_ratemap)
    hp,trajectory_generator,model,fig_path = get_model(sl=10,model_idx=1,act='relu')
    # fig1_lib.hd_tuning_panel_S1(hp,trajectory_generator,model,fig_path,unit_idx)
    # fig1_lib.speed_tuning_panel_S1(hp,trajectory_generator,model,fig_path,unit_idx)
    # fig1_lib.visual_tuning_panel_S1(hp,trajectory_generator,model,fig_path,unit_idx)
    #fig1_lib.ratemap_units_panel_S1(hp,trajectory_generator,model,fig_path,unit_idx)
    #fig1_lib.ratemap_units(hp,trajectory_generator,model,fig_path,unit_idx)
    #fig1_lib.speed_tuning_S1(hp,trajectory_generator,model,fig_path,unit_idx)
    fig1_lib.visual_tuning2(hp,trajectory_generator,model,fig_path,unit_idx)

    #fig1_lib.hd_tuning_S1(hp,trajectory_generator,model,fig_path,unit_idx)
    #fig1_lib.Speed_hd_ratemap(hp,trajectory_generator,model,fig_path,unit_idx)
S1_speed_tuning()



def calculate_speed_tuning():

    per_boths = []
    per_vs = []
    per_hds = []

    # for i in range(10):
    #     model_idx=i
    #     hp,trajectory_generator,model,fig_path = get_model(sl=10,model_idx=model_idx,act='relu')
    #     per_both, per_v, per_hd = fig1_lib.cal_percent_speed_tuning(hp,trajectory_generator,model,fig_path,model_idx)
    #     per_boths.append(per_both)
    #     per_vs.append(per_v)
    #     per_hds.append(per_hd)
    # np.save('per_both'+'.npy',per_boths)
    # np.save('per_v'+'.npy',per_vs)
    # np.save('per_hd'+'.npy',per_hds)

    per_boths = np.load('per_both'+'.npy')
    per_vs = np.load('per_v'+'.npy')
    per_hds = np.load('per_hd'+'.npy')

    mean_boths = np.mean(per_boths)
    mean_vs =np.mean(per_vs)
    mean_hds =np.mean(per_hds)



calculate_speed_tuning()