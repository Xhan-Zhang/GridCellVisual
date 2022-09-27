import numpy as np
import os


def get_default_hp(random_seed=None):
    if random_seed is None:
        seed = np.random.randint(10000)
    else:
        seed = random_seed

    rng = np.random

    root_path = os.path.abspath(os.path.join(os.getcwd(),"../.."))
    save_dir = os.path.join(root_path, 'models_saved')
    figure_dir = os.path.join(root_path, 'z_figure')
    root_path_1 = os.path.abspath(os.path.join(os.getcwd(),"."))
    picture_dir = os.path.join(root_path_1, 'picture3/')


    hp = {
        'picture_dir':picture_dir,
        'figure_dir': figure_dir,
        'save_dir': save_dir,
        'n_epochs':200,
        'sequence_length':20,
        'Np':1024,
        'place_cell_rf':0.12,
        'surround_scale':2,


        'RNN_type': 'RNN',
        'activation':'relu',
        'activation_RNN':'relu',
        'DoG':True,
        'periodic': False,
        'box_width':2.2,
        'box_height':2.2,
        'seed': seed,
        'rng': rng,
        'is_cuda':False,
        'sigma_rec': 0.0,#0.05
        'initial_std': 0.0, #   0.3
        'alpha': 1,
        'n_steps':1000000,

        'l1_weight': 0.0,
        'l2_weight': 0.0,

        'learning_rate':0.0005,
        'Ng':512,
        'weight_entropy':1e-4,
        'weight_decay':0.0,
        'L2_norm_1':True,
        'clip_grad_value':True,
        'critic_value':5,
        'input_size':2+20,

        'n_components':20,
        'activation_rnn':'relu',
        'weight_perturb': 'no_perturb',
        'e_prop':0.8,

        'speed_scale':1,
        'speed_var':0.1,

        'batch_size':256,
        'seq_length_model':10,

        'act_func':'relu',
        'vis_input':True,
        'visual':'visual',
        'batch_size_test':5000,
        'seq_length_analysis':10,

        'use_relu_grid':False,
        'model_idx':1,
        'rgb':False


    }

    return hp



class Key_to_value:
    def __init__(self,dict1):
        for key, value in dict1.items():
            self.__dict__[key] = value






