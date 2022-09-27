import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
from matplotlib import pyplot as plt
import argparse
import cv2
import numpy as np

from utils   			  import generate_run_ID
from model 	 			  import Network
from trainer 			  import Trainer
from defaults             import get_default_hp
from place_cells          import PlaceCells
from trajectory_generator import TrajectoryGenerator


parser = argparse.ArgumentParser()
parser.add_argument('--n_steps',    		 type=int,	default=1000000,				help='batches per epoch')
parser.add_argument('--RNN_type',			 type=str,	default='RNN',					help='RNN or LSTM')
parser.add_argument('--is_cuda', 			 action='store_true',default=False,			help='whether gpu')

parser.add_argument('--activation',			 type=str,	default='relu',					help='recurrent nonlinearity')
parser.add_argument('--activation_RNN',		 type=str,	default='relu',					help='recurrent nonlinearity')
parser.add_argument('--weight_entropy','-we', type=float, default=0.0001,				help='strength of weight decay on recurrent weights')

parser.add_argument('--weight_decay','-wd', type=float, default=0.0,				help='strength of weight decay on recurrent weights')
parser.add_argument('--Np',  				 type=int,	default=1024,					help='number of place cells')
parser.add_argument('--Ng',					 type=int,	default=512,					help='number of grid cells')
parser.add_argument('--sequence_length','-sl',	 type=int,	default=10,						help='number of steps in trajectory')
parser.add_argument('--batch_size', '-bs',	 type=int,	default=256,					help='number of trajectories per batch')
parser.add_argument('--learning_rate','-lr', type=float,	default=0.0005,					help='gradient descent learning rate')
parser.add_argument('--model_idx','-idx', type=int,	default=1,					help='we train 10 model')



def train_model(hp=None,is_cuda=True):

	place_cells = PlaceCells(hp,is_cuda=is_cuda)
	img=cv2.imread('0.png')
	new_pca_matrix = np.load("pca_matrix_img0.npy")
	trajectory_generator = TrajectoryGenerator(hp, place_cells,img=img,new_pca_matrix =new_pca_matrix,is_cuda=is_cuda)

	model = Network(hp)
	#print('model',model)

	trainer = Trainer(hp, place_cells, model, trajectory_generator,is_cuda=is_cuda)
	trainer.train(n_steps=hp['n_steps'], visual_input=hp['vis_input'], save=True)


	root_path = os.path.abspath(os.path.join(os.getcwd(),".."))
	#save_dir = os.path.join(root_path, hp['activation_rnn']+'_models_saved'+hp['visual']+str('g'))
	save_dir = os.path.join(root_path, 'rnn-ei1_'+hp['act_func']+'_models_saved'+'_'+hp['visual'])
	figure_path = os.path.join(save_dir, hp['run_ID']+'/')


	plt.figure(figsize=(12,3))
	plt.subplot(121)
	plt.plot(trainer.err_list, c='black')
	plt.savefig(figure_path+'error.png')

	plt.title('Decoding error (m)'); plt.xlabel('train step')
	plt.subplot(122)
	plt.plot(trainer.loss_list, c='black')
	plt.title('Loss'); plt.xlabel('train step')
	plt.savefig(figure_path+'loss.png')#






#load parames

if __name__ == "__main__":
	arg = parser.parse_args()
	hp = get_default_hp()
	hp['n_steps'] = arg.n_steps
	hp['sequence_length'] = arg.sequence_length
	hp['Ng'] = arg.Ng
	hp['Np'] = arg.Np
	hp['learning_rate'] = arg.learning_rate
	hp['is_cuda'] = arg.is_cuda
	hp['activation_RNN']=arg.activation_RNN
	hp['batch_size'] = arg.batch_size
	hp['weight_decay'] = arg.weight_decay
	hp['model_idx'] = arg.model_idx



	root_path = os.path.abspath(os.path.join(os.getcwd(),"../"))
	save_dir = os.path.join(root_path, 'models_saved')
	hp['save_dir'] = save_dir
	hp['run_ID'] = generate_run_ID(arg)



	# Display hp
	for key, val in hp.items():
		print('{:20s} = '.format(key) + str(val))

	train_model(hp=hp,is_cuda=hp['is_cuda'])










