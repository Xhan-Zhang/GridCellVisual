# -*- coding: utf-8 -*-
import torch
import numpy as np

#from visualize import save_ratemaps
import os, sys
import time
import utils
import defaults
import pdb
import datetime
#device = torch.device('cuda' if torch.cuda.is_available else 'cpu')



hp = defaults.get_default_hp()

#
class Trainer(object):
    def __init__(self, hp,place_cells, model, trajectory_generator, is_cuda=True):

        self.is_cuda = is_cuda
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.hp = hp
        self.Ng = hp['Ng']
        self.model = model
        self.place_cells = place_cells
        self.trajectory_generator = trajectory_generator

        self.weight_entory = hp['weight_entropy']
        self.weight_decay = hp['weight_decay']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hp['learning_rate'], weight_decay=self.weight_decay)

        self.model.to(self.device)
        self.max_norm = torch.tensor(1., device=self.device)



    def compute_loss(self, inputs, pc_outputs_truth, pos):

        preds_pc_output = self.model.forward_predict(inputs)

        y_hat = self.model.softmax(preds_pc_output)
        loss = -(pc_outputs_truth * torch.log(y_hat)).sum(-1).mean()

        # Weight regularization
        if self.hp['L2_norm_1']:
            L2_norm = (self.model.RNN_layer.h2h.weight**2).sum()
        else:
            L2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
        loss += self.weight_entory * L2_norm

        pred_pos = self.place_cells.get_nearest_cell_pos(preds_pc_output)

        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err

    def evaluate_performance(self):

        inputs, pos, pc_outputs = self.trajectory_generator.get_batch_for_test(batch_size=512)
        if self.is_cuda:
            inputs = [inputs[i].to(self.device) for i in range(2)]
            pc_outputs = pc_outputs.to(self.device)
            pos = pos.to(self.device)

        pred_activity = self.model.forward_predict(inputs)
        pred_pos = self.place_cells.get_nearest_cell_pos(pred_activity)

        evaluate_performance_error = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()
        evaluate_performance_error = evaluate_performance_error.cpu().numpy()


        return evaluate_performance_error





    def train_step(self, inputs, pc_outputs, pos):

            self.optimizer.zero_grad()
            loss, err = self.compute_loss(inputs, pc_outputs, pos)

            loss.backward()
            if hp['clip_grad_value']:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
            self.optimizer.step()
            self.model.RNN_layer.self_weight_clipper()

            return loss.item(), err.item()



    def print_step(self,t,n_steps,t_start, loss_value,error_value,evaluate_performance_error):
        for name, parms in self.model.named_parameters():
            print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                  ' -->grad_value:',parms.grad)



    #save trained model
    def save_final_result(model=None, model_dir=None,hp=None, log=None):
        save_path = os.path.join(model_dir, 'finalResult')
        tools.mkdir_p(save_path)
        model.save(save_path)
        log['model_dir'] = save_path
        tools.save_log(log)
        tools.save_hp(hp, save_path)



    def train(self, n_epochs=500,n_steps=100000, visual_input=True, save=True):
        '''
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''
        # Construct generator
        root_path = os.path.abspath(os.path.join(os.getcwd(),".."))
        save_dir = os.path.join(root_path, 'rnn-ei1_'+hp['act_func']+'_models_saved'+'_'+hp['visual'])
        model_dir = os.path.join(save_dir, self.hp['run_ID'])

        gen = self.trajectory_generator.get_generator_for_training_visual_img(visual_input=visual_input)

        t_start = time.time()
        self.loss_list = []
        self.err_list = []


        for t in range(n_steps):



            #load the data for training#
            inputs, pos, pc_outputs_truth = next(gen)


            # put the data on device
            inputs = [inputs[i].to(self.device) for i in range(2)]
            pc_outputs_truth = pc_outputs_truth.to(self.device)
            pos = pos.to(self.device)


            # compute the loss and error
            loss, err = self.train_step(inputs, pc_outputs_truth, pos)

            self.loss_list.append(loss)
            self.err_list.append(err)
            loss_value = np.round(loss,7)
            error_value = np.round(100*err,3)

            evaluate_performance_error = self.evaluate_performance()
            evaluate_performance_error = np.round(100*evaluate_performance_error,2)



            if t%n_epochs==0:

                self.print_step(t,n_steps,t_start,loss_value,error_value,evaluate_performance_error)

                if t==n_steps/2 or t==n_steps/4 or t==n_steps/6 or t==n_steps/8 or t==n_steps/10:

                    self.model.save(model_dir)

                if (8.5 <= error_value<=9) or (7.8<=error_value<=8) or (6.8<=error_value<=7)or (5.8<=error_value<=6):

                    self.model.save(model_dir)


            if (error_value<=self.hp['critic_value'] and evaluate_performance_error<=self.hp['critic_value']) or t>=n_steps-1:

                now_time = datetime.datetime.now()
                datetime.datetime.now().strftime('%Y-%m-%d')
                print('now_time',now_time)


                self.model.save(model_dir)


                log = dict()
                log['model_dir']=model_dir
                log['elapsed_time'] = utils.elapsed_time(time.time() - t_start)
                log['loss_value'] = loss_value
                log['error_value'] = error_value
                log['evaluate_performance_error'] = evaluate_performance_error
                utils.save_log(log)

                print("Optimization finished!")
                print('run_ID:  ',self.hp['run_ID'])

                break




















