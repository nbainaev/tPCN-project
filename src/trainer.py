# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F

import os
from tqdm import tqdm

from src.data_utils.preload_data import *
import src.utils
import wandb

import os
from tqdm import tqdm

from src.data_utils.preload_data import *
import src.utils

class BaselineTrainer(object):
    def __init__(self, model, env, options):
        self.model = model
        self.options = options
        self.traj_gen = TrajectoryGenerator(env, options)
        self.acc = {'mean': [], 'std': []}
        self.acc_eval = {'mean': [], 'std': []}
        self.acc_time = {'epoch': [], 'acc': {'mean': [], 'std': []}}
        self.acc_time_eval = {'epoch': [], 'acc': {'mean': [], 'std': []}}
        self.loss = None
        self.energy = None
        self.n_epochs = options['n_epochs']
    def train(self, ini_pos=None, save=True):

        if self.options['use_preloaded']:
            dpath = self.options['data_path'] if self.options['data_path'] is not None else 'data'
            path = os.path.join(
                dpath, 
                f'{self.options['mode']}_{self.options['batch_size']*self.options['n_steps']}_{self.options['sequence_length']}_{self.options['obs_size']}_encode=False'
            )
            # check if the file exists
            if os.path.exists(path):
                print(f'Loading pre-generated data at {path}...')
            else:
                print(f'Generating new data at {path}...')
                self.traj_gen.generate_traj_data(ini_pos, path=path, encode=False)

            dataloader = get_traj_loader(path, self.options)

        # if self.options.is_wandb == True and self.options.sweep == False:
        #     wandb.init(project='place-cell-tpc', config=self.options)
        for epoch_idx in range(self.options['n_epochs']):
            epoch_acc = {'mean': [], 'std': []}
            if self.options['evaluation']:
                epoch_acc_eval = {'mean': [], 'std': []}
            iterable = dataloader if self.options['use_preloaded'] else range(self.options['n_steps'])
            tbar = tqdm(iterable, leave=True)
            for item in tbar:
                if self.options['use_preloaded']:
                    inputs, pc_outputs, init_actv = item
                    inputs = inputs.numpy().squeeze().astype(int)
                    pc_outputs = pc_outputs.numpy().astype(int)
                    init_actv = init_actv.numpy().squeeze().astype(int)
                else:
                    ini_pos = self.options['train_with_ini_pos'](self.options['room'], self.options['batch_size']) if self.options['train_with_ini_pos'] else ini_pos
                    inputs, pc_outputs, init_actv = self.traj_gen.generate_traj_data(ini_pos, encode=False, save=False)
                    pc_outputs =  pc_outputs.numpy().astype(int)
                    inputs = inputs.numpy().squeeze().astype(int)
                    init_actv = init_actv.numpy().squeeze().astype(int)
                if self.options['mode'] == 'pomdp':
                    pc_outputs = pc_outputs.squeeze(3)
                if self.options['model'] == 'chmm':
                    if self.options['mode'] == 'pomdp':
                        obs_list = np.concatenate((init_actv[:, np.newaxis], pc_outputs.squeeze(2)), axis=1)
                    else:
                        obs_list = np.concatenate((init_actv[:, np.newaxis, :], pc_outputs), axis=1)
                    self.model.observe_sequence(obs_list, inputs)
                    pred_pos = np.array(self.model.predict_sequence(inputs, pc_outputs.squeeze(), init_actv))[:, 1:, :]
                else:
                    self.model.observe_sequence(pc_outputs, inputs, init_actv)
                    pred_pos = self.model.predict_sequence(inputs, pc_outputs, init_actv)

                acc = np.all(pc_outputs == pred_pos, axis=2)
                if epoch_idx + self.options['collect_acc_last'] >= self.n_epochs:
                    if not len(self.acc_time['acc']['mean']):
                        self.acc_time['acc']['mean'] = acc.mean(0)
                        acc_time_batch = np.array([arr.mean(0) for arr in np.array_split(acc, self.options['aggregation_points'])])
                        self.acc_time['acc']['std'] = acc_time_batch
                    else:
                        self.acc_time['acc']['mean'] += acc.mean(0)
                        acc_time_batch = np.array([arr.mean(0) for arr in np.array_split(acc, self.options['aggregation_points'])])
                        self.acc_time['acc']['std'] = np.concatenate((self.acc_time['acc']['std'], acc.mean(0).reshape((1, -1))), axis=0) 
    
                acc = acc.mean(-1)
                acc_batch = [arr.mean(-1) for arr in np.array_split(acc, self.options['aggregation_points'])]
                
                epoch_acc['mean'] += [float(acc.mean())]
                epoch_acc['std'] += [acc_batch]
                
                is_eval = self.options['evaluation'] and epoch_idx % self.options['eval_every'] == 0
                
                if is_eval:
                    ini_pos_eval = self.options['validate_with_ini_pos'](self.options['room'], self.options['batch_size']) if self.options['validate_with_ini_pos'] else ini_pos
                    inputs, pc_outputs, init_actv = self.traj_gen.generate_traj_data(ini_pos_eval, save=False, encode=False)
                    inputs = inputs[:self.options['batch_size']].numpy().squeeze().astype(int)
                    if self.options['mode'] == 'pomdp':
                        pc_outputs = pc_outputs[:self.options['batch_size']].numpy().squeeze(3).astype(int)
                    else:
                        pc_outputs = pc_outputs[:self.options['batch_size']].numpy().astype(int)
                    init_actv = init_actv[:self.options['batch_size']].numpy().squeeze().astype(int)
                    if self.options['model'] == 'chmm':
                        pred_pos = np.array(self.model.predict_sequence(inputs, pc_outputs.squeeze(), init_actv))[:, 1:, :]
                    else:
                        pred_pos = self.model.predict_sequence(inputs, pc_outputs, init_actv)
                    #pc_outputs_decoded = self.traj_gen.decode_trajectory(pc_outputs)
                    # pred_pos = self.traj_gen.decode_trajectory(pred_xs)
                    acc_eval = np.all(pc_outputs == pred_pos, axis=2)
                    if epoch_idx + self.options['eval_every'] * self.options['collect_acc_last'] >= self.n_epochs:
                        if not len(self.acc_time_eval['acc']['mean']):
                            self.acc_time_eval['acc']['mean'] = acc_eval.mean(0)
                            acc_time_batch_eval = np.array([arr.mean(0) for arr in np.array_split(acc_eval, self.options['aggregation_points'])])
                            self.acc_time_eval['acc']['std'] = acc_time_batch_eval
                        else:
                            self.acc_time_eval['acc']['mean'] += acc_eval.mean(0)
                            acc_time_batch_eval = np.array([arr.mean(0) for arr in np.array_split(acc_eval, self.options['aggregation_points'])])
                            self.acc_time_eval['acc']['std'] = np.concatenate((self.acc_time_eval['acc']['std'], acc_time_batch_eval), axis=0)
                    acc_eval = acc_eval.mean(-1)
                    acc_batch_eval = [arr.mean(-1) for arr in np.array_split(acc_eval, self.options['aggregation_points'])]
                    epoch_acc_eval['mean'] += [acc_eval.mean()]
                    epoch_acc_eval['std'] += [acc_batch_eval]
                desc = f'Epoch: {epoch_idx+1}/{self.options['n_epochs']}. Acc: {np.round(100 * acc.mean(), 2)}.'
                desc += f'Acc eval: {np.round(100 * acc_eval.mean(), 2)}' if is_eval else ''
                tbar.set_description(desc)

            self.acc['mean'].append(np.array(epoch_acc['mean']).mean())
            self.acc['std'].append(np.array(epoch_acc['std']).std())
            if is_eval:
                self.acc_eval['mean'].append(np.array(epoch_acc_eval['mean']).mean())
                self.acc_eval['std'].append(np.array(epoch_acc_eval['std']).std())
        
        self.acc_time['acc']['mean'] /= self.options['collect_acc_last']
        self.acc_time['acc']['std'] = self.acc_time['acc']['std'].std(0)
        self.acc_time_eval['acc']['mean'] /= self.options['collect_acc_last']
        self.acc_time_eval['acc']['std'] = self.acc_time_eval['acc']['std'].std(0)
        tbar.close()


class PCTrainer(object):
    def __init__(self, options, model, init_model, env):
        self.options = options
        self.model = model
        self.init_model = init_model
        self.lr = options['learning_rate']
        self.inf_iters = options['inf_iters']
        self.test_inf_iters = options['test_inf_iters']
        self.inf_lr = options['inf_lr']
        self.n_epochs = options['n_epochs']
        self.n_steps = options['n_steps']
        self.restore = None
        self.traj_gen = TrajectoryGenerator(env, options)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr,
        )
        self.init_optimizer = torch.optim.Adam(
            self.init_model.parameters(),
            lr=self.lr,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=options['decay_step_size'], 
            gamma=options['decay_rate'],
        )

        self.loss = []
        self.acc = {'mean': [], 'std': []}
        self.energy = []
        self.acc_eval = {'mean': [], 'std': []}
        self.acc_time = {'epoch': [], 'acc': {'mean': [], 'std': []}}
        self.acc_time_eval = {'epoch': [], 'acc': {'mean': [], 'std': []}}
        # Set up checkpoints when not tuning hyperparameters
        if options['sweep'] == False:
            self.ckpt_dir = os.path.join(options.save_dir, 'models')
            ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            
            # when restoring pre-trained models
            if self.restore is not None:
                restore_path = os.path.join(
                    os.path.join("./results/tpc", self.restore),
                    "models", 
                    "most_recent_model.pth"
                )
                restore_ckpt = torch.load(restore_path)
                self.model.load_state_dict(restore_ckpt["model"])
                self.init_model.load_state_dict(restore_ckpt["init_model"])
                self.init_optimizer.load_state_dict(restore_ckpt['init_optimizer'])
                self.optimizer.load_state_dict(restore_ckpt['optimizer'])
                self.scheduler.load_state_dict(restore_ckpt['scheduler'])
                print(f"Restored trained model from {self.restore}")
            else:
                print("Initializing new model from scratch.")

    def train_step(self, vs, pc_outputs, init_actv):
        ''' 
        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].
        '''
        self.model.train()
        self.init_model.train()
        total_loss = 0 # average loss across time steps
        total_energy = 0 # average energy across time steps

        # train the initial static pcn to get the initial hidden state
        self.init_optimizer.zero_grad()
        self.init_model.inference(self.inf_iters, self.inf_lr, init_actv)
        energy, obs_loss = self.init_model.get_energy()
        energy.backward()
        self.init_optimizer.step()
        
        total_loss += obs_loss.item()
        total_energy += energy.item()
        # get the initial hidden state from the initial static model
        prev_hidden = self.init_model.z.clone().detach()
        for k in range(self.options['sequence_length']):
            p = pc_outputs[:, k].to(self.options['device'])
            v = vs[:, k].to(self.options['device'])
            self.optimizer.zero_grad()
            self.model.inference(self.inf_iters, self.inf_lr, v, prev_hidden, p)
            energy, obs_loss = self.model.get_energy()
            energy.backward()
            self.optimizer.step()

            # update the hidden state
            prev_hidden = self.model.z.clone().detach()

            # add up the loss value at each time step
            total_loss += obs_loss.item()
            total_energy += energy.item()

        return total_energy / (self.options['sequence_length'] + 1), total_loss / (self.options['sequence_length'] + 1)

    def train(self, ini_pos=None, save=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            preloaded_data: If true, load pre-generated data from file.
            save: If true, save a checkpoint after each epoch.
        '''

        if self.options['use_preloaded']:
            dpath = self.options['data_path'] if self.options['data_path'] is not None else 'data'
            path = os.path.join(
                dpath, 
                f'{self.options['mode']}_{self.options['batch_size']*self.options['n_steps']}_{self.options['sequence_length']}_{self.options['obs_size']}'
            )
            # check if the file exists
            if os.path.exists(path):
                print(f'Loading pre-generated data at {path}...')
            else:
                print(f'Generating new data at {path}...')
                self.traj_gen.generate_traj_data(ini_pos, path=path)

            dataloader = get_traj_loader(path, self.options)

        for epoch_idx in range(self.n_epochs):
            epoch_loss = 0
            epoch_energy = 0
            epoch_acc = {'std': [], 'mean': []}
            
            if self.options['evaluation']:
                epoch_acc_eval = {'std': [], 'mean': []}
            iterable = dataloader if self.options['use_preloaded'] else range(self.n_steps)
            tbar = tqdm(iterable, leave=True)
            for item in tbar:
                if self.options['use_preloaded']:
                    inputs, pc_outputs, init_actv = item
                else:
                    ini_pos = self.options['train_with_ini_pos'](self.options['room'], self.options['batch_size']) if self.options['train_with_ini_pos'] else ini_pos
                    inputs, pc_outputs, init_actv = self.traj_gen.generate_traj_data(ini_pos, save=False)
                energy, loss = self.train_step(inputs, pc_outputs, init_actv)
                pred_xs, _ = self.predict(inputs, init_actv)

                # this softmax intends to find the highest activities among neurons,
                # whcih is different from a nonlinearity
                if not isinstance(self.model.out_activation, src.utils.Softmax):
                    pred_xs = F.softmax(pred_xs, dim=-1)

                pc_outputs_decoded = self.traj_gen.decode_trajectory(pc_outputs)
                pred_pos = self.traj_gen.decode_trajectory(pred_xs)
                
                acc = np.all(
                    pc_outputs_decoded == pred_pos, 
                    axis=2
                )
                if epoch_idx + self.options['collect_acc_last'] >= self.n_epochs:
                    if not len(self.acc_time['acc']['mean']):
                        self.acc_time['acc']['mean'] = acc.mean(0)
                        acc_time_batch = np.array([arr.mean(0) for arr in np.array_split(acc, self.options['aggregation_points'])])
                        self.acc_time['acc']['std'] = acc_time_batch
                    else:
                        self.acc_time['acc']['mean'] += acc.mean(0)
                        acc_time_batch = np.array([arr.mean(0) for arr in np.array_split(acc, self.options['aggregation_points'])])
                        self.acc_time['acc']['std'] = np.concatenate((self.acc_time['acc']['std'], acc.mean(0).reshape((1, -1))), axis=0) 
                acc = acc.mean(-1)
                acc_batch = [arr.mean(-1) for arr in np.array_split(acc, self.options['aggregation_points'])]

                epoch_acc['mean'] += [float(acc.mean())]
                epoch_acc['std'] += [acc_batch]

                epoch_loss += loss
                epoch_energy += energy
                is_eval = self.options['evaluation'] and epoch_idx % self.options['eval_every'] == 0
                
                if is_eval:
                    ini_pos_eval = self.options['validate_with_ini_pos'](self.options['room'], self.options['batch_size']) if self.options['validate_with_ini_pos'] else ini_pos
                    inputs, pc_outputs, init_actv = self.traj_gen.generate_traj_data(ini_pos_eval, save=False)
                    inputs = inputs[:self.options['batch_size']]
                    pc_outputs = pc_outputs[:self.options['batch_size']]
                    init_actv = init_actv[:self.options['batch_size']]
                    pred_xs, _ = self.predict(inputs, init_actv)
                    
                    if not isinstance(self.model.out_activation, src.utils.Softmax):
                        pred_xs = F.softmax(pred_xs, dim=-1)

                    pc_outputs_decoded = self.traj_gen.decode_trajectory(pc_outputs)
                    pred_pos = self.traj_gen.decode_trajectory(pred_xs)
                    
                    acc_eval = np.all(
                        pc_outputs_decoded == pred_pos, 
                        axis=2
                    )
                    if epoch_idx + self.options['eval_every'] * self.options['collect_acc_last'] >= self.n_epochs:
                        if not len(self.acc_time_eval['acc']['mean']):
                            self.acc_time_eval['acc']['mean'] = acc_eval.mean(0)
                            acc_time_batch_eval = np.array([arr.mean(0) for arr in np.array_split(acc_eval, self.options['aggregation_points'])])
                            self.acc_time_eval['acc']['std'] = acc_time_batch_eval
                        else:
                            self.acc_time_eval['acc']['mean'] += acc_eval.mean(0)
                            acc_time_batch_eval = np.array([arr.mean(0) for arr in np.array_split(acc_eval, self.options['aggregation_points'])])
                            self.acc_time_eval['acc']['std'] = np.concatenate((self.acc_time_eval['acc']['std'], acc_time_batch_eval), axis=0)
                    acc_eval = acc_eval.mean(-1)
                    acc_batch_eval = [arr.mean(-1) for arr in np.array_split(acc_eval, self.options['aggregation_points'])]
                    epoch_acc_eval['mean'] += [acc_eval.mean()]
                    epoch_acc_eval['std'] += [acc_batch_eval]
                desc = f'Epoch: {epoch_idx+1}/{self.n_epochs}. Loss: {np.round(loss, 4)}. PC Energy: {np.round(energy, 4)}. Acc: {np.round(100 * acc.mean(), 2)}.'
                desc += f'Acc eval: {np.round(100 * acc_eval.mean(), 2)}' if is_eval else ''
                tbar.set_description(desc)

            self.loss.append(epoch_loss / self.n_steps)
            self.acc['mean'].append(np.array(epoch_acc['mean']).mean())
            self.acc['std'].append(np.array(epoch_acc['std']).std())
            if is_eval:
                self.acc_eval['mean'].append(np.array(epoch_acc_eval['mean']).mean())
                self.acc_eval['std'].append(np.array(epoch_acc_eval['std']).std())
            self.energy.append(epoch_energy / self.n_steps)

            # Update learning rate
            self.scheduler.step()

        self.acc_time['acc']['mean'] /= self.options['collect_acc_last']
        self.acc_time['acc']['std'] = self.acc_time['acc']['std'].std(0)
        self.acc_time_eval['acc']['mean'] /= self.options['collect_acc_last']
        self.acc_time_eval['acc']['std'] = self.acc_time_eval['acc']['std'].std(0)
        tbar.close()


    def predict(self, vs, init_actv):
        self.model.eval()
        self.init_model.eval()
        pred_zs = []
        with torch.no_grad():
            self.init_model.inference(self.test_inf_iters, self.inf_lr, init_actv.to(self.options['device']))
            prev_hidden = self.init_model.z.clone().detach()
            for k in range(self.options['sequence_length']):
                v = vs[:, k].to(self.options['device'])
                prev_hidden = self.model.g(v, prev_hidden)
                pred_zs.append(prev_hidden)

            pred_zs = torch.stack(pred_zs, dim=1) # [batch_size, sequence_length, Ng]
            pred_xs = self.model.decode(pred_zs)
            
        return pred_xs, pred_zs