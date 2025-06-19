# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F

import os
from tqdm import tqdm

from src.data_utils.preload_data import *
import src.utils
import wandb

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
        self.acc = []
        self.energy = []
        self.acc_eval = []

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

        # if self.options.is_wandb == True and self.options.sweep == False:
        #     wandb.init(project='place-cell-tpc', config=self.options)
        for epoch_idx in range(self.n_epochs):
            epoch_loss = 0
            epoch_energy = 0
            epoch_acc = 0
            if self.options['evaluation']:
                epoch_acc_eval = 0
            iterable = dataloader if self.options['use_preloaded'] else range(self.n_steps)
            tbar = tqdm(iterable, leave=True)
            for item in tbar:
                if self.options['use_preloaded']:
                    inputs, pc_outputs, init_actv = item
                else:
                    inputs, pc_outputs, init_actv = self.traj_gen.generate_traj_data(ini_pos)
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
                ).mean(-1).mean()
                # acc = torch.min(
                #     pc_outputs.to(self.options['device']) == pred_pos, 
                #     dim=-1
                # ).values.mean(-1, dtype=torch.float32).mean().item()
                
                epoch_acc += acc
                epoch_loss += loss
                epoch_energy += energy
                is_eval = self.options['evaluation'] and epoch_idx % self.options['eval_every'] == 0
                
                if is_eval:
                    ini_pos_eval = next(self.options['validate_with_ini_pos']) if self.options['validate_with_ini_pos'] else ini_pos
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
                    ).mean(-1).mean()
                    epoch_acc_eval += acc_eval
                
                desc = f'Epoch: {epoch_idx+1}/{self.n_epochs}. Loss: {np.round(loss, 4)}. PC Energy: {np.round(energy, 4)}. Acc: {np.round(100 * acc, 2)}.'
                desc += f'Acc eval: {np.round(100 * acc_eval, 2)}' if is_eval else ''
                tbar.set_description(desc)

            # grad_norm = torch.norm(self.model.Wr.weight.grad, p='fro').item()

            # if self.options.is_wandb:
            #     wandb.log({
            #         'loss': epoch_loss / self.n_steps,
            #         'err': epoch_err / self.n_steps,
            #         'energy': epoch_energy / self.n_steps,
            #         'grad_norm': grad_norm,
            #     })
            self.loss.append(epoch_loss / self.n_steps)
            self.acc.append(epoch_acc / self.n_steps)
            self.energy.append(epoch_energy / self.n_steps)
            if is_eval:
                self.acc_eval.append(epoch_acc_eval / self.n_steps)

            # Update learning rate
            self.scheduler.step()

            # if save and (epoch_idx + 1) % self.options['save_every'] == 0:
            #     # Save checkpoint
            #     torch.save(
            #         {
            #             'init_model': self.init_model.state_dict(),
            #             'model': self.model.state_dict(),
            #         }, 
            #         os.path.join(
            #             self.ckpt_dir,
            #             f'epoch_{epoch_idx + 1}.pth'
            #         )
            #     )

        # torch.save(
        #     {
        #         'init_model': self.init_model.state_dict(),
        #         'model': self.model.state_dict(),
        #         'init_optimizer': self.init_optimizer.state_dict(),
        #         'optimizer': self.optimizer.state_dict(),
        #         'scheduler': self.scheduler.state_dict(),
        #     }, 
        #     os.path.join(
        #         self.ckpt_dir,
        #         'most_recent_model.pth'
        #     )
        # )

        tbar.close()
        # if self.options.is_wandb:
        #     wandb.finish()

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