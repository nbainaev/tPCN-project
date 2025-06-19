# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import src.utils as utils
from src.constants import ACTIVATION_FUNCS


class HierarchicalPCN(nn.Module):
    def __init__(self, options):
        super().__init__()

        self.Wout = nn.Linear(options['latent_size'], options['obs_size'], bias=False)
        self.mu = nn.Parameter(torch.zeros((options['latent_size'])))

        # sparse penalty
        self.sparse_z = options['lambda_z_init']

        self.out_activation = ACTIVATION_FUNCS[options['out_activation']]
        self.loss = options['loss']

    def set_sparsity(self, sparsity):
        self.sparse_z = sparsity

    def set_nodes(self, inp):
        # intialize the value nodes
        self.z = self.mu.clone()
        self.x = inp.clone()

        # computing error nodes
        self.update_err_nodes()

    def decode(self, z):
        return self.out_activation(self.Wout(z))

    def update_err_nodes(self):
        self.err_z = self.z - self.mu
        pred_x = self.decode(self.z)
        if isinstance(self.out_activation, utils.Tanh):
            self.err_x = self.x - pred_x
        elif isinstance(self.out_activation, utils.Softmax):
            self.err_x = self.x / (pred_x + 1e-8)
        else:
            self.err_x = self.x / (pred_x + 1e-8) + (1 - self.x) / (1 - pred_x + 1e-8)

    def inference_step(self, inf_lr):
        Wout = self.Wout.weight.clone().detach()
        if isinstance(self.out_activation, utils.Softmax):
            delta = (
                self.err_z
                - (
                    self.out_activation.deriv(self.Wout(self.z))
                    @ self.err_x.unsqueeze(-1)
                ).squeeze(-1)
                @ Wout
            )
        else:
            delta = (
                self.err_z
                - (self.out_activation.deriv(self.Wout(self.z)) * self.err_x) @ Wout
            )
        delta += self.sparse_z * torch.sign(self.z)
        self.z = self.z - inf_lr * delta

    def inference(self, inf_iters, inf_lr, inp):
        self.set_nodes(inp)
        for itr in range(inf_iters):
            with torch.no_grad():
                self.inference_step(inf_lr)
            self.update_err_nodes()

    def get_energy(self):
        """Function to obtain the sum of all layers' squared MSE"""
        if self.loss == "CE":
            obs_loss = F.cross_entropy(self.Wout(self.z), self.x)
        elif self.loss == "BCE":
            obs_loss = F.binary_cross_entropy_with_logits(self.Wout(self.z), self.x)
        else:
            obs_loss = torch.sum(self.err_x**2, -1).mean()
        latent_loss = torch.sum(self.err_z**2, -1).mean()
        energy = obs_loss + latent_loss
        return energy, obs_loss


class TemporalPCN(nn.Module):
    """Multi-layer tPC class, using autograd"""

    def __init__(self, options):
        super(TemporalPCN, self).__init__()
        self.Wr = nn.Linear(options['latent_size'], options['latent_size'], bias=False)
        self.Win = nn.Linear(options['dir_size'], options['latent_size'], bias=False)
        self.Wout = nn.Linear(options['latent_size'], options['obs_size'], bias=False)

        self.sparse_z = options['lambda_z']
        self.weight_decay = options['weight_decay']

        self.out_activation = ACTIVATION_FUNCS[options['out_activation']]
        self.rec_activation = ACTIVATION_FUNCS[options['rec_activation']]
        self.loss = options['loss']

    def set_nodes(self, v, prev_z, p):
        """Set the initial value of the nodes;

        In particular, we initialize the hiddden state with a forward pass.

        Args:
            v: velocity input at a particular timestep in stimulus
            prev_z: previous hidden state
            p: place cell activity at a particular timestep in stimulus
        """
        self.z = self.g(v, prev_z)
        self.x = p.clone()
        self.update_err_nodes(v, prev_z)

    def update_err_nodes(self, v, prev_z):
        self.err_z = self.z - self.g(v, prev_z)
        pred_x = self.decode(self.z)
        if isinstance(self.out_activation, utils.Tanh):
            self.err_x = self.x - pred_x
        elif isinstance(self.out_activation, utils.Softmax):
            self.err_x = self.x / (pred_x + 1e-9)
        else:
            self.err_x = self.x / (pred_x + 1e-9) - (1 - self.x) / (1 - pred_x + 1e-9)

    def g(self, v, prev_z):
        return self.rec_activation(self.Wr(prev_z) + self.Win(v))

    def decode(self, z):
        return self.out_activation(self.Wout(z))

    def inference_step(self, inf_lr, v, prev_z):
        """Take a single inference step"""
        Wout = self.Wout.weight.detach().clone()  # shape [Np, Ng]
        if isinstance(self.out_activation, utils.Softmax):
            delta = (
                self.err_z
                - (
                    self.out_activation.deriv(self.Wout(self.z))
                    @ self.err_x.unsqueeze(-1)
                ).squeeze(-1)
                @ Wout
            )
        else:
            delta = (
                self.err_z
                - (self.out_activation.deriv(self.Wout(self.z)) * self.err_x) @ Wout
            )
        delta += self.sparse_z * torch.sign(self.z)
        self.z = self.z - inf_lr * delta

    def inference(self, inf_iters, inf_lr, v, prev_z, p):
        """Run inference on the hidden state"""
        self.set_nodes(v, prev_z, p)
        for i in range(inf_iters):
            with torch.no_grad():  # ensures self.z won't have grad when we call backward
                self.inference_step(inf_lr, v, prev_z)
            self.update_err_nodes(v, prev_z)

    def get_energy(self):
        """Returns the average (across batches) energy of the model"""
        if self.loss == "CE":
            obs_loss = F.cross_entropy(self.Wout(self.z), self.x)
        elif self.loss == "BCE":
            obs_loss = F.binary_cross_entropy_with_logits(self.Wout(self.z), self.x)
        else:
            obs_loss = torch.sum(self.err_x**2, -1).mean()
        latent_loss = torch.sum(self.err_z**2, -1).mean()
        energy = obs_loss + latent_loss
        energy += self.weight_decay * (torch.mean(self.Wr.weight**2))

        return energy, obs_loss

# TODO: replace multilayerPCN with HierarchicalPCN in pc_pcn.py
class MultilayerPCN(nn.Module):
    def __init__(self, nodes, nonlin, lamb=0.0, use_bias=False, relu_inf=True):
        super().__init__()
        self.n_layers = len(nodes)
        self.layers = nn.Sequential()
        for l in range(self.n_layers - 1):
            self.layers.add_module(
                f"layer_{l}",
                nn.Linear(
                    in_features=nodes[l],
                    out_features=nodes[l + 1],
                    bias=use_bias,
                ),
            )

        self.mem_dim = nodes[0]
        self.memory = nn.Parameter(torch.zeros((nodes[0],)))
        self.relu_inf = relu_inf

        if nonlin == "tanh":
            nonlin = utils.Tanh()
        elif nonlin == "ReLU":
            nonlin = utils.ReLU()
        elif nonlin == "linear":
            nonlin = utils.Linear()
        self.nonlins = [nonlin] * (self.n_layers - 1)
        self.use_bias = use_bias

        # initialize nodes
        self.val_nodes = [[] for _ in range(self.n_layers)]
        self.errs = [[] for _ in range(self.n_layers)]

        # sparse penalty
        self.lamb = lamb

    def set_sparsity(self, sparsity):
        self.lamb = sparsity

    def get_inf_losses(self):
        return self.inf_losses  # inf_iters,

    def update_err_nodes(self):
        for l in range(0, self.n_layers):
            if l == 0:
                self.errs[l] = self.val_nodes[l] - self.memory
            else:
                preds = self.layers[l - 1](self.nonlins[l - 1](self.val_nodes[l - 1]))
                self.errs[l] = self.val_nodes[l] - preds

    def update_val_nodes(self, inf_lr):
        for l in range(0, self.n_layers - 1):
            derivative = self.nonlins[l].deriv(self.val_nodes[l])
            # sparse penalty
            penalty = self.lamb if l == 0 else 0.0
            delta = (
                -self.errs[l]
                - penalty * torch.sign(self.val_nodes[l])
                + derivative * torch.matmul(self.errs[l + 1], self.layers[l].weight)
            )
            self.val_nodes[l] = (
                F.relu(self.val_nodes[l] + inf_lr * delta)
                if self.relu_inf
                else self.val_nodes[l] + inf_lr * delta
            )

    def set_nodes(self, batch_inp):
        # computing val nodes
        self.val_nodes[0] = self.memory.clone()
        for l in range(1, self.n_layers - 1):
            self.val_nodes[l] = self.layers[l - 1](
                self.nonlins[l - 1](self.val_nodes[l - 1])
            )
        self.val_nodes[-1] = batch_inp.clone()

        # computing error nodes
        self.update_err_nodes()

    def inference(self, batch_inp, n_iters, inf_lr):
        self.set_nodes(batch_inp)
        self.batch_size = batch_inp.shape[0]
        self.inf_losses = []

        for itr in range(n_iters):
            with torch.no_grad():
                self.update_val_nodes(inf_lr)
            self.update_err_nodes()
            self.inf_losses.append(self.get_energy().item())

    def get_energy(self):
        """Function to obtain the sum of all layers' squared MSE"""
        total_energy = 0
        for l in range(self.n_layers):
            total_energy += torch.sum(
                self.errs[l] ** 2
            )  # average over batch and feature dimensions
        return total_energy * (100 / self.batch_size)
