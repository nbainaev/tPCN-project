import numpy as np
import random as rd

# class tPCAgent():
#     def __init__(self):
#         self._rng = np.random.default_rng(seed=42)
    
#     def predict_obs(self, obss: list, actions: list) -> int:
#         pred_obs = rd.choice(obss)
#         return pred_obs

#     def act(self, avaiable_actions: int) -> int:
#         action = self._rng.integers(avaiable_actions)
#         return action

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, Tuple


import torch
import torch.nn as nn
from torch import Tensor

class CombinedLoss(nn.Module):
    def __init__(self):
        """
        Combined MSE loss that returns per-batch-item losses.
        Always outputs tensor of shape (n_batch,).
        """
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, 
                input_p: Tensor, 
                target_p: Tensor,
                input_g: Tensor, 
                target_g: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute per-item combined loss.
        
        Args:
            input_p: Prediction states (n_batch, ...)
            target_p: Target states (n_batch, ...)
            input_g: Prediction observations (n_batch, ...)
            target_g: Target observations (n_batch, ...)
            
        Returns:
            Tuple of tensors with shape (n_batch,):
            - total_loss (per-item)
            - prediction_loss (per-item)
            - state_loss (per-item)
        """
        # Calculate MSE losses (per-element)
        loss_p = self.mse_loss(input_p, target_p)
        loss_g = self.mse_loss(input_g, target_g)
        
        # Reduce to per-batch-item (n_batch,)
        loss_p = loss_p.view(loss_p.size(0), -1).mean(dim=1)
        loss_g = loss_g.view(loss_g.size(0), -1).mean(dim=1)
        
        # Combined loss
        total_loss = loss_p + loss_g
        
        return total_loss, loss_p, loss_g

class tPCN(nn.Module):
    def __init__(self, 
                 hidden_size=64, 
                 max_iter=10,
                 lr_iter=0.001,
                 lr_Win=0.001,
                 lr_Wr=0.001,
                 lr_Wout=0.001,
                 tol=0.001,
                 n_obs=5, 
                 n_dir=4,
                 batch_size=None,
                 loss="mse"):
        super(tPCN, self).__init__()

        self.n_obs = n_obs
        self.max_iter = max_iter
        self.n_dir = n_dir
        self.hidden_size = hidden_size
        self.tol = tol
        self.lr_iter = lr_iter
        self.loss_fn = loss
        self.batch_size = batch_size

        self.lr_Win = lr_Win
        self.lr_Wr = lr_Wr
        self.lr_Wout = lr_Wout
        self.Wr = torch.randn(hidden_size, hidden_size)
        self.Win = torch.randn(hidden_size, n_dir)
        self.Wout = torch.randn(n_obs, hidden_size) * 2.0
        
        self.obs_encode_dict = {}
        self.obs_decode_dict = {}
        self.dir_encode_dict = {}
        self.dir_decode_dict = {}
        
        self.h = torch.nn.ReLU()
        self.f = torch.nn.Softmax(dim=1)
        self.norm = lambda x: x / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
        
        self.obs_memory = np.array([])
        self.directions_memory = np.array([])
        self.prev_states = torch.randn([batch_size, hidden_size, 1])
        self.prev_states = self.norm(self.prev_states)

    def softmax_jacobian(self, z):
        # z shape: (batch_size, output_dim)
        z = z.squeeze(2)
        f = torch.softmax(z, dim=1)  # shape: (batch_size, output_dim)
        
        # Create batch of identity matrices
        eye = torch.eye(f.size(1), device=z.device).unsqueeze(0).expand(f.size(0), -1, -1)  # shape: (batch_size, output_dim, output_dim)
        
        # Compute J = diag(f) - f.T @ f for each batch element
        J = eye * f.unsqueeze(2) - torch.bmm(f.unsqueeze(2), f.unsqueeze(1))  # shape: (batch_size, output_dim, output_dim)
        
        return J
    
    def relu_jacobian(self, x):
        x = x.squeeze(2)
        # x shape: (batch_size, state_dim)
        deriv_mask = (x > 0).float()  # shape: (batch_size, state_dim)
        
        # Create batch of diagonal matrices
        J = torch.diag_embed(deriv_mask)  # shape: (batch_size, state_dim, state_dim)
        
        return J
    
    def reset(self):
        self.prev_states = torch.randn([self.batch_size, self.hidden_size, 1])
        self.prev_states = self.norm(self.prev_states)
        self.obs_memory = np.array([])
        self.directions_memory = np.array([])
        return self
    
    def forward(self, directions, obs):
        # directions shape: (batch_size,)
        # obs shape: (batch_size,)
        
        # One-hot encoding for batch
        directions_embed = self.encode(directions, mode="dir")  # shape: (batch_size, dir_vocab_size)
        obs_embed = self.encode(obs, mode="obs")  # shape: (batch_size, obs_vocab_size)

        Win_batch = self.Win.unsqueeze(0).expand(self.batch_size, -1, -1)
        Wr_batch = self.Wr.unsqueeze(0).expand(self.batch_size, -1, -1)
        Wout_batch = self.Wout.unsqueeze(0).expand(self.batch_size, -1, -1)
        
        current_states = self.h(torch.bmm(Wr_batch, self.prev_states) + torch.bmm(Win_batch, directions_embed))
        current_states = self.norm(current_states)
        prev_states = self.prev_states.detach().clone()
        i = 0
        while i < self.max_iter and abs(float(torch.mean(prev_states - current_states))) > self.tol:
            eps_p = obs_embed - self.f(torch.bmm(Wout_batch, current_states))
            
            eps_g = current_states - self.h(torch.bmm(Wr_batch, prev_states) + torch.bmm(Win_batch, directions_embed))
            prev_states = current_states.detach().clone()
            if self.loss_fn == "mse":
                current_states = current_states + self.lr_iter * (-eps_g +\
                                        torch.bmm(self.Wout.t().unsqueeze(0).expand(self.batch_size, -1, -1),
                                                torch.bmm(self.softmax_jacobian(torch.bmm(Wout_batch, current_states)), eps_p)))
            elif self.loss_fn == "cross-entropy":
                current_states = current_states + self.lr_iter * (-eps_g + torch.bmm(
                    self.Wout.T.unsqueeze(0).expand(self.batch_size, -1, -1), eps_p)
                )
            current_states = self.norm(current_states)
            i += 1
        next_obs_preds = self.f(torch.bmm(Wout_batch, current_states))
        return next_obs_preds, current_states

    def optim_step(self, directions, obs):
        directions_embed = self.encode(directions, mode="dir")
        obs_embed = self.encode(obs, mode="obs")
        next_pred_states, current_states = self.forward(directions, obs)

        Win_batch = self.Win.unsqueeze(0).expand(self.batch_size, -1, -1)
        Wr_batch = self.Wr.unsqueeze(0).expand(self.batch_size, -1, -1)
        Wout_batch = self.Wout.unsqueeze(0).expand(self.batch_size, -1, -1)

        eps_p = obs_embed - self.f(torch.bmm(Wout_batch, current_states))
        eps_g = current_states - self.h(torch.bmm(Wr_batch, self.prev_states) + torch.bmm(Win_batch, directions_embed))
        predicted_states = torch.bmm(Wr_batch, self.prev_states) + torch.bmm(Win_batch, directions_embed)


        if self.loss_fn == "mse":
            delta_Wout = torch.bmm(
                                torch.bmm(
                                    self.softmax_jacobian(
                                        torch.bmm(Wout_batch, current_states)
                                    ), 
                                    eps_p
                                ), 
                                torch.transpose(current_states, 1, 2)
                            )
        elif self.loss_fn == "cross-entropy":
            delta_Wout = torch.bmm(eps_p, torch.transpose(current_states, 1, 2))

        delta_Wr = torch.bmm(
                            torch.bmm(
                                self.relu_jacobian(
                                    predicted_states
                                ),
                                eps_g
                            ),
                            torch.transpose(self.prev_states, 1, 2)
                        )
        delta_Win = torch.bmm(
                    torch.bmm(
                        self.relu_jacobian(
                            predicted_states
                        ),
                        eps_g
                    ),
                    torch.transpose(directions_embed, 1, 2)
                )
        
        with torch.no_grad():
            self.Win = self.Win + self.lr_Win * delta_Win.mean(dim=0)
            self.Wr = self.Wr + self.lr_Wr * delta_Wr.mean(dim=0)
            self.Wout = self.Wout + self.lr_Wout * delta_Wout.mean(dim=0)
        # print(
        #     f"Gradients: "
        #     f"Wout={torch.norm(delta_Wout):.3f}, "
        #     f"Wr={torch.norm(delta_Wr):.3f}, "
        #     f"Win={torch.norm(delta_Win):.3f}"
        # )
        return next_pred_states, current_states
    
    def decode(self, x, mode="obs"):
        # Проверяем, что вход - это батч (тензор с размерностью >= 2)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Если на входе вектор, превращаем его в батч из 1 элемента
        
        N = x.shape[0]  # Размер батча
        decoded_values = []
        
        if mode == "obs":
            decode_dict = self.obs_decode_dict
        else:
            decode_dict = self.dir_decode_dict
        
        # Для каждого элемента в батче находим argmax и декодируем
        for i in range(N):
            n = int(torch.argmax(x[i]))
            if n not in decode_dict:
                n = 0  # Значение по умолчанию, если индекса нет в словаре
            decoded_values.append(decode_dict[n])
        
        return decoded_values if N > 1 else decoded_values[0]  # Возвращаем список или одно значение
    
    def encode(self, x, mode="obs"):
        # Преобразуем вход в список, если это не батч
        if not isinstance(x, (list, tuple, torch.Tensor, np.ndarray)):
            x = [x]
        
        # Преобразуем все элементы в int
        x = [int(val) for val in x]
        N = len(x)  # Размер батча
        
        if mode == "obs":
            encode_dict = self.obs_encode_dict
            decode_dict = self.obs_decode_dict
            n_total = self.n_obs  # Общее количество возможных значений
        else:
            encode_dict = self.dir_encode_dict
            decode_dict = self.dir_decode_dict
            n_total = self.n_dir

        # Добавляем новые значения в словарь, если их там нет
        for val in x:
            if val not in encode_dict:
                new_idx = len(encode_dict)
                encode_dict[val] = new_idx
                decode_dict[new_idx] = val
        
        # Создаем тензор нулей размера N x C
        C = n_total
        vec = torch.zeros((N, C), dtype=torch.float32)
        
        # Заполняем one-hot кодировку
        for i, val in enumerate(x):
            idx = encode_dict[val]
            vec[i, idx] = 1.0
        
        return vec.unsqueeze(2)

    def update_memory(self, obs, directions=None):
        self.obs_memory = obs
        if directions is not None:
            self.directions_memory = directions


class tPCAgent:
    def __init__(self, 
                 hidden_size=64, 
                 max_iter=10,
                 lr_iter=0.001,
                 lr_Win=0.001,
                 lr_Wr=0.001,
                 lr_Wout=0.001,
                 tol=0.001,
                 n_obs=5, 
                 n_dir=4,
                 batch_size=32,
                 loss="mse"):
        self.tpcn = tPCN(hidden_size=hidden_size, 
                         max_iter=max_iter, 
                         lr_iter=lr_iter,
                         lr_Win=lr_Win,
                         lr_Wr=lr_Wr,
                         lr_Wout=lr_Wout,
                         tol=tol,
                         n_obs=n_obs, 
                         n_dir=n_dir,
                         batch_size=batch_size,
                         loss=loss)
        self.loss_fn = CombinedLoss()
        self._rng = np.random.default_rng(seed=42)
        self.n_obs = n_obs
        self.batch_size = batch_size
        self.losses = []
        self.losses_p = []
        self.losses_g = []

    def reset(self):
        self.tpcn.reset()
        self.losses = []
        self.losses_p = []
        self.losses_g = []
        return self   
    
    def process_step(self, next_obs_batch, directions_batch):
        """
        Process a batch of observations and directions
        
        Args:
            next_obs_batch: Tensor of shape (batch_size,) - batch of next observations
            directions_batch: Tensor of shape (batch_size,) - batch of directions
        """
        with torch.autograd.set_detect_anomaly(True):
            # Filter out negative observations (if needed)
            # valid_mask = next_obs_batch > 0
            # if valid_mask.any():
            #     valid_obs = next_obs_batch[valid_mask]
            #     valid_directions = directions_batch[valid_mask]
                
            #     # Update memory only for valid observations
            #     for obs, dir in zip(valid_obs, valid_directions):
            #         self.tpcn.update_memory(obs, dir)
            self.tpcn.update_memory(next_obs_batch, dir)
            # Encode the entire batch
            directions_embed = self.tpcn.encode(directions_batch, mode="dir")
            
            # Process the entire batch through TPCN
            next_pred_states, current_states = self.tpcn.optim_step(directions_batch, next_obs_batch)

            # Compute previous states for the batch
            prev_states = self.tpcn.h(
                torch.bmm(self.tpcn.Wr.unsqueeze(0).expand(self.batch_size, -1, -1), self.tpcn.prev_states) + 
                torch.bmm(self.tpcn.Win.unsqueeze(0).expand(self.batch_size, -1, -1), directions_embed)
            )
            
            # Compute loss for the batch
            target_obs_embed = self.tpcn.encode(next_obs_batch, mode="obs").detach()
            target_states = current_states.detach()
            
            loss, loss_p, loss_g = self.loss_fn(
                next_pred_states,  # predictions
                target_obs_embed,  # target observations
                prev_states,       # previous states
                target_states      # target states
            )
            # Store mean losses for the batch
            self.losses.append(loss)
            self.losses_p.append(loss_p)
            self.losses_g.append(loss_g)
            return loss, loss_p, loss_g
        
    def predict_observation(self, directions):
        with torch.no_grad():
            pred_proba, _ = self.tpcn(directions, self.tpcn.obs_memory)
            preds = self.tpcn.decode(pred_proba, mode="obs")
            return preds, pred_proba
    
    def act(self, avaiable_actions: int) -> int:
        actions = self._rng.integers(avaiable_actions, size=self.batch_size)
        return actions

