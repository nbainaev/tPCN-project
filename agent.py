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


class CombinedLoss(nn.MSELoss):
    def __init__(self):
        super(CombinedLoss, self).__init__()
    
    def forward(self, input_p: Tensor, target_p: Tensor, input_g: Tensor, target_g: Tensor) -> Tensor:
        # assert not torch.isnan(input_p).any(), "input_p contains NaN!"
        # assert not torch.isinf(input_p).any(), "input_p contains inf!"
        # assert not torch.isnan(target_p).any(), "target_p contains NaN!"
        # assert not torch.isinf(target_p).any(), "target_p contains inf!"
        eps = 1e-8
        loss_p = F.mse_loss(input_p, target_p, reduction=self.reduction)
        loss_g = F.mse_loss(input_g, target_g, reduction=self.reduction)
        return loss_p + loss_g, loss_p, loss_g

class tPCN(nn.Module):
    def __init__(self, 
                 hidden_size=64, 
                 memory_size=10, 
                 max_iter=10,
                 lr_iter=0.001,
                 lr_Win=0.001,
                 lr_Wr=0.001,
                 lr_Wout=0.001,
                 tol=0.001,
                 n_obs=5, 
                 n_dir=4):
        super(tPCN, self).__init__()

        self.n_obs = n_obs
        self.max_iter = max_iter
        self.n_dir = n_dir
        self.hidden_size = hidden_size
        self.tol = tol
        self.lr_iter = lr_iter
        self.lr_Win = lr_Win
        self.lr_Wr = lr_Wr
        self.lr_Wout = lr_Wout
        # self.direction_encoder = nn.Linear(1, n_dir, bias=False)
        # self.obs_encoder = nn.Linear(1, n_obs, bias=False)
        # self.Wr = nn.Parameter(torch.randn(hidden_size, hidden_size))
        # self.Win = nn.Parameter(torch.randn(hidden_size, n_dir))

        # self.Wout = nn.Parameter(torch.randn(n_obs, hidden_size))
        self.Wr = torch.randn(hidden_size, hidden_size)
        self.Win = torch.randn(hidden_size, n_dir)

        self.Wout = torch.randn(n_obs, hidden_size) * 2.0

        self.obs_encode_dict = {}
        self.obs_decode_dict = {}
        self.dir_encode_dict = {}
        self.dir_decode_dict = {}
        
        self.h = torch.relu
        self.f = torch.softmax
        
        self.memory_size = memory_size
        self.obs_memory = deque(maxlen=memory_size)
        self.direction_memory = deque(maxlen=memory_size)
        self.prev_state = torch.randn([hidden_size, 1])
        self.prev_state = self.prev_state / (torch.norm(self.prev_state) + 1e-6)
    

    def softmax_jacobian(self, z):
        f = torch.softmax(z, dim=0).squeeze()
        J = torch.diag(f) - torch.outer(f, f)
        return J
    
    def relu_jacobian(self, x):

        deriv_mask = (x.squeeze() > 0).float()
        J = torch.diag(deriv_mask)
        return J
    
    def reset(self):
        self.prev_state = torch.randn([self.hidden_size, 1])
        self.prev_state = self.prev_state / (torch.norm(self.prev_state) + 1e-6)
        self.obs_memory = deque(maxlen=self.memory_size)
        self.direction_memory = deque(maxlen=self.memory_size)
        return self
    
    def forward(self, direction):
        # current_obs = torch.FloatTensor(self.obs_memory[-1])
        # direction = torch.FloatTensor([direction])
        # obs_embed = self.obs_encoder(current_obs).T
        # direction_embed = self.direction_encoder(direction).unsqueeze(1)
        # print(obs_embed.shape)
        # print(direction_embed.shape)

        direction_embed = self.encode(direction, mode="dir")
        obs_embed = self.encode(self.obs_memory[-1], mode="obs")
        current_state = self.h(torch.mm(self.Wr, self.prev_state) + torch.mm(self.Win, direction_embed))
        current_state = current_state / (torch.norm(current_state) + 1e-6)
        prev_state = self.prev_state.detach().clone()
        i = 0
        while i < self.max_iter and abs(float(torch.mean(prev_state - current_state))) > self.tol:
            eps_p = obs_embed - self.f(torch.mm(self.Wout, current_state), dim=0)
            eps_g = current_state - self.h(torch.mm(self.Wr, prev_state) + torch.mm(self.Win, direction_embed))
            prev_state = current_state.detach().clone()
            current_state = current_state + self.lr_iter * (-0.1*eps_g +\
                                    torch.mm(self.Wout.t(),
                                             torch.mm(self.softmax_jacobian(torch.mm(self.Wout, current_state)), eps_p)))
            current_state = current_state / (torch.norm(current_state) + 1e-6)
            i += 1
        next_obs_pred = self.f(torch.mm(self.Wout, current_state), dim=0)
        return next_obs_pred, current_state

    def optim_step(self, direction):
        direction_embed = self.encode(direction, mode="dir")
        obs_embed = self.encode(self.obs_memory[-1], mode="obs")
        next_pred_state, current_state = self.forward(direction)
        eps_p = obs_embed - self.f(torch.mm(self.Wout, current_state), dim=0)
        eps_g = current_state - self.h(torch.mm(self.Wr, self.prev_state) + torch.mm(self.Win, direction_embed))
        delta_Wout = torch.mm(torch.mm(self.softmax_jacobian(torch.mm(self.Wout, current_state)), eps_p), current_state.T)
        predicted_state = torch.mm(self.Wr, self.prev_state) + torch.mm(self.Win, direction_embed)
        delta_Wr = torch.mm(
                            torch.mm(
                                self.relu_jacobian(
                                    predicted_state
                                ),
                                eps_g
                            ),
                            self.prev_state.T
                        )
        delta_Win = torch.mm(
                    torch.mm(
                        self.relu_jacobian(
                            predicted_state
                        ),
                        eps_g
                    ),
                    direction_embed.T
                )
        # delta_Wout = delta_Wout * (0.1 / (torch.norm(delta_Wout) + 1e-6))
        self.Win = self.Win + self.lr_Win * delta_Win
        self.Wr = self.Wr + self.lr_Wr * delta_Wr
        self.Wout = self.Wout + self.lr_Wout * delta_Wout
        # print(
        #     f"Gradients: "
        #     f"Wout={torch.norm(delta_Wout):.3f}, "
        #     f"Wr={torch.norm(delta_Wr):.3f}, "
        #     f"Win={torch.norm(delta_Win):.3f}"
        # )
        return next_pred_state, current_state
    def encode(self, x, mode="obs"):
        x = int(x)
        if mode == "obs":
            n = len(self.obs_encode_dict.values())
            if x not in self.obs_encode_dict.keys():
                self.obs_encode_dict[x] = n 
                self.obs_decode_dict[n] = x
            vec = [0] * self.n_obs
            vec[self.obs_encode_dict[x]] = 1
            return torch.FloatTensor(vec).unsqueeze(1)
        else:
            n = len(self.dir_encode_dict.values())
            if x not in self.dir_encode_dict.keys():
                self.dir_encode_dict[x] = n 
                self.dir_decode_dict[n] = x
            vec = [0] * self.n_dir
            vec[self.dir_encode_dict[x]] = 1
            return torch.FloatTensor(vec).unsqueeze(1)
    
    def decode(self, x, mode="obs"):
        if mode == "obs":
            n = int(torch.argmax(x))
            if n not in self.obs_decode_dict.keys():
                n = 0
            return self.obs_decode_dict[n]
        else:
            n = int(torch.argmax(x))
            if n not in self.dir_decode_dict.keys():
                n = 0
            return self.dir_decode_dict[n]

    def update_memory(self, obs, direction=None):
        self.obs_memory.append(obs)
        if direction is not None:
            self.direction_memory.append(direction)

class tPCAgent:
    def __init__(self, 
                 hidden_size=64, 
                 memory_size=10, 
                 max_iter=10,
                 lr_iter=0.001,
                 lr_Win=0.001,
                 lr_Wr=0.001,
                 lr_Wout=0.001,
                 tol=0.001,
                 n_obs=5, 
                 n_dir=4):
        self.tpcn = tPCN(hidden_size=hidden_size, 
                         memory_size=memory_size, 
                         max_iter=max_iter, 
                         lr_iter=lr_iter,
                         lr_Win=lr_Win,
                         lr_Wr=lr_Wr,
                         lr_Wout=lr_Wout,
                         tol=tol,
                         n_obs=n_obs, 
                         n_dir=n_dir)
        # self.optimizer = optim.Adam(self.tpcn.parameters(), lr=1e-3)
        self.loss_fn = CombinedLoss()
        self._rng = np.random.default_rng(seed=42)
        self.n_obs = n_obs
        self.losses = []
        self.losses_p = []
        self.losses_g = []

    def reset(self):
        self.tpcn.reset()
        self.losses = []
        return self   
    
    def process_step(self, next_obs, direction):
        
        with torch.autograd.set_detect_anomaly(True):
            self.tpcn.update_memory(next_obs, direction)
            next_obs = torch.FloatTensor([next_obs])
            direction_embed = self.tpcn.encode(direction, mode="dir")
            # pred, current_state = self.tpcn(direction)
            # prev_state = self.tpcn.h(torch.mm(self.tpcn.Wr, self.tpcn.prev_state) + torch.mm(self.tpcn.Win, direction_embed))
            # loss = self.loss_fn(pred, self.tpcn.encode(next_obs, mode="obs").detach(), prev_state, current_state.detach())
            # print(loss)

            next_pred_state, current_state = self.tpcn.optim_step(direction)
            prev_state = self.tpcn.h(torch.mm(self.tpcn.Wr, self.tpcn.prev_state) + torch.mm(self.tpcn.Win, direction_embed))
            loss, loss_p, loss_g = self.loss_fn(next_pred_state, self.tpcn.encode(next_obs, mode="obs").detach(), prev_state, current_state.detach())
            
            self.losses.append(loss)
            self.losses_p.append(loss_p)
            self.losses_g.append(loss_g)
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
        
    def predict_observation(self, direction):
        with torch.no_grad():
            pred, _ = self.tpcn(direction)
            pred = self.tpcn.decode(pred, mode="obs")
            return pred
    
    def act(self, avaiable_actions: int) -> int:
        action = self._rng.integers(avaiable_actions)
        return action

