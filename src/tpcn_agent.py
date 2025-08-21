import torch
import numpy as np
from models.tpcn import TemporalPCN, HierarchicalPCN
from src.data_utils.data_utils import GridWorldEncoder

class tPCNAgent:
    def __init__(self, options, model=None, init_model=None):
        self.options = options
        self.init_model = HierarchicalPCN(options) if init_model is None else init_model
        self.tpc_model = TemporalPCN(options) if model is None else model
        self.lr = options['learning_rate']
        self.cum_reward = 0
        self.observations = []
        self.is_first = True
        self.dir_encoder = GridWorldEncoder(
            categories=[0, 1, 2, 3], 
            mode='directions', 
            encoder=options['encoder'])
        self.obs_encoder = GridWorldEncoder(
            categories= sorted(list(np.unique(options['room'][0]))  + [-1]) if options['mode'] == 'pomdp' else list(range(len(options['room'][0]))),
            mode=options['mode'],
            collision_hint=options['conf']['collision_hint'],
            encoder=options['encoder']
        )
        self.test_inf_iters = options['test_inf_iters']
        self.inf_lr = options['inf_lr']
        self.inf_iters = options['inf_iters']
        self.tpc_optimizer = torch.optim.Adam(
            self.tpc_model.parameters(), 
            lr=self.lr,
        )
        self.init_optimizer = torch.optim.Adam(
            self.init_model.parameters(),
            lr=self.lr,
        )
        self.prev_hidden = None
        self.prev_encoded_dir = None
        self.actions = []

        self._rng = np.random.default_rng(seed=42)
        
    def observe(self, obs, reward):
        self.observations.append(obs)
        self.cum_reward += reward
        encoded_obs = torch.tensor(self.obs_encoder.transform(obs), dtype=torch.float32).unsqueeze(0)

        if self.is_first:
            self.init_model.train()
            self.init_optimizer.zero_grad()
            self.init_model.inference(self.inf_iters, self.inf_lr, encoded_obs)
            energy, obs_loss = self.init_model.get_energy()
            energy.backward()
            self.init_optimizer.step()
            self.is_first = False
            self.prev_hidden = self.init_model.z.clone().detach()
        else:
            self.tpc_model.train()
            self.tpc_optimizer.zero_grad()
            self.tpc_model.inference(self.inf_iters, self.inf_lr, self.prev_encoded_dir, self.prev_hidden, encoded_obs)
            energy, obs_loss = self.tpc_model.get_energy()
            energy.backward()
            self.tpc_optimizer.step()

            # update the hidden state
            self.prev_hidden = self.tpc_model.z.clone().detach()
    
    def generate_sf(self, n_steps=5, gamma=0.95):
        _, pred_hidden = self.predict(self.prev_encoded_dir)
        return self.tpc_model.generate_successor_features(
            n_steps=n_steps, 
            prev_z=pred_hidden, 
            gamma=gamma).detach().numpy().squeeze()
    
    def predict(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action).unsqueeze(0)
        pred_hidden = self.tpc_model.g(action, self.prev_hidden)
        return self.tpc_model.decode(pred_hidden), pred_hidden
   
    def evaluate_actions(self):
        pass

    def act(self, avaiable_actions: int) -> int:
        action = self._rng.integers(avaiable_actions, size=1)
        self.actions.append(action)
        self.prev_encoded_dir = torch.tensor(self.dir_encoder.transform(action), dtype=torch.float32).unsqueeze(0)
        return action
    
    def reset(self):
        self.is_first = True
        self.cum_reward = 0
        self.observations = []
        self.actions = []
        self.prev_hidden = None
        self.prev_encoded_dir = None
