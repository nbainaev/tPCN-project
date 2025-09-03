from agent import tPCNAgent
from base import BaseAgent
import numpy as np
from encoders import GridWorldEncoder, SimpleOneHotEncoder


class TPCNWrapper(BaseAgent):
    def __init__(self, conf):
        
        self.conf = conf
        self.encoder_type = conf['encoder_type']
        
        self.n_actions = conf['agent']['n_actions']
        self.n_obs_states = conf['agent']['n_obs_states']
        
        self.agent = tPCNAgent(**self.conf['agent'])

    def observe(self, obs, action, reward=0):
        return self.agent.observe(self.encoder.transform(obs), action, reward)
    
    def sample_action(self):
        return self.agent.sample_action()

    def reinforce(self, reward):
        return self.agent.reinforce(reward)
    
    def reset(self):
        return self.agent.reset()

    
    @property
    def state_value(self):
        action_values = self.agent.action_values
        if action_values is None:
            action_values = self.agent.evaluate_actions()
        state_value = np.sum(action_values)
        return state_value

    @property
    def action_values(self):
        action_values = self.agent.action_values
        if action_values is None:
            action_values = self.agent.evaluate_actions()
        return action_values
    
    @property
    def state_repr(self):
        return self.agent.prev_hidden.detach().numpy().squueze()

    def _make_encoder(self):
        
        if self.encoder_type == 'onehot':
            self.encoder = SimpleOneHotEncoder(max_categories=self.n_obs_states)
        else:
            raise ValueError(f'Encoder type {self.encoder_type} is not supported')
    
