import os
import numpy as np
import pickle as pkl
import torch
from gridworld.gridworld import GridWorld
from base import BaseRunner
from wrapper import TPCNWrapper
from base_models import HierarchicalPCN, TemporalPCN


class tPCNRunner(BaseRunner):
    
    @staticmethod
    def make_environment(env_type, conf, setup):
        if env_type == 'gridworld':
            from envs import GridWorldWrapper
            env = GridWorldWrapper(conf, setup)
        else:
            raise NotImplementedError
        
        return env
    
    def make_agent(agent_type, conf, unique_obs):
        if agent_type == 'tpc':
            from agent import tPCNAgent
            agent = TPCNWrapper(conf)

        return agent

    def switch_strategy(self, strategy):
        if strategy == 'random':
            self.reward_free = True
        elif strategy == 'non-random':
            self.reward_free = False
    
    def save_agent(self, dir_path):
        if self.logger is not None:
            name = self.logger.name
        else:
            from names_generator import generate_name
            name = generate_name()

        with open(os.path.join(dir_path, f'agent_{name}.pkl'), 'wb') as file:
            pkl.dump(self.agent.agent, file)
        
    def save_model(self, dir_path):
        init_model = self.agent.agent.init_model
        model = self.agent.agent.tpc_model
        if isinstance(model, TemporalPCN):
            torch.save(model, os.path.join(dir_path, f'{self.logger.name}_{self.episodes}.pt'))
        
        if isinstance(init_model, HierarchicalPCN):
            torch.save(init_model, os.path.join(dir_path, f'{self.logger.name}_{self.episodes}_init.pt'))
    
    @property
    def state(self):
        env = self.environment.environment
        assert isinstance(env, GridWorld)
        r, c = env.r, env.c
        return r * env.w + c
    
    @property
    def state_visited(self):
        env = self.environment.environment
        assert isinstance(env, GridWorld)
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        values[r, c] = 1

        return values, 1
    
    @property
    def state_value(self):
        env = self.environment.environment
        assert isinstance(env, GridWorld)
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        state_value = self.agent.state_value
        values[r, c] = state_value

        counts = np.zeros_like(values)
        counts[r, c] = 1
        return values, counts
    
    @property
    def state_size(self):
        env = self.environment.environment
        assert isinstance(env, GridWorld)
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        state_size = len(self.agent.agent.cluster)
        values[r, c] = state_size

        counts = np.zeros_like(values)
        counts[r, c] = 1
        return values, counts
    
    @property
    def state_representation(self):
        internal_messages = self.agent.agent.prev_hidden.detach().numpy().squueze()
        return internal_messages
    
    @property
    def q_value(self):
        env = self.environment.environment
        assert isinstance(env, GridWorld)
        # left, right, up, down
        actions = self.environment.actions
        shifts = np.array([[0, 0], [0, env.w], [env.h, 0], [env.h, env.w]])

        r, c = env.r, env.c
        values = np.zeros((env.h * 2, env.w * 2))
        action_values = self.agent.action_values
        counts = np.zeros_like(values)

        for value, shift in zip(action_values, shifts):
            x, y = r + shift[0], c + shift[1]
            values[x, y] = value
            counts[x, y] = 1

        return values, counts
    
    @property
    def rewards(self):
        agent = self.agent.agent
        return agent.rewards.reshape(1, -1)
    
    def save_encoder(self, path):
        with open(
            os.path.join(path,
             f'{self.logger.name}_{self.episodes}episodes_sp.pkl'),
            'wb'
        ) as file:
            pkl.dump(self.agent.encoder, file=file)