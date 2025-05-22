#  Copyright (c) 2025 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from gridworld import GridWorld
from ruamel.yaml import YAML
from pathlib import Path
import numpy as np
import wandb
import os
from agent import tPCAgent
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from typing import Callable, Optional, List



os.environ['WANDB_ENTITY'] = 'nik-baynaev-national-research-nuclear-university-mephi'

def make_sync_envs(
    env_creator: Callable[[], gym.Env],
    num_envs: int = 4,
    render_mode: Optional[str] = None
) -> gym.vector.VectorEnv:
    """
    Создает синхронизированные копии кастомной среды
    
    Args:
        env_creator: Функция, создающая экземпляр среды
        num_envs: Количество сред
        render_mode: Режим рендеринга
        
    Returns:
        Векторизированная среда с поддержкой кастомных методов
    """
    class WrappedSyncVectorEnv(SyncVectorEnv):
        """Обертка для поддержки кастомных методов"""
        def __getattr__(self, name):
            if name.startswith('_'):
                raise AttributeError(name)
            
            # Проксируем вызовы кастомных методов
            def handler(*args, **kwargs):
                results = []
                for env in self.envs:
                    if hasattr(env, name):
                        results.append(getattr(env, name)(*args, **kwargs))
                return results
            return handler

    # Создаем список фабрик сред
    env_fns = [lambda: env_creator() for _ in range(num_envs)]
    
    # Инициализируем векторную среду
    envs = WrappedSyncVectorEnv(env_fns)
    
    # Устанавливаем режим рендеринга для всех сред
    if render_mode:
        for env in envs.envs:
            env.render_mode = render_mode
            
    return envs

def read_config(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with filepath.open('r') as config_io:
        yaml = YAML()
        return yaml.load(config_io)


if __name__ == '__main__':
    conf_path = 'config/pomdp.yaml'
    setup_path = 'config/free.yaml'
    run_path = 'config/run.yaml'
    agent_conf_path = 'config/agent_conf.yaml'
    seed = 10
    
    conf = read_config(conf_path)
    setup = read_config(setup_path)
    run = read_config(run_path)
    agent_conf = read_config(agent_conf_path)
    log_update_rate = run["log_update_rate"]
    if "start_position" in setup:
        # check that config has start position parameter
        ini_pos = setup.pop("start_position")
    else:
        # if None, initial agent's position is random
        ini_pos = (None, None)

    # env = GridWorld(setup['room'], **conf)
    envs = make_sync_envs(
        env_creator=lambda: GridWorld(setup['room'], **conf),
        num_envs=run['batch_size'],
    )
    if run['log']:
        # start wandb logger
        logger = wandb.init(
            project=run['project_name'], entity=os.environ['WANDB_ENTITY'],
            config=conf
        )
    else:
        logger = None
    episodes = run['episodes']
    steps = run['steps']
    n_obs = len(np.unique(setup['room'])) + 1
    agent = tPCAgent(n_obs=n_obs, batch_size=run['batch_size'], **agent_conf)
    legend = []
    losses = np.array([])
    losses_p = np.array([])
    losses_g = np.array([])
    accuracy = np.array([])
    # run several episodes
    for episode in tqdm(range(episodes), desc="Processing"):
        # Reset environment with optional starting position
        if ini_pos is not None:
            observations, infos = envs.reset(options={'start_r': ini_pos[0], 'start_c': ini_pos[1]})
        else:
            observations, infos = envs.reset()
        
        truncated = False
        terminated = False
        agent.reset()
        actions = []
        pred_observations = []
        true_observations = []

        
        # do steps in one episode
        for step in range(steps):
            if step > 0:
                # Get action from agent (assuming agent.act returns action index)
                # print(envs.action_space.shape)
                # action_shape = envs.action_space.shape[0] - 1
                action_idx = agent.act(4)  # Use action_space.n for number of actions
                action = action_idx  # Assuming discrete actions
                actions.append(action)
                
                # Get predicted observation from agent
                observation, observation_proba = agent.predict_observation(action)
                pred_observations.append(observation)
                
                # Execute action in environment
                observations, rewards, terminated, truncated, infos = envs.step(action)
            
            # Process observation
            obs = observations[0] if isinstance(observations, list) else observations.squeeze()
            
            
            if step > 0:
                # Process step with agent
                true_observations.append(obs)
                agent.process_step(obs, action)
            elif step == 0:
                # Initialize agent's memory with first observation
                prev_obs = obs
                agent.tpcn.update_memory(obs=obs)
            
            # finish episode early if terminal state is entered
            # if terminated or truncated:
            #     break
        losses = np.append(losses, agent.losses[-1].mean())
        losses_p = np.append(losses_p, agent.losses_p[-1].mean())
        losses_g = np.append(losses_g, agent.losses_g[-1].mean())
        true_observations = np.array(true_observations)
        pred_observations = np.array(pred_observations)
        accuracy = np.append(accuracy, ((true_observations == pred_observations).sum(axis=0) / pred_observations.shape[0]).mean())
        #print(accuracy.shape)
        #print(f"Accuracy on episode {episode}: {accuracy[-1]}")
        if logger is not None:
            logger.log(
                {
                    "Accuracy": accuracy[-1],
                    "Overall Loss": losses[-1],
                    "Loss prediction": losses_p[-1],
                    "loss state": losses_g[-1]
                }
            )
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].plot(range(episodes), accuracy)
    ax[0, 0].legend(["Accuracy"])
    ax[0, 1].plot(range(episodes), losses)
    ax[0, 1].legend(["Overall loss"])
    ax[1, 0].plot(range(episodes), losses_p)
    ax[1, 0].legend(["Losses prediction"])
    ax[1, 1].plot(range(episodes), losses_g)
    ax[1, 1].legend(["Losses state"])
    plt.show()

