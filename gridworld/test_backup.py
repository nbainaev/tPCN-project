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
from src.agents.backup_agent_torch import tPCAgent
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
    render_mode: Optional[str] = Nones
) -> gym.vector.VectorEnv:
 
    class WrappedSyncVectorEnv(SyncVectorEnv):

        def __getattr__(self, name):
            if name.startswith('_'):
                raise AttributeError(name)
 
            def handler(*args, **kwargs):
                results = []
                for env in self.envs:
                    if hasattr(env, name):
                        results.append(getattr(env, name)(*args, **kwargs))
                return results
            return handler


    env_fns = [lambda: env_creator() for _ in range(num_envs)]
    

    envs = WrappedSyncVectorEnv(env_fns)
    

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
    conf_mdp_path = 'config/mdp.yaml'
    conf_pomdp_path = 'config/pomdp.yaml'
    setup_path = 'config/free.yaml'
    run_path = 'config/run.yaml'
    agent_conf_path = 'config/agent_conf.yaml'
    seed = 10
    
    
    
    setup = read_config(setup_path)
    run = read_config(run_path)
    agent_conf = read_config(agent_conf_path)
    log_update_rate = run["log_update_rate"]

    if run['mdp_mode'] == 'mdp':
        n_obs = 2*len(setup['room'][0][0])
        conf = read_config(conf_mdp_path)
    else:
        n_obs = len(np.unique(setup['room'])) + 1
        conf = read_config(conf_pomdp_path)
    if "start_position" in setup:
        ini_pos = setup.pop("start_position")
    else:
        ini_pos = (None, None)


    envs = make_sync_envs(
        env_creator=lambda: GridWorld(setup['room'], **conf),
        num_envs=run['batch_size'],
    )
    if run['log']:
        logger = wandb.init(
            project=run['project_name'], entity=os.environ['WANDB_ENTITY'],
            config=conf
        )
    else:
        logger = None
    episodes = run['episodes']
    steps = run['steps']

    agent = tPCAgent(n_obs=n_obs, mdp_mode=run['mdp_mode'], batch_size=run['batch_size'], **agent_conf)
    legend = []
    losses = np.array([])
    losses_p = np.array([])
    losses_g = np.array([])
    accuracy = np.array([])
    agent_path = []
    remembered = False
    for episode in tqdm(range(episodes), desc="Processing"):

        if ini_pos is not None:
            observations, infos = envs.reset(options={'start_r': ini_pos[0], 'start_c': ini_pos[1]})
        else:
            observations, infos = envs.reset()
        
        truncated = False
        terminated = False
        agent.reset()
        pred_observations = []
        true_observations = []

        active_episodes = [False] * run['batch_size']
        for step in range(steps):
            
            if step > 0:
                if remembered == False:
                    action = agent.act(4)
                    agent_path.append(action)
                else:
                    action = agent_path[step-1]
                observation, observation_proba = agent.predict_observation(action)
                if run['mdp_mode'] == 'mdp':
                    observation = observation.numpy().astype(np.float32)
                else:
                    observation = np.array(observation, dtype=np.float32)
                for i, active in enumerate(active_episodes):
                    if active:
                        observation[i] = np.nan
                pred_observations.append(observation)
                

                observations, rewards, terminated, truncated, infos = envs.step(action)
            obs = observations[0] if isinstance(observations, list) else observations.squeeze()
            
            
            if step > 0:
                obs = np.array(obs, dtype=np.float32)
                for i, active in enumerate(active_episodes):
                    if active:
                        obs[i] = np.nan
                true_observations.append(obs)
                agent.process_step(obs, action, active_episodes)
                for i, term in enumerate(terminated):
                    if term:
                        active_episodes[i] = True
            elif step == 0:
                prev_obs = obs
                agent.tpcn.update_memory(obs=obs, active_episodes=active_episodes)

            if np.all(active_episodes):
                break

        remembered = True
        losses = np.append(losses, agent.losses[-1])
        losses_p = np.append(losses_p, agent.losses_p[-1])
        losses_g = np.append(losses_g, agent.losses_g[-1])
        true_observations = np.array(true_observations)
        pred_observations = np.array(pred_observations)
        # print("true observations: ", true_observations[:, 1])
        # print("Predicted observation: ", pred_observations[:, 1])
        acc_buf = []
        if true_observations.ndim == 2:
            true_observations = true_observations[:, :, np.newaxis]
            pred_observations = pred_observations[:, :, np.newaxis]
        elif true_observations.ndim == 1:
            true_observations = true_observations[:, np.newaxis, np.newaxis]
            pred_observations = pred_observations[:, np.newaxis, np.newaxis]
        for i in range(run['batch_size']):
            true_obs = true_observations[:, i]
            pred_obs = pred_observations[:, i]
            acc_buf.append(np.where(~np.isnan(true_obs) & ~np.isnan(pred_obs), true_obs == pred_obs, False).min(axis=1).sum() / len(pred_obs))
        acc_buf = np.array(acc_buf)
        accuracy = np.append(accuracy, acc_buf.mean())

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

