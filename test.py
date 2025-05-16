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

os.environ['WANDB_ENTITY'] = 'nik-baynaev-national-research-nuclear-university-mephi'

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

    env = GridWorld(setup['room'], **conf)
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
    agent = tPCAgent(n_obs=n_obs, **agent_conf)
    legend = []
    losses = []
    losses_p = []
    losses_g = []
    accuracy = []
    # run several episodes
    for episode in tqdm(range(episodes), desc="Processing"):
        env.reset(*ini_pos)
        agent.reset()
        actions = []
        pred_observations = []
        true_observations = []
        # do steps in one episode
        for step in range(steps):
            if step > 0:
                # choose any random action
                action = agent.act(len(env.actions))
                actions.append(action)
                observation, observation_proba = agent.predict_observation(action)
                pred_observations.append(observation)
                env.act(action)
            env.step()
            obs, reward, is_terminal = env.obs()
            obs = obs[0, 0]
            true_observations.append(obs)
            if step > 0:
                # if obs < 0:
                #     obs = prev_obs
                agent.process_step(obs, action)
                # print(agent.tpcn.obs_encode_dict)
                # print(observation_proba.T, observation, obs[0, 0])
            elif step == 0:
                prev_obs = obs
                agent.tpcn.update_memory(obs=obs)
            
            # finish episode early if terminal state is entered
            if is_terminal:
                break
        losses.append(agent.losses[-1])
        losses_p.append(agent.losses_p[-1])
        losses_g.append(agent.losses_g[-1])
        accuracy.append(sum([1 if x == y 
                             else 0 
                             for x, y in zip(true_observations[1:], pred_observations)]) / len(pred_observations))
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
    

