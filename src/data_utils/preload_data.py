import gymnasium as gym
import numpy as np
import torch
import os
from typing import Callable, Optional, Any
from gymnasium.vector import SyncVectorEnv
from sklearn.preprocessing import OneHotEncoder


def make_sync_envs(
    env: Any,
    confs: list,  # Теперь принимаем список конфигураций вместо одной
    room: Any = None,
    num_envs: int = 4,
    render_mode: Optional[str] = None
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
        
        def reset(self, *, seed=None, options=None, **kwargs):
            # Полностью переопределяем логику reset
            if isinstance(seed, int):
                seed = [seed + i for i in range(self.num_envs)]
            elif seed is None:
                seed = [None] * self.num_envs
            
            if options is not None and not isinstance(options, list):
                options = [options.copy() for _ in range(self.num_envs)]
            
            observations = []
            infos = {}
            for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
                single_options = options[i] if options else None
                obs, info = env.reset(seed=single_seed, options=single_options)
                observations.append(obs)
                infos = self._add_info(infos, info, i)
            
            return np.array(observations), infos

    # Создаем env_fns с разными конфигурациями
    env_fns = [lambda i=i: env(room, **confs[i]) for i in range(num_envs)]
    
    envs = WrappedSyncVectorEnv(env_fns)
    return envs


class ObservationEncoder:
    def __init__(self, mode, collision_hint):
        self.mode = mode
        self.collision_hint = collision_hint
        if mode == "pomdp":
            if collision_hint:
                self.obs_encoder = OneHotEncoder(categories=[[-2, -1, 0, 1, 2, 3]], sparse_output=False)
                self.obs_encoder.fit(np.array([-2, -1, 0, 1, 2, 3]).reshape(-1, 1))
            else:
                self.obs_encoder = OneHotEncoder(categories=[[0, 1, 2, 3]], sparse_output=False)
                self.obs_encoder.fit(np.array([0, 1, 2, 3]).reshape(-1, 1))
        else:
            self.obs_encoder = [OneHotEncoder(categories=[[0, 1, 2, 3, 4]], sparse_output=False), 
                OneHotEncoder(categories=[[0, 1, 2, 3, 4]], sparse_output=False)]
            self.obs_encoder[0].fit(np.arange(5).reshape(-1, 1))
            self.obs_encoder[1].fit(np.arange(5).reshape(-1, 1))
    
    def transform(self, data):
        if self.mode == "mdp":
            vec_x = self.obs_encoder[0].transform(data[:, 0].reshape(-1, 1)) / 2
            vec_y = self.obs_encoder[1].transform(data[:, 1].reshape(-1, 1)) / 2
            data_trans = np.concatenate((vec_x, vec_y), axis=1)
        elif self.mode == "pomdp":
            data_trans = self.obs_encoder.transform(data.reshape((-1, 1)))
        return data_trans
    
    def inverse_transform(self, data):
        if self.mode == "mdp":
            vec_x = data[:, :data.shape[1] // 2] + 1.0e-8
            vec_y = data[:, data.shape[1] // 2:] + 1.0e-8
            for i in range(vec_x.shape[0]):
                if (vec_x[i] == 0).all():
                    vec_x[i, 0] == 0.5
                if (vec_y[i] == 0).all():
                    vec_y[i, 0] == 0.5
            preds_x = self.obs_encoder[0].inverse_transform(vec_x)
            preds_y = self.obs_encoder[1].inverse_transform(vec_y)
            preds = np.concatenate((preds_x, preds_y), axis=1)
        elif self.mode == "pomdp":
            preds = self.obs_encoder.inverse_transform(data)
        return preds

class TrajectoryGenerator:
    def __init__(self, env, options):
        self.n_steps = options['n_steps']
        self.batch_size = options['batch_size']
        self.sequence_len = options['sequence_length']
        self.dir_encoder = OneHotEncoder(categories=[[0, 1, 2, 3]], sparse_output=False)
        self.dir_encoder.fit(np.array([0, 1, 2, 3]).reshape(-1, 1))
        self.obs_encoder = ObservationEncoder(options['mode'], options['conf']['collision_hint'])
        self.obs_size = options['obs_size']
        self.dir_size = options['dir_size']
        self.use_preloaded = options['use_preloaded']
        if options['use_preloaded']:
            self.n_samples = self.batch_size * self.n_steps
        else:
            self.n_samples = self.batch_size
        self.envs = make_sync_envs(
            env=env,
            confs=[options['conf']] * self.n_samples,
            room=options['room'],
            num_envs=self.n_samples,
        )
        self._rng = np.random.default_rng(seed=42)
    
    def generate_traj_data(self, ini_pos, save=True, encode=True, path=None):
        data = {'actions': None, 'obs': None, 'init_obs': None}
        active_episodes = np.array([True] * self.batch_size)
        if ini_pos is not None:
            if isinstance(ini_pos, np.ndarray) and ini_pos.shape[0] == self.batch_size:
                option_list = [{'start_r': ini_pos[i, 0], 'start_c': ini_pos[i, 1]} for i in range(ini_pos.shape[0])]
                observations, infos = self.envs.reset(options=option_list)
            else:
                observations, infos = self.envs.reset(options={'start_r': ini_pos[0], 'start_c': ini_pos[1]})
        else:
            observations, infos = self.envs.reset()
        if encode:
            data['init_obs'] = self.obs_encoder.transform(observations)
            #seq = observations[:, np.newaxis, :]
            for step in range(self.sequence_len):
                
                action = self._rng.integers(4, size=self.n_samples)
                action_enc = self.dir_encoder.transform(action.reshape(-1, 1))
                
                observations, rewards, terminated, truncated, infos = self.envs.step(action)
                observations_enc = self.obs_encoder.transform(observations)
                observations_enc[~active_episodes, :] = np.nan
                action_enc[~active_episodes, :] = np.nan
                active_episodes &= ~terminated
                if data['actions'] is not None:
                    data['actions'] = np.concatenate((data['actions'], action_enc[:, np.newaxis, :]), axis=1)
                else:
                    data['actions'] = action_enc[:, np.newaxis, :]
                if data['obs'] is not None:
                    data['obs'] = np.concatenate((data['obs'], observations_enc[:, np.newaxis, :]), axis=1)
                else:
                    data['obs'] = observations_enc[:, np.newaxis, :]
        else:
            data['init_obs'] = observations
            for step in range(self.sequence_len):
                
                action = self._rng.integers(4, size=self.n_samples)
                
                observations, rewards, terminated, truncated, infos = self.envs.step(action)
                action[~active_episodes] = np.nan
                observations[~active_episodes] = np.nan
                active_episodes &= ~terminated
                if data['actions'] is not None:
                    data['actions'] = np.concatenate((data['actions'], action[:, np.newaxis]), axis=1)
                else:
                    data['actions'] = action[:, np.newaxis]
                if data['obs'] is not None:
                    data['obs'] = np.concatenate((data['obs'], observations[:, np.newaxis]), axis=1)
                else:
                    data['obs'] = observations[:, np.newaxis]
        
        if self.use_preloaded:
            if save:
                if path is not None:
                    if not os.path.exists(path):
                        os.mkdir(path)
                    np.save(path + "\\directions.npy", data['actions'])
                    np.save(path + "\\observations.npy", data['obs'])
                    np.save(path + "\\init_pos.npy", data['init_obs'])
                else:
                    path = "saved_data"
                    if not os.path.exists(path):
                        os.mkdir(path)

                    np.save(path + "\\directions.npy", data['actions'])
                    np.save(path + "\\observations.npy", data['obs'])
                    np.save(path + "\\init_pos.npy", data['init_obs'])
            return torch.tensor(data['actions'], dtype=torch.float32), torch.tensor(data['obs'], dtype=torch.float32),\
                   torch.tensor(data['init_obs'], dtype=torch.float32)
        else:
            if not save:
                return torch.tensor(data['actions'], dtype=torch.float32), torch.tensor(data['obs'], dtype=torch.float32),\
                   torch.tensor(data['init_obs'], dtype=torch.float32)
            return torch.tensor(data['actions'], dtype=torch.float32), torch.tensor(data['obs'], dtype=torch.float32),\
                   torch.tensor(data['init_obs'], dtype=torch.float32)
    
    def decode_trajectory(self, data):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        data_dec = None
        for seq in data:
            if data_dec is None:
                decoded_seq = np.full((seq.shape[0], 1), np.nan)
                mask = ~np.isnan(seq)
                seq = seq[mask].reshape(-1, self.obs_size)
                if len(seq):
                    decoded_seq[np.unique(np.where(mask)[0])] = self.obs_encoder.inverse_transform(seq)
                data_dec = decoded_seq[np.newaxis, :, :]
            else:
                decoded_seq = np.full((seq.shape[0], 1), np.nan)
                mask = ~np.isnan(seq)
                seq = seq[mask].reshape(-1, self.obs_size)
                if len(seq):
                    decoded_seq[np.unique(np.where(mask)[0])] = self.obs_encoder.inverse_transform(seq)
                data_dec = np.concatenate((data_dec, decoded_seq[np.newaxis, :, :]), axis=0)
        return data_dec


class Trajectory(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = {}
        self.v = np.load(path + "\\directions.npy")
        self.obs = np.load(path + "\\observations.npy")
        self.init_obs = np.load(path + "\\init_pos.npy")
    
    def __len__(self):

        return self.obs.shape[0]

    def __getitem__(self, idx):
        sample_v = torch.tensor(self.v[idx]).to(torch.float32)
        sample_obs = torch.tensor(self.obs[idx]).to(torch.float32)
        sample_init_obs = torch.tensor(self.init_obs[idx]).to(torch.float32)

        return sample_v, sample_obs, sample_init_obs

def get_traj_loader(path, options):
    # Create a Trajectory dataset
    dataset = Trajectory(path)

    # Create a DataLoader from the Trajectory dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=options['batch_size'], shuffle=True)

    return loader


    