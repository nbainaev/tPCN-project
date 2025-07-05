import numpy as np

class FOPModel:
    def __init__(self, options):
        self.mode = options['mode']
        if self.mode == 'mdp':
            self.n_obs = (options['obs_size'] // 2) ** 2
            self.mdp_mat = np.arange(self.n_obs).reshape((options['obs_size'] // 2, options['obs_size'] // 2))
        else:
            self.n_obs = options['obs_size']
            self.obs_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: -2, 5: -1}
        self.dir = options['dir_size']
        self.count_matrix = np.zeros((self.n_obs, self.dir, self.n_obs))
        self.prev_obs = None
        self.delta = 0.001
    
    def reset(self):
        self.prev_obs = None
    
    def first_observation(self, obs):
        self.prev_obs = obs
        return self
    
    def observe(self, obs, dir):
        if self.mode == 'pomdp':
            for i in range(obs.shape[0]):
                self.count_matrix[self.prev_obs[i], dir[i], obs[i]] += self.delta
            self.prev_obs = obs
        else:
            for i in range(obs.shape[0]):
                enc_prev_obs = self.mdp_mat[self.prev_obs[i, 0], self.prev_obs[i, 1]]
                enc_obs = self.mdp_mat[obs[i, 0], obs[i, 1]]
                self.count_matrix[enc_prev_obs, dir[i], enc_obs] += self.delta
            self.prev_obs = obs
        return self
    
    def predict(self, dir):
        preds = []
        if self.mode == 'pomdp':
            for i in range(self.prev_obs.shape[0]):
                pred_ = self.count_matrix[self.prev_obs[i], dir[i]].argmax()
                preds.append(self.obs_dict[int(pred_)])
        else:
            for i in range(self.prev_obs.shape[0]):
                enc_prev_obs = self.mdp_mat[self.prev_obs[i, 0], self.prev_obs[i, 1]]
                pred_ = self.count_matrix[enc_prev_obs, dir[i]].argmax()
                pred_decoded = np.array(np.where(self.mdp_mat == pred_)).reshape(-1)
                preds.append(pred_decoded)
        return np.array(preds)
    
    def predict_proba(self, obs, dir):
        proba = []
        for i in range(obs.shape[0]):
            counts = self.count_matrix[self.prev_obs[i], dir[i]]
            proba.append(counts / counts.sum())
        return np.array(proba)
    
    def observe_sequence(self, obss, dirs, init_pos):
        obss = obss.astype(int).squeeze()
        dirs = dirs.astype(int)
        self.prev_obs = init_pos.squeeze()
        if self.mode == 'pomdp':
            for obs, dir in zip(obss.T, dirs.T):
                self.observe(obs, dir)
        else:
            for obs, dir in zip(np.transpose(obss, axes=(1, 0, 2)), dirs.T):
                self.observe(obs, dir)
        self.reset()
        return self
    
    def predict_sequence(self, dirs, init_pos, obss = None, prediction_mode = 'online'):
        result = []
        self.prev_obs = init_pos.squeeze()
        dirs = dirs.astype(int)
        if prediction_mode == 'online':
            if obss is None:
                raise ValueError("With online prediction mode observation must be passed!")
            obss = obss.astype(int).squeeze()
            if self.mode == "mdp":
                obss = np.transpose(obss, axes=(1, 0, 2))
            else:
                obss = obss.T
            for obs, dir in zip(obss, dirs.T):
                result += [self.predict(dir)]
                self.prev_obs = obs
            self.reset()
            if self.mode == 'mdp':
                return np.transpose(np.array(result).squeeze(), axes=(1, 0, 2))
            else:
                return np.array(result).T[:, :, np.newaxis]
        else:
            for dir in dirs.T:
                result += [self.predict(dir)]
                self.prev_obs = result[-1]
            self.reset()
            if self.mode == 'mdp':
                return np.transpose(np.array(result).squeeze(), axes=(1, 0, 2))
            else:
                return np.array(result).T[:, :, np.newaxis]

class RandomModel:
    def __init__(self, options):
        self.mode = options['mode']
        if self.mode == 'mdp':
            self.n_obs = options['obs_size'] // 2
        else:
            self.n_obs = options['obs_size']
            self.obs_dict = {0: -1, 1: -1, 1: 0, 2: 1, 4: 2, 5: 3}
            self.min = min(list(self.obs_dict.values()))
            self.max = max(list(self.obs_dict.values()))
        self._rng = np.random.default_rng(seed=42) 

    def observe_sequence(self, obss, dirs, init_pos):
        pass
    
    def predict_sequence(self, dirs, init_pos, obs, prediction_mode):
        if self.mode == 'pomdp':
            return self._rng.integers(self.min, self.max, size=(dirs.shape[0], dirs.shape[1], 2 if self.mode == 'mdp' else 1))
        else:
            return self._rng.integers(0, self.n_obs, size=(dirs.shape[0], dirs.shape[1], 2))