import numpy as np

class FOPModel:
    def __init__(self, options):
        self.mode = options['mode']
        if self.mode == 'mdp':
            self.n_obs = (options['obs_size'] // 2) ** 2
            self.mdp_mat = np.arange(self.n_obs).reshape((options['obs_size'] // 2, options['obs_size'] // 2))
        else:
            self.n_obs = options['obs_size']
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
    
    def predict(self, obs, dir):
        preds = []
        if self.mode == 'pomdp':
            for i in range(obs.shape[0]):
                pred_ = self.count_matrix[self.prev_obs[i], dir[i]].argmax()
                preds.append(pred_ if pred_ < self.n_obs-2 else pred_ - self.n_obs)
        else:
            for i in range(obs.shape[0]):
                enc_pred_obs = self.mdp_mat[self.prev_obs[i, 0], self.prev_obs[i, 1]]
                pred_ = self.count_matrix[enc_pred_obs, dir[i]].argmax()
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
        self.prev_obs = np.array(init_pos).astype(int)
        if self.mode == 'pomdp':
            for obs, dir in zip(obss.T, dirs.T):
                self.observe(obs, dir)
        else:
            for obs, dir in zip(np.transpose(obss, axes=(1, 0, 2)), dirs.T):
                self.observe(obs, dir)
        self.reset()
        return self
    
    def predict_sequence(self, dirs, init_pos):
        result = None
        dirs = dirs.astype(int)
        self.prev_obs = np.array(init_pos).astype(int)
        for dir in dirs.T:
            if result is not None:
                result += [self.predict(self.prev_obs, dir)]
            else:
                result = [self.predict(self.prev_obs, dir)]
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
        
        self._rng = np.random.default_rng(seed=42) 
    
    def observe_sequence(self, obss, dirs, init_pos):
        pass
    
    def predict_sequence(self, dirs, init_pos):
        return self._rng.integers(self.n_obs, size=(dirs.shape[0], dirs.shape[1], 2 if self.mode == 'mdp' else 1))