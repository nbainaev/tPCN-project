import numpy as np
from typing import Literal, Optional
from scipy.special import logsumexp

INI_MODE = Literal['dirichlet', 'normal', 'uniform']

def softmax(x: np.ndarray, temp=1.) -> np.ndarray:
    """Computes softmax values for a vector `x` with a given temperature."""
    temp = np.clip(temp, 1e-5, 1e+3)
    e_x = np.exp((x - np.max(x, axis=-1)) / temp)
    return e_x / e_x.sum(axis=-1)

class CHMMGridWorld:
    def __init__(
            self,
            n_observations: int,
            n_actions: int,
            cells_per_column: int,
            lr: float = 0.1,
            batch_size: int = 1,
            initialization: Literal['dirichlet', 'normal', 'uniform'] = 'dirichlet',
            sigma: float = 1.0,
            alpha: float = 1.0,
            pseudocount: float = 1e-12,
            seed: Optional[int] = None,
            mode = 'pomdp'
    ):
        self.mode = mode
        if self.mode == 'mdp':
            self.n_observations = (n_observations // 2) ** 2
        else:
            self.n_observations = n_observations
            self.obs_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: -2, 5: -1}
        self.len_room = n_observations // 2
        self.n_actions = n_actions
        self.cells_per_column = cells_per_column
        self.n_states = cells_per_column * self.n_observations
        self.initialization = initialization
        self.lr = lr
        self.batch_size = batch_size
        self.pseudocount = pseudocount
        self._rng = np.random.default_rng(seed)
        self.column_indices = [
            np.arange(i*self.cells_per_column, (i+1)*self.cells_per_column)
            for i in range(self.n_observations)
        ]

        if self.initialization == 'dirichlet':
            self.transition_probs = self._rng.dirichlet(
                alpha=[alpha]*self.n_states,
                size=(self.n_states, self.n_actions)
            )
            self.state_prior = self._rng.dirichlet(alpha=[alpha]*self.n_states)
        elif self.initialization == 'normal':
            self.log_transition_factors = self._rng.normal(
                scale=sigma,
                size=(self.n_states, self.n_actions, self.n_states)
            )
            self.log_state_prior = self._rng.normal(scale=sigma, size=self.n_states)
        
        elif self.initialization == 'uniform':
            self.log_transition_factors = np.zeros((self.n_states, self.n_actions, self.n_states))
            self.log_state_prior = np.zeros(self.n_states)

        if self.initialization != 'dirichlet':
            self.transition_probs = np.zeros(self.log_transition_factors.shape)
            for a in range(self.n_actions):
                self.transition_probs[:, a, :] = np.vstack(
                    [softmax(x) for x in self.log_transition_factors[:, a, :]]
                )
                self.state_prior = softmax(self.log_state_prior)
        else:
            self.log_transition_factors = np.log(self.transition_probs)
            self.log_state_prior = np.log(self.state_prior)

    def _get_column_indices(self, obs_idx: int) -> np.ndarray:
        """Get state indices for a given observation"""
        return self.column_indices[obs_idx]
    
    def forward_pass(self, obs_seq: np.ndarray, act_seq: np.ndarray) -> np.ndarray:
        T = len(obs_seq)
        alpha = np.zeros((T, self.cells_per_column))

            
        # Initialize with first observation
        first_col = self._get_column_indices(obs_seq[0])
        alpha[0] = self.state_prior[first_col].copy().reshape(1, self.cells_per_column)
        alpha[0] /= np.sum(alpha[0] + self.pseudocount)
        
        # Forward recursion
        for t in range(1, T):
            prev_col = self._get_column_indices(obs_seq[t-1])
            curr_col = self._get_column_indices(obs_seq[t])
            a = int(act_seq[t-1])
            transition_block = self.transition_probs[:, a, :]
            # Get transition block
            transition_block = transition_block[np.ix_(prev_col, curr_col)]
            
            # Update alpha
            alpha[t] = alpha[t-1][None, :] @ transition_block
            alpha[t] /= np.sum(alpha[t] + self.pseudocount)
            
        return alpha

    def backward_pass(self, obs_seq: np.ndarray, act_seq: np.ndarray) -> np.ndarray:
        T = len(obs_seq)
        beta = np.zeros((T, self.cells_per_column))
        beta[-1] = 1
        beta[-1] /= beta[-1].sum()
        # Backward recursion
        for t in range(T-2, -1, -1):
            curr_col = self._get_column_indices(obs_seq[t])
            next_col = self._get_column_indices(obs_seq[t+1])
            a = int(act_seq[t])
            # Get transition block
            transition_block = self.transition_probs[:, a, :]
            transition_block = transition_block [np.ix_(curr_col, next_col)]
            
            # Update beta
            beta[t] = (transition_block @ beta[t+1][:, None]).T
            beta[t] /= np.sum(beta[t] + self.pseudocount)
            
        return beta

    def e_step(self, obs_seq: np.ndarray, act_seq: np.ndarray) -> tuple:
        T = len(obs_seq)
        alpha = self.forward_pass(obs_seq, act_seq)
        beta = self.backward_pass(obs_seq, act_seq)
        
        gamma = np.zeros((T, self.cells_per_column))
        xi = np.zeros((T-1, self.n_states, self.n_actions, self.n_states))
        # Compute gamma

        for t in range(T):
            gamma[t] = alpha[t] * beta[t]
            gamma[t] /= (gamma[t] + self.pseudocount).sum()
        # Compute xi
        for t in range(T-1):
            prev_col = self._get_column_indices(obs_seq[t])
            curr_col = self._get_column_indices(obs_seq[t+1])
            a = int(act_seq[t])
            
            # Get transition block
            transition_block = self.transition_probs[:, a, :]
            transition_block = transition_block[np.ix_(prev_col, curr_col)]

            xi_t = alpha[t][:, None] * transition_block * beta[t+1][None, :]
            xi_t /= np.sum(xi_t + self.pseudocount)
            xi[t, :, a, :][np.ix_(prev_col, curr_col)] = xi_t
        return gamma, xi

    def m_step(self, obs_seq: np.ndarray, act_seq: np.ndarray, 
               gamma: np.ndarray, xi: list) -> None:
        # Update state prior for first observation
        first_col = self._get_column_indices(obs_seq[0])
        self.state_prior[first_col] = gamma[0]
        self.state_prior /= np.sum(self.state_prior + self.pseudocount)
        sum_xi = xi.sum(axis=0)
        # Update transition matrix
        self.transition_probs = self.lr * self.transition_probs + (1 - self.lr) * sum_xi
        
        # Apply pseudocount and normalize
        self.transition_probs /= (self.transition_probs + self.pseudocount).sum(axis=(1, 2), keepdims=True)
        self.log_transition = np.log(self.transition_probs + self.pseudocount)
    
    def observe_sequence(self, obs_seqs: np.ndarray, act_seqs: np.ndarray) -> None:
        """
        Process multiple observation-action sequences (batch processing)
        
        Args:
            obs_seqs: 2D array of shape (batch_size, seq_len_obs)
            act_seqs: 2D array of shape (batch_size, seq_len_act)
        """
        # Transform initial positions to 1D array (batch_size, len_sequence)
        if self.mode == 'mdp':
            obs_seqs = self.len_room * obs_seqs[..., 0] + obs_seqs[..., 1]
        if obs_seqs.shape[0] != act_seqs.shape[0]:
            raise ValueError("Batch sizes for observations and actions must match")
        
        for obs_seq, act_seq in zip(obs_seqs, act_seqs):
            if len(obs_seq) != len(act_seq) + 1:
                raise ValueError("Each observation sequence should be 1 longer than action sequence")
                
            gamma, xi = self.e_step(obs_seq, act_seq)
            self.m_step(obs_seq, act_seq, gamma, xi)
    
    def predict_sequence(self, act_seqs: np.ndarray, obs_seqs, init_pos: np.ndarray) -> np.ndarray:
        """
        Predict sequences of observations given action sequences and initial positions
        
        Args:
            act_seqs: 2D array of shape (batch_size, seq_len_actions)
            init_pos: 1D array of shape (batch_size,) with initial observation indices
            
        Returns:
            2D array of shape (batch_size, seq_len_obs) with predicted observations
            where seq_len_obs = seq_len_actions + 1
        """
        # Transform initial positions to 1D array (batch_size, )
        if self.mode == 'mdp':
            init_pos = self.len_room * init_pos[..., 0] + init_pos[..., 1]
            obs_seqs = self.len_room * obs_seqs[..., 0] + obs_seqs[..., 1]
        
        obs_seqs = np.concatenate((init_pos[:, None], obs_seqs), axis=1)
        batch_size = act_seqs.shape[0]
        action_seq_len = act_seqs.shape[1]
        obs_seq_len = action_seq_len + 1
        
        # Initialize output array
        pred_seqs = np.zeros((batch_size, obs_seq_len), dtype=int)
        pred_seqs[:, 0] = init_pos

        for i in range(batch_size):
            for j in range(action_seq_len):
                current_col = self._get_column_indices(obs_seqs[i, j])
                a = int(act_seqs[i, j])
                alpha = self.forward_pass(obs_seqs[i, :j+1], act_seqs[i, :j])
                state_probs = np.zeros((self.n_states, ))
                state_probs[current_col] = alpha[-1]
                state_probs /= np.sum(state_probs + self.pseudocount)
                next_state_probs = (state_probs[None, :] @ self.transition_probs[:, a, :]).squeeze()
                
                # Convert to observation probabilities
                obs_probs = np.zeros(self.n_observations)
                for obs_idx in range(self.n_observations):
                    col = self._get_column_indices(obs_idx)
                    obs_probs[obs_idx] = next_state_probs[col].sum()
                
                # Select most probable observation
                next_obs = np.argmax(obs_probs)
                pred_seqs[i, j+1] = next_obs

            if self.mode == 'pomdp':
                pred_seqs[i] = np.array([self.obs_dict[int(val)] for val in pred_seqs[i]])
        if self.mode == 'mdp':
            pred_seqs = np.stack((pred_seqs // self.len_room, pred_seqs % self.len_room), axis=-1)
        else:
            pred_seqs = pred_seqs[:, :, None]
        return pred_seqs