import numpy as np
from typing import Literal, Optional
from scipy.special import logsumexp

INI_MODE = Literal['dirichlet', 'normal', 'uniform']

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
            pseudocount: float = 1e-10,
            seed: Optional[int] = None,
            mode = 'pomdp'
    ):
        self.mode = mode
        if self.mode == 'mdp':
            self.n_observations = (n_observations // 2) ** 2
        else:
            self.n_observations = n_observations
        self.len_room = n_observations // 2
        self.n_actions = n_actions
        self.cells_per_column = cells_per_column
        self.n_states = cells_per_column * self.n_observations
        
        self.lr = lr
        self.batch_size = batch_size
        self.pseudocount = pseudocount
        self._rng = np.random.default_rng(seed)
        
        # Initialize state prior
        
        self.state_prior = np.ones(self.n_states) / self.n_states
        # Initialize transition matrix
        if initialization == 'dirichlet':
            self.transition_probs = self._rng.dirichlet(
                [alpha] * self.n_states,
                size=(self.n_states, n_actions)
            )
        elif initialization == 'normal':
            logits = self._rng.normal(0, sigma, (self.n_states, n_actions, self.n_states))
            self.transition_probs = np.exp(logits - logsumexp(logits, axis=2, keepdims=True))
        else:  # uniform
            self.transition_probs = np.full(
                (self.n_states, n_actions, self.n_states),
                1.0 / self.n_states
            )
        
        # Apply pseudocount and normalize
        self.transition_probs = (self.transition_probs + pseudocount) / \
                              (1 + pseudocount * self.n_states)
        self.log_transition = np.log(self.transition_probs)

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
                
            gamma, xi_list = self.e_step(obs_seq, act_seq)
            self.m_step(obs_seq, act_seq, gamma, xi_list)

    def e_step(self, obs_seq: np.ndarray, act_seq: np.ndarray) -> tuple:
        T = len(obs_seq)
        alpha = self.forward_pass(obs_seq, act_seq)
        beta = self.backward_pass(obs_seq, act_seq)
        
        # Compute gamma
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        
        # Compute xi
        xi_list = []
        for t in range(T-1):
            prev_col = self._get_column_indices(obs_seq[t])
            curr_col = self._get_column_indices(obs_seq[t+1])
            a = int(act_seq[t])
            
            # Get transition block
            transition_block = self.transition_probs[prev_col[:, None], a, curr_col]

            # print(beta[t+1][:, None].T.shape)
            # print(transition_block.shape)
            # print(alpha[t][:, None].shape)
            xi_t = alpha[t][:, None] * transition_block * beta[t+1][:, None].T
            xi_t /= np.sum(xi_t)
            xi_list.append(xi_t)
        return gamma, xi_list

    def _get_column_indices(self, obs_idx: int) -> np.ndarray:
        """Get state indices for a given observation"""
        return np.arange(
            obs_idx * self.cells_per_column,
            (obs_idx + 1) * self.cells_per_column
        )

    def forward_pass(self, obs_seq: np.ndarray, act_seq: np.ndarray) -> np.ndarray:
        T = len(obs_seq)
        alpha = np.zeros((T, self.cells_per_column))

            
        # Initialize with first observation
        first_col = self._get_column_indices(obs_seq[0])
        alpha[0] = self.state_prior[first_col].copy()
        alpha[0] /= np.sum(alpha[0])
        
        # Forward recursion
        for t in range(1, T):
            prev_col = self._get_column_indices(obs_seq[t-1])
            curr_col = self._get_column_indices(obs_seq[t])
            a = int(act_seq[t-1])
            
            # Get transition block
            transition_block = self.transition_probs[prev_col[:, None], a, curr_col].T
            
            # Update alpha
            alpha[t] = alpha[t-1] @ transition_block
            alpha[t] /= np.sum(alpha[t])
            
        return alpha

    def backward_pass(self, obs_seq: np.ndarray, act_seq: np.ndarray) -> np.ndarray:
        T = len(obs_seq)
        beta = np.ones((T, self.cells_per_column)) / self.cells_per_column
        # Backward recursion
        for t in range(T-2, -1, -1):
            curr_col = self._get_column_indices(obs_seq[t])
            next_col = self._get_column_indices(obs_seq[t+1])
            a = int(act_seq[t])
            
            # Get transition block
            transition_block = self.transition_probs[curr_col[:, None], a, next_col]
            
            # Update beta
            beta[t] = transition_block @ beta[t+1]
            beta[t] /= np.sum(beta[t])
            
        return beta

    def m_step(self, obs_seq: np.ndarray, act_seq: np.ndarray, 
               gamma: np.ndarray, xi_list: list) -> None:
        # Update state prior for first observation
        first_col = self._get_column_indices(obs_seq[0])
        self.state_prior[first_col] = (1 - self.lr) * self.state_prior[first_col] + self.lr * gamma[0]
        self.state_prior /= np.sum(self.state_prior)

        # Update transition matrix
        for t in range(len(xi_list)):
            prev_col = self._get_column_indices(obs_seq[t])
            curr_col = self._get_column_indices(obs_seq[t+1])
            a = int(act_seq[t])
            
            # Apply update with learning rate
            self.transition_probs[prev_col[:, None], a, curr_col] = \
                (1 - self.lr) * self.transition_probs[prev_col[:, None], a, curr_col] + \
                self.lr * xi_list[t]
            
        # Apply pseudocount and normalize
        self.transition_probs /= self.transition_probs.sum(axis=(1, 2), keepdims=True)
        self.log_transition = np.log(self.transition_probs)
    
    
    def predict_sequence(self, act_seqs: np.ndarray, init_pos: np.ndarray) -> np.ndarray:
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
        batch_size = act_seqs.shape[0]
        action_seq_len = act_seqs.shape[1]
        obs_seq_len = action_seq_len + 1
        
        # Initialize output array
        pred_seqs = np.zeros((batch_size, obs_seq_len), dtype=int)
        pred_seqs[:, 0] = init_pos
        
        for i in range(batch_size):
            # Initialize state distribution
            current_col = self._get_column_indices(init_pos[i])
            state_probs = np.zeros(self.n_states)
            
            for j in range(action_seq_len):
                a = int(act_seqs[i, j])
                
                if j == 0: # initial position
                    state_probs[current_col] = self.state_prior[current_col]
                    state_probs /= np.sum(state_probs)
                else:
                    alpha = self.forward_pass(pred_seqs[i, :j+1], act_seqs[i, :j+1])
                    state_probs[current_col] = alpha[-1]
                next_state_probs = (state_probs[None, :] @ self.transition_probs[:, a, :]).squeeze()
                # Convert to observation probabilities
                obs_probs = np.zeros(self.n_observations)
                for obs_idx in range(self.n_observations):
                    col = self._get_column_indices(obs_idx)
                    obs_probs[obs_idx] = np.sum(next_state_probs[col])
                
                # Select most probable observation
                next_obs = np.argmax(obs_probs)
                pred_seqs[i, j+1] = next_obs
                
                # Update state distribution
                state_probs = np.zeros(self.n_states)
                next_col = self._get_column_indices(next_obs)
                state_probs[next_col] = next_state_probs[next_col]
                state_probs /= np.sum(state_probs)
        if self.mode == 'mdp':
            pred_seqs = np.stack((pred_seqs // self.len_room, pred_seqs % self.len_room), axis=-1)
        else:
            pred_seqs = pred_seqs[:, :, None]
        return pred_seqs