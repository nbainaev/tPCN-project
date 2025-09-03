import torch
import numpy as np
from src.models.tpcn import TemporalPCN, HierarchicalPCN
from src.data_utils.data_utils import GridWorldEncoder
from enum import Enum, auto
from src.utils import safe_divide, softmax, sparse_to_dense

class ExplorationPolicy(Enum):
    SOFTMAX = 1
    EPS_GREEDY = auto()

class tPCNAgent:
    def __init__(self, options, model=None, init_model=None):
        self.options = options
        self.init_model = HierarchicalPCN(options) if init_model is None else init_model
        self.tpc_model = TemporalPCN(options) if model is None else model
        self.lr = options['learning_rate']
        self.cum_reward = 0
        self.n_actions = options['dir_size']
        self.n_obs = options['obs_size']
        self.hidden_dim = options['latent_size']
        self.exploration_eps = options['exploration_eps']
        self.device = options['device']
        self.observations = [[]]
        self.actions = [[]]
        self.episode = 1
        self.is_first = True
        self.batch_size = options['batch_size']
        self.inverse_temp = options['inverse_temp']
        self.dir_encoder = GridWorldEncoder(
            categories=[0, 1, 2, 3], 
            mode='directions', 
            encoder=options['encoder'])
        self.obs_encoder = GridWorldEncoder(
            categories= sorted(list(np.unique(options['room'][0]))  + [-1]) if options['mode'] == 'pomdp' else list(range(len(options['room'][0]) ** 2)),
            mode=options['mode'],
            collision_hint=options['conf']['collision_hint'],
            encoder=options['encoder']
        )
        self.test_inf_iters = options['test_inf_iters']
        self.inf_lr = options['inf_lr']
        self.inf_iters = options['inf_iters']
        self.tpc_optimizer = torch.optim.Adam(
            self.tpc_model.parameters(), 
            lr=self.lr,
        )
        self.init_optimizer = torch.optim.Adam(
            self.init_model.parameters(),
            lr=self.lr,
        )
        self.learn_rewards_from_state = options['learn_rewards_from_state']
        if self.learn_rewards_from_state:
            rewards = np.zeros((self.hidden_dim))
        else:
            rewards = np.zeros((self.n_obs))
        
        self.rewards = rewards.flatten()
        self.reward_lr = options['reward_lr']
        self.prev_hidden = None
        self.prev_encoded_dir = None
        self.action = None

        if self.exploration_eps < 0:
            self.exploration_policy = ExplorationPolicy.SOFTMAX
            self.exploration_eps = 0
        else:
            self.exploration_policy = ExplorationPolicy.EPS_GREEDY
            self.exploration_eps = self.exploration_eps
        self._rng = np.random.default_rng(seed=42)
        
    def observe(self, obs, reward, terminated, learn=True):

        self.cum_reward += reward
        encoded_obs = torch.tensor(self.obs_encoder.transform(obs), dtype=torch.float32).reshape(1, 1, -1)
        self.observations[-1].append(encoded_obs)
        if self.prev_encoded_dir is not None:
            self.actions[-1].append(self.prev_encoded_dir)

        if self.is_first:
            self.init_model.eval()
            with torch.no_grad():
                self.init_model.inference(self.inf_iters, self.inf_lr, encoded_obs)
            
            self.is_first = False
            self.prev_hidden = self.init_model.z.clone().detach()
        
        elif terminated:
            self.actions[-1] = self._list_to_tensor(self.actions[-1])
            self.observations[-1] = self._list_to_tensor(self.observations[-1])
            
            if learn and self.episode == self.batch_size:
                self.actions = self._list_to_tensor(self.actions)
                self.observations = self._list_to_tensor(self.observations)
                energy, loss = self.memorize()
                print("Memorized successfully")
            self.episode += 1
        else:
            self.tpc_model.eval()
            with torch.no_grad():
                self.tpc_model.inference(self.inf_iters, self.inf_lr, 
                                        self.prev_encoded_dir, self.prev_hidden, encoded_obs)
            
            # update the hidden state
            self.prev_hidden = self.tpc_model.z.clone().detach()

    def generate_sf(self, init_hidden, n_steps=5, gamma=0.95, learn_sr=False):
        zs = [init_hidden]
        sr = init_hidden
        discounts = [1.0]
        v = (torch.ones([self.tpc_model.Win.weight.shape[1]]) / self.tpc_model.Win.weight.shape[1]).reshape(1, -1)
        for _ in range(n_steps):
            zs.append(self.tpc_model.g(v, zs[-1]))

            if learn_sr:
                sr += discounts[-1] * zs[-1]
            discounts.append(discounts[-1] * gamma)
        
        zs = self.tpc_model.decode(torch.vstack(zs))
        discounts = torch.tensor(discounts).reshape(-1, 1, 1).expand(zs.shape)
        if learn_sr:
            return (zs * discounts).sum(dim=0), sr
        else:
            return (zs * discounts).sum(dim=0)

    def predict(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(self.dir_encoder.transform(np.array(action)), dtype=torch.float32).unsqueeze(0)
        pred_hidden = self.tpc_model.g(action, self.prev_hidden)
        return self.tpc_model.decode(pred_hidden), pred_hidden
    
    def sample_action(self):
        """Evaluate and sample actions."""
        self.action_values = self.evaluate_actions(n_steps=7, gamma=0.9)
        self.action_dist = self._get_action_selection_distribution(
            self.action_values, on_policy=True
        )
        self.action = self._rng.choice(self.n_actions, p=self.action_dist)
        self.prev_encoded_dir = torch.tensor(self.dir_encoder.transform(np.array(self.action)), dtype=torch.float32).reshape(1, 1, -1)
        return self.action
    
    def reinforce(self, reward):
        
        deltas = self.tpc_model.decode(self.prev_hidden).detach().numpy().squeeze() * (reward - self.rewards)
        self.rewards += self.reward_lr * deltas
    
    def evaluate_actions(self, n_steps=5, gamma=0.95):
        
        n_actions = self.n_actions
        action_values = np.zeros(n_actions)
        dense_action = torch.tensor(np.zeros_like(action_values), dtype=torch.float32)

        for action in range(n_actions):
            dense_action[action - 1] = 0
            dense_action[action] = 1

            pred_hidden = self.tpc_model.g(dense_action, self.prev_hidden)
            sf = self.generate_sf(
                init_hidden=pred_hidden, 
                n_steps=n_steps, 
                gamma=gamma).detach().numpy()

            action_values[action] = np.sum(
                    sf * self.rewards
                ) / self.n_obs
        return action_values

    def reset(self):
        self.is_first = True
        self.cum_reward = 0
        self.prev_hidden = None
        self.prev_encoded_dir = None
        self.action = None
        print(self.episode)
        if self.episode > self.batch_size:
            print(self.actions.shape, self.observations.shape)
            self.episode = 1
            self.observations = [[]]
            self.actions = [[]]
        else:
            print(self.actions[-1].shape, self.observations[-1].shape)
            self.actions.append([])
            self.observations.append([])

    def _get_action_selection_distribution(
            self, action_values, on_policy: bool = True
    ) -> np.ndarray:
        # off policy means greedy, on policy — with current exploration strategy
        if on_policy and self.exploration_policy == ExplorationPolicy.SOFTMAX:
            # normalize values before applying softmax to make the choice
            # of the softmax temperature scale invariant
            action_values = safe_divide(action_values, np.abs(action_values.sum()))
            action_dist = softmax(action_values, beta=self.inverse_temp)
        else:
            # greedy off policy or eps-greedy
            best_action = np.argmax(action_values)
            # make greedy policy
            # noinspection PyTypeChecker
            action_dist = sparse_to_dense([best_action], like=action_values)

            if on_policy and self.exploration_policy == ExplorationPolicy.EPS_GREEDY:
                # add uniform exploration
                action_dist[best_action] = 1 - self.exploration_eps
                action_dist[:] += self.exploration_eps / self.n_actions

        return action_dist
    
    def act(self, n_actions=4):
        action = np.array(np.random.randint(0, 3))
        self.prev_encoded_dir = torch.tensor(self.dir_encoder.transform(action), dtype=torch.float32)
        return int(action)
    
    def _check_shapes(self, lst: list) -> bool:
        if lst[0].shape[:2] == (1, 1):
            return False
        else:
            return True
    
    def _concat_with_padding(self, tensor_list: list) -> torch.Tensor:
        # Находим максимальный размер по оси N
        max_n = max(tensor.shape[1] for tensor in tensor_list)
        
        padded_tensors = []
        for tensor in tensor_list:
            current_n = tensor.shape[1]
            if current_n < max_n:
                # Создаем тензор с NaN значениями для дополнения
                pad_size = max_n - current_n
                nan_padding = torch.full((1, pad_size, tensor.shape[2]), float('nan'),
                                    dtype=tensor.dtype, device=tensor.device)
                # Конкатенируем по оси N
                padded_tensor = torch.cat([tensor, nan_padding], dim=1)
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(tensor)
        
        # Объединяем все тензоры в один
        return torch.cat(padded_tensors, dim=0)

    def _list_to_tensor(self, tensors: list) -> torch.Tensor:
        
        is_train = self._check_shapes(tensors)

        if not is_train:
            return torch.cat(tensors, dim=1)
        else:
            return self._concat_with_padding(tensors)

    def memorize(self):
        
        self.tpc_model.train()
        self.init_model.train()
        total_loss = 0 # average loss across time steps
        total_energy = 0 # average energy across time steps
        # train the initial static pcn to get the initial hidden state
        self.init_optimizer.zero_grad()

        init_actv = self.observations[:, 0, :]
        self.init_model.inference(self.inf_iters, self.inf_lr, init_actv)
        energy, obs_loss = self.init_model.get_energy()
        energy.backward()
        self.init_optimizer.step()
        
        total_loss += obs_loss.item()
        total_energy += energy.item()
        # get the initial hidden state from the initial static model
        prev_inds = torch.all(~init_actv.isnan(), dim=1).nonzero().flatten()
        prev_hidden = self.init_model.z.clone().detach()
        for k in range(self.actions.shape[1]):
            p = self.observations[:, k+1].to(self.device)
            mask = torch.all(~p.isnan(), dim=1)
            mask = mask[prev_inds]
            prev_hidden = prev_hidden[mask]
            prev_inds = torch.all(~p.isnan(), dim=1).nonzero().flatten()
            p = p[~p.isnan()].reshape(-1, self.n_obs)
            v = self.actions[:, k].to(self.device)
            v = v[~v.isnan()].reshape(-1, self.n_actions)

            self.tpc_optimizer.zero_grad()
            self.tpc_model.inference(self.inf_iters, self.inf_lr, v, prev_hidden, p)
            energy, obs_loss = self.tpc_model.get_energy()
            energy.backward()
            self.tpc_optimizer.step()

            # update the hidden state
            prev_hidden = self.tpc_model.z.clone().detach()

            # add up the loss value at each time step
            total_loss += obs_loss.item()
            total_energy += energy.item()

        return total_energy / (self.options['sequence_length'] + 1), total_loss / (self.options['sequence_length'] + 1)
