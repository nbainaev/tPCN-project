#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from copy import copy

import pygame
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io


import numpy as np
from copy import copy
import gymnasium as gym
from gymnasium import spaces
import pygame
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io


class GridWorld(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(
            self,
            room,
            default_reward=0,
            observation_radius=0,
            collision_hint=False,
            collision_reward=0,
            headless=True,
            random_floor_colors=False,
            n_random_colors=0,
            markov_radius=0,
            seed=None,
            render_mode=None
    ):
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._is_closed = False
        room = np.asarray(room)
        self.colors, self.rewards, self.terminals, self.landmarks = (
            room[0, :, :], room[1, :, :], room[2, :, :], room[3, :, :]
        )

        self.random_floor_colors = random_floor_colors
        if self.random_floor_colors:
            if markov_radius == 0:
                colors = self._rng.integers(0, n_random_colors, size=self.colors.shape)
            else:
                colors = generate_map(markov_radius, self.colors.shape, seed)

            floor_mask = self.colors >= 0
            self.colors[floor_mask] = colors[floor_mask]
            self.colors[self.terminals == 1] = np.max(colors) + 1

        self.colors[self.landmarks == 1] = (
                np.max(self.colors) + np.arange(np.count_nonzero(self.landmarks)) + 1
        )

        self.h, self.w = self.colors.shape

        self.return_state = observation_radius < 0
        self.observation_radius = observation_radius
        self.collision_hint = collision_hint
        self.collision_reward = collision_reward
        self.render_mode = render_mode
        self.headless = headless if render_mode is None else False

        self.shift = max(self.observation_radius, 1)

        self.colors = np.pad(
            self.colors,
            self.shift,
            mode='constant',
            constant_values=-1
        ).astype(np.int32)

        self.unique_colors = np.unique(self.colors)

        if (not self.collision_hint) and (self.observation_radius <= 0):
            self.unique_colors = self.unique_colors[self.unique_colors >= 0]

        self.n_colors = len(self.unique_colors)

        if not self.return_state:
            self.observation_shape = (2*self.observation_radius + 1, 2*self.observation_radius + 1)
            self.observation_space = spaces.Box(
                low=-1,
                high=self.n_colors-1,
                shape=self.observation_shape,
                dtype=np.int32
            )
        else:
            self.observation_shape = (2,)
            self.observation_space = spaces.Box(
                low=0,
                high=max(self.h, self.w),
                shape=self.observation_shape,
                dtype=np.int32
            )

        self.action_space = spaces.Discrete(4)  # 0: left, 1: right, 2: up, 3: down

        self.start_r = None
        self.start_c = None
        self.r = None
        self.c = None
        self.action = None
        self.action_success = None
        self.temp_obs = None
        self.actions = {0, 1, 2, 3}
        self.default_reward = default_reward

        if self.render_mode == 'human':
            pygame.init()
            self.canvas = None
        else:
            self.canvas = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        start_r = options.get('start_r', None) if options else None
        start_c = options.get('start_c', None) if options else None
        
        if (start_r is None) or (start_c is None):
            while True:
                start_r = self._rng.integers(self.h)
                start_c = self._rng.integers(self.w)
                if self.colors[start_r + self.shift, start_c + self.shift] >= 0:
                    if self.terminals[start_r, start_c] == 0:
                        break
        else:
            assert self.colors[start_r + self.shift, start_c + self.shift] >= 0

        self.start_r, self.start_c = start_r, start_c
        self.r, self.c = start_r, start_c

        self.temp_obs = None
        self.action = None
        self.action_success = True

        observation = self._get_obs(self.r, self.c) if not self.return_state else np.array([self.r, self.c])
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action):
        assert action in self.actions
        self.action = action

        prev_r = self.r
        prev_c = self.c

        if action == 0:
            self.c -= 1
        elif action == 1:
            self.c += 1
        elif action == 2:
            self.r -= 1
        elif action == 3:
            self.r += 1

        # Check whether action is taking to inaccessible states.
        temp_x = self.colors[self.r+self.shift, self.c+self.shift]
        if temp_x < 0:
            self.r = prev_r
            self.c = prev_c

            if (not self.return_state) and self.collision_hint:
                self.temp_obs = np.full(self.observation_shape, fill_value=temp_x)

            self.action_success = False
        else:
            self.action_success = True

        observation = self._get_obs(self.r, self.c) if not self.return_state else np.array([self.r, self.c])
        reward = self.rewards[self.r, self.c] + self.default_reward
        if not self.action_success:
            reward += self.collision_reward

        terminated = bool(self.terminals[self.r, self.c])
        truncated = False
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        shift = self.shift
        im = self.colors.copy()
        min_vis_color = np.min(self.colors)

        if shift > 0:
            im = im[shift:-shift, shift:-shift]

        plt.figure()
        sns.heatmap(im, annot=True, cmap='Pastel1', square=True, vmin=min_vis_color, cbar=False)
        if (self.r is not None) and (self.c is not None):
            plt.text(self.c, self.r+1, 'A', size='x-large')

        for s in np.flatnonzero(self.terminals):
            plt.text(s % self.w, s // self.w + 1, 'G', size='x-large')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches="tight")
        plt.close()
        buf.seek(0)
        im = Image.open(buf)

        if self.render_mode == 'human':
            if self.canvas is None:
                pygame.init()
                self.canvas = pygame.display.set_mode(im.size)
                pygame.display.set_caption('Gridworld')
            
            pygame_surface = pygame.image.fromstring(im.tobytes(), im.size, im.mode)
            self.canvas.blit(pygame_surface, (0, 0))
            pygame.display.update()
            return None
        elif self.render_mode == 'rgb_array':
            return np.array(im)

    def close(self):
        if self.canvas is not None:
            pygame.quit()
            self.canvas = None
        self._is_closed = True

    def _get_obs(self, r, c):
        if self.return_state:
            return np.array([r, c])
        
        if self.temp_obs is not None:
            obs = copy(self.temp_obs)
            self.temp_obs = None
            return obs
        
        r += self.shift
        c += self.shift
        start_r, start_c = r - self.observation_radius, c - self.observation_radius
        end_r, end_c = r + self.observation_radius + 1, c + self.observation_radius + 1
        obs = self.colors[start_r:end_r, start_c:end_c]
        return obs

    def _get_info(self):
        return {
            'position': (self.r, self.c),
            'action_success': self.action_success,
            'start_position': (self.start_r, self.start_c)
        }

    def get_true_map(self):
        true_map = self.colors.copy()
        if self.shift > 0:
            true_map = true_map[self.shift:-self.shift, self.shift:-self.shift]

        for i, color in enumerate(self.unique_colors):
            true_map[true_map == color] = i

        return true_map
    

def generate_map(markov_radius: int, size: tuple[int, int], seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = np.full(size, fill_value=-1, dtype=np.int32)

    for _ in range(2):
        for r in range(markov_radius, size[0] - markov_radius):
            for c in range(markov_radius, size[1] - markov_radius):
                start_r, start_c = max(0, r - markov_radius), max(0, c - markov_radius)
                end_r, end_c = r + markov_radius + 1, c + markov_radius + 1

                window = colors[start_r:end_r, start_c:end_c]
                shape = window.shape
                window = window.flatten()

                # remove duplicates
                unique, positions = np.unique(window, return_index=True)
                window = np.full_like(window, fill_value=-1)
                window[positions] = unique

                # fill empty space
                empty_mask = window == -1
                n_nonzero = np.count_nonzero(empty_mask)
                if n_nonzero == 0:
                    continue

                candidates = np.arange(window.size)
                candidates = candidates[np.isin(candidates, window, invert=True)]
                candidates = candidates[:n_nonzero]
                rng.shuffle(candidates)
                window[empty_mask] = candidates
                colors[start_r:end_r, start_c:end_c] = window.reshape(shape)
    return colors


