from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(base_dir))

import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import HandyRL base environment class
from ..environment import BaseEnvironment

# Import snakes3v3 game environment
from env.chooseenv import make

# Neural network for snake agent (1/2)
class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h

# Neural network for snake agent (2/2)
class SnakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32

        self.conv0 = TorusConv2d(25, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = self.head_p(h_head)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))

        return {'policy': p, 'value': v}

class Environment(BaseEnvironment):
    ACTION = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    DIRECTION = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    def __init__(self, args={}):
        super().__init__()
        self.env = make("snakes_3v3", conf=None)
        self.BOARD_WIDTH = self.env.board_width
        self.BOARD_HEIGHT = self.env.board_height
        self.NUM_AGENTS = self.env.n_player
        self.reset()

    def reset(self, args={}):
        state = self.env.reset()
        self.update((state[0], {}, None, False), True)

    def update(self, info, reset):
        state, last_actions, last_reward, last_terminal = info
        if reset:
            self.state_list = []
            self.steps = 0
        self.state_list.append(state)
        self.last_actions = last_actions
        self.last_reward = last_reward
        self.last_terminal = last_terminal
        self.steps += 1

    # Handle the actions of all players
    def step(self, actions):
        action_list = [actions.get(p, None) or 0 for p in self.players()]
        next_state, reward, is_terminal, _, info = self.env.step(self.env.encode(action_list))
        # State transition
        self.update((next_state[0], action_list, reward, is_terminal), False)

    # List of player IDs that can act in the turn (currently all players, 1 player/team, 6 teams)
    def turns(self):
        return [p for p in self.players()]

    # Check if the game is finished
    def terminal(self):
        if self.last_terminal or self.steps >= 200:
            return True
        return False

    # Outcome of a match
    def outcome(self):
        state_copy = self.state_list[-1].copy()

        lengths = [len(state_copy[i+2]) for i in self.players()]
        outcomes = {p: 0 for p in self.players()}

        longest_index = max(range(len(lengths)), key=lengths.__getitem__)

        # Match outcome score is the final length of snakes
        for i in self.players():
            outcomes[i] = 1 if i == longest_index else -1
        
        return outcomes

    # List of legal action indices
    def legal_actions(self, player):
        return list(range(len(self.ACTION)))

    # Length of all actions (determines output size of policy function)
    def action_length(self):
        return len(self.ACTION)

    # List of snakes on the board
    def players(self):
        return list(range(self.NUM_AGENTS))

    # PyTorch neural network model
    def net(self):
        return SnakeNet

    # Input for neural network
    # Note: observation is on a per-snake basis
    # Self position        : 0:head_x; 1:head_y
    # Head surroundings    : 2:head_up; 3:head_down; 4:head_left; 5:head_right
    # Beans positions      : (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
    # Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)
    def observation(self, player=None):
        if player is None:
            agent_index = 0
        else:
            agent_index = player

        b = np.zeros((self.NUM_AGENTS * 4 + 1, self.BOARD_WIDTH * self.BOARD_HEIGHT), dtype=np.float32)

        state_copy = self.state_list[-1].copy()
        snakes_positions = [state_copy[i+2] for i in self.players()]

        for p, snake in enumerate(snakes_positions):
            # Head position
            for pos in snake[:1]:
                b[0 + (p - player) % self.NUM_AGENTS, pos[0] * self.BOARD_WIDTH + pos[1]] = 1
            # Tip position
            for pos in snake[-1:]:
                b[4 + (p - player) % self.NUM_AGENTS, pos[0] * self.BOARD_WIDTH + pos[1]] = 1
            # Whole position
            for pos in snake:
                b[8 + (p - player) % self.NUM_AGENTS, pos[0] * self.BOARD_WIDTH + pos[1]] = 1

        # Previous head position
        if len(self.state_list) > 1:
            state_prev_copy = self.state_list[-2].copy()
            snakes_prev = [state_prev_copy[i+2] for i in self.players()]
            for p, snake in enumerate(snakes_prev):
                for pos in snake[:1]:
                    b[12 + (p - player) % self.NUM_AGENTS, pos[0] * self.BOARD_WIDTH + pos[1]] = 1

        # Food
        food_positions = state_copy[1]
        for pos in food_positions:
            b[16, pos[0] * self.BOARD_WIDTH + pos[1]] = 1

        return b.reshape(-1, self.BOARD_HEIGHT, self.BOARD_WIDTH)

    # Utility for observation (1/2)
    def get_surrounding(self, state, width, height, x, y):
        surrounding = [state[(y - 1) % height][x], # up
                    state[(y + 1) % height][x],    # down
                    state[y][(x - 1) % width],     # left
                    state[y][(x + 1) % width]]     # right

        return surrounding

    # Utility for observation (2/2)
    def make_grid_map(self, board_width, board_height, beans_positions:list, snakes_positions:dict):
        snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
        for index, pos in snakes_positions.items():
            for p in pos:
                snakes_map[p[0]][p[1]][0] = index

        for bean in beans_positions:
            snakes_map[bean[0]][bean[1]][0] = 1

        return snakes_map

if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            actions = {p: e.legal_actions(p) for p in e.turns()}
            e.step({p: random.choice(alist) for p, alist in actions.items()})
        print(e.outcome())