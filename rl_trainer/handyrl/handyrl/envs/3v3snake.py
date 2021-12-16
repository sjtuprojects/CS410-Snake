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

# Import opponents
from agent.greedy import submission as greedyAgent
from agent.MCTS import submission as mctsAgent
from agent.rl import submission as rlAgent

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
        self.update((state[0], {}, [0,0,0,0,0,0], False, None), True)

    def update(self, info, reset):
        state, last_actions, last_reward, last_terminal, last_info = info
        if reset:
            self.state_list = []
            self.steps = 0
            self.episode_reward = np.zeros(6)
        self.state_list.append(state)
        self.last_actions = last_actions
        self.last_reward = last_reward
        self.last_terminal = last_terminal
        self.last_info = last_info
        self.episode_reward += np.array(last_reward)
        self.steps += 1

    # Handle the actions of all players
    def step(self, actions):
        action_list = [actions.get(p, None) or 0 for p in self.players()]
        next_state, reward, is_terminal, _, info = self.env.step(self.env.encode(action_list))

        #grid_map = self.make_grid_map(next_state[0])
        #print('\n'.join([''.join([str(cell) for cell in row]) for row in grid_map]), '\n')

        # State transition
        self.update((next_state[0], action_list, reward, is_terminal, info), False)

    # [Utility] For printing map
    def make_grid_map(self, state):
        snakes_map = [[0]*self.BOARD_WIDTH for i in range(self.BOARD_HEIGHT)]

        snakes_positions = [state[i+2] for i in self.players()]
        food_positions = state[1]

        for index, pos in enumerate(snakes_positions):
            for p in pos:
                snakes_map[p[0]][p[1]] = index

        for food in food_positions:
            snakes_map[food[0]][food[1]] = 8

        return snakes_map

    # List of player IDs that can act in the turn (currently all players, 1 player/team, 6 teams)
    def turns(self):
        return [p for p in self.players()]

    # Check if the game is finished
    def terminal(self):
        if self.last_terminal or self.steps >= 200:
            return True
        return False

    # Reward of a step
    '''
    def reward(self):
        snakes_position = np.array(self.last_info['snakes_position'], dtype=object)
        beans_position = np.array(self.last_info['beans_position'], dtype=object)
        snake_heads = [snake[0] for snake in snakes_position]
        step_reward = {p: 0 for p in self.players()}

        if np.sum(self.episode_reward[:3]) > np.sum(self.episode_reward[3:]):
            for i in range(0, 3):
                step_reward[i] += 50
                step_reward[self.NUM_AGENTS - 1 - i] -= 25
        elif np.sum(self.episode_reward[:3]) < np.sum(self.episode_reward[3:]):
            for i in range(0, 3):
                step_reward[i] -= 25
                step_reward[self.NUM_AGENTS - 1 - i] += 50

        for i in self.players():
            if self.last_reward[i] > 0:
                step_reward[i] += 20
            else:
                self_head = np.array(snake_heads[i])
                dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
                step_reward[i] -= min(dists)
                if self.last_reward[i] < 0:
                    step_reward[i] -= 10

        return step_reward
    '''

    # Outcome of a match
    def outcome(self):
        state_copy = self.state_list[-1].copy()

        lengths = [len(state_copy[i+2]) for i in self.players()]
        outcomes = {p: 0 for p in self.players()}

        team_1_score = lengths[0] + lengths[1] + lengths[2]
        team_2_score = lengths[3] + lengths[4] + lengths[5]

        if team_1_score > team_2_score:
            for i in range(0, 3):
                outcomes[i] = 1
                outcomes[self.NUM_AGENTS - 1 - i] = -1
        elif team_1_score < team_2_score:
            for i in range(0, 3):
                outcomes[i] = -1
                outcomes[self.NUM_AGENTS - 1 - i] = 1
        
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

    # Opponent for evaluation (1)
    def agent_greedy(self, player):
        action = greedyAgent.my_controller(self.state_list[-1].copy(), None)
        return action[0].index(max(action[0]))

    # Opponent for evaluation (2)
    def agent_mcts(self, player):
        action = mctsAgent.my_controller(self.state_list[-1].copy(), None)
        return action[0].index(max(action[0]))

    # Opponent for evaluation (3)
    def agent_rl(self, player):
        action = rlAgent.my_controller(self.state_list[-1].copy(), None, True)
        return action[0].index(max(action[0]))

    # PyTorch neural network model
    def net(self):
        return SnakeNet

    def get_snake_directions(self, snake):
        directions = {p: [] for p in [0, 1, 2, 3]}

        prev_pos = snake[:1]

        for pos in snake:
            if pos[0] - prev_pos[0] == -1: # FACE UP
                directions[0].append(pos)
            elif pos[0] - prev_pos[0] == 1: # FACE DOWN
                directions[1].append(pos)
            elif pos[1] - prev_pos[1] == -1: # FACE LEFT
                directions[2].append(pos)
            elif pos[1] - prev_pos[1] == 1: # FACE RIGHT
                directions[3].append(pos)

        return directions

    # Input for neural network
    # Note: observation is on a per-snake basis
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
                b[0 + (p - player) % self.NUM_AGENTS, pos[1] * self.BOARD_WIDTH + pos[0]] = 1
            # Tip position
            for pos in snake[-1:]:
                b[6 + (p - player) % self.NUM_AGENTS, pos[1] * self.BOARD_WIDTH + pos[0]] = 1
            # Whole position
            for pos in snake:
                b[12 + (p - player) % self.NUM_AGENTS, pos[1] * self.BOARD_WIDTH + pos[0]] = 1

            directions = self.get_snake_directions(snake)

            # Direction Up
            # for pos in directions[0]:
            #     b[18 + (p - player) % self.NUM_AGENTS, pos[1] * self.BOARD_WIDTH + pos[0]] = 1
            # # Direction Down
            # for pos in directions[1]:
            #     b[24 + (p - player) % self.NUM_AGENTS, pos[1] * self.BOARD_WIDTH + pos[0]] = 1
            # # Direction Left
            # for pos in directions[2]:
            #     b[30 + (p - player) % self.NUM_AGENTS, pos[1] * self.BOARD_WIDTH + pos[0]] = 1
            # # Direction Right
            # for pos in directions[3]:
            #     b[36 + (p - player) % self.NUM_AGENTS, pos[1] * self.BOARD_WIDTH + pos[0]] = 1

        # Previous head position
        if len(self.state_list) > 1:
            state_prev_copy = self.state_list[-2].copy()
            snakes_prev = [state_prev_copy[i+2] for i in self.players()]
            for p, snake in enumerate(snakes_prev):
                for pos in snake[:1]:
                    b[42 + (p - player) % self.NUM_AGENTS, pos[1] * self.BOARD_WIDTH + pos[0]] = 1

        # Food
        food_positions = state_copy[1]
        for pos in food_positions:
            b[48, pos[1] * self.BOARD_WIDTH + pos[0]] = 1

        # Steps
        #b[49] = np.full(self.BOARD_WIDTH * self.BOARD_HEIGHT, np.tanh((200 - self.steps) / 16))
        #b[50] = np.full(self.BOARD_WIDTH * self.BOARD_HEIGHT, np.tanh((200 - self.steps) / 128))

        return b.reshape(-1, self.BOARD_HEIGHT, self.BOARD_WIDTH)

if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            actions = {p: e.legal_actions(p) for p in e.turns()}
            e.step({p: random.choice(alist) for p, alist in actions.items()})
        print(e.outcome())