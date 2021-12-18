import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.conv0 = TorusConv2d(49, filters, (3, 3), True)
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

    def get_snake_directions(self, snake):
        directions = {p: [] for p in [0, 1, 2, 3]}

        prev_pos = snake[:1][0]

        for pos in snake:
            if pos[0] - prev_pos[0] == -1: # FACE UP
                directions[0].append(pos)
            elif pos[0] - prev_pos[0] == 1: # FACE DOWN
                directions[1].append(pos)
            elif pos[1] - prev_pos[1] == -1: # FACE LEFT
                directions[2].append(pos)
            elif pos[1] - prev_pos[1] == 1: # FACE RIGHT
                directions[3].append(pos)
            prev_pos = pos

        return directions

    def make_input(state, player):
        NUM_AGENTS = 6
        BOARD_WIDTH = state["board_width"]
        BOARD_HEIGHT = state["board_height"]

        b = np.zeros((NUM_AGENTS * 4 + 1, BOARD_WIDTH * BOARD_HEIGHT), dtype=np.float32)

        state_copy = state.copy()
        snakes_positions = [state_copy[i+2] for i in range(NUM_AGENTS)]

        for p, snake in enumerate(snakes_positions):
            # Head position
            for pos in snake[:1]:
                b[0 + (p - player) % NUM_AGENTS, pos[0] * BOARD_WIDTH + pos[1]] = 1
            # Tip position
            for pos in snake[-1:]:
                b[6 + (p - player) % NUM_AGENTS, pos[0] * BOARD_WIDTH + pos[1]] = 1
            # Whole position
            for pos in snake:
                b[12 + (p - player) % NUM_AGENTS, pos[0] * BOARD_WIDTH + pos[1]] = 1

            directions = self.get_snake_directions(snake)

            # Direction Up
            for pos in directions[0]:
                b[18 + (p - player) % self.NUM_AGENTS, pos[0] * self.BOARD_WIDTH + pos[1]] = 1
            # Direction Down
            for pos in directions[1]:
                b[24 + (p - player) % self.NUM_AGENTS, pos[0] * self.BOARD_WIDTH + pos[1]] = 1
            # Direction Left
            for pos in directions[2]:
                b[30 + (p - player) % self.NUM_AGENTS, pos[0] * self.BOARD_WIDTH + pos[1]] = 1
            # Direction Right
            for pos in directions[3]:
                b[36 + (p - player) % self.NUM_AGENTS, pos[0] * self.BOARD_WIDTH + pos[1]] = 1

        # Previous head position
        if len(self.state_list) > 1:
            state_prev_copy = self.state_list[-2].copy()
            snakes_prev = [state_prev_copy[i+2] for i in self.players()]
            for p, snake in enumerate(snakes_prev):
                for pos in snake[:1]:
                    b[42 + (p - player) % self.NUM_AGENTS, pos[0] * self.BOARD_WIDTH + pos[1]] = 1

        # Food
        food_positions = state_copy[1]
        for pos in food_positions:
            b[48, pos[0] * BOARD_WIDTH + pos[1]] = 1

        return b.reshape(-1, BOARD_HEIGHT, BOARD_WIDTH)

def my_controller(observation, action_space, is_act_continuous=False):   
    obs = observation.copy()

    index = obs["controlled_snake_index"]

    model = SnakeNet()
    model_path = "./models/latest.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    x = make_input(obs, index-2)

    with torch.no_grad():
        xt = torch.from_numpy(x).unsqueeze(0)
        output = model(xt)
   
    policy = output["policy"].squeeze(0).detach().numpy()
    value = output["value"].item()

    #print(policy, value)

    action_index = argmax(policy)

    if action_index == 0: action = [[1,0,0,0]]
    if action_index == 1: action = [[0,1,0,0]]
    if action_index == 2: action = [[0,0,1,0]]
    if action_index == 3: action = [[0,0,0,1]]

    return action