import math
import time
from pathlib import Path
import sys
import os

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8
global count
count = 0

class MCTS():
    def __init__(self, nnet, timeLimit=1.9, cpuct=4.0, stepLimit = 10):
        self.nnet = nnet
        self.cpuct = cpuct
        self.timeLimit = timeLimit
        #print(timeLimit)
        self.stepLimit = stepLimit
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, state, temp=1):
        start_time = time.time()
        search_count = 0
        while time.time() - start_time < self.timeLimit:
            self.search(state)
            search_count += 1

        print(search_count)

        s = string_representation(state)
        i = state["controlled_snake_index"]
        counts = [
            self.Nsa[(s, i, a)] if (s, i, a) in self.Nsa else 0
            for a in range(get_action_size())
        ]
        #print(counts)

        counts_sum = float(sum(counts))
        if counts_sum == 0:
            return greedy(state,i)
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state):
        s = string_representation(state)

        if not "steps" in state:
            state["steps"] = 0

        if s not in self.Es:
            dead_snakes = are_snakes_dead(state)
            self.Es[s] = 0
            if dead_snakes or state["steps"] >= self.stepLimit or len(state[1]) < 5:
                # terminal node
                x = make_input(state,0)
                with torch.no_grad():
                    xt = torch.from_numpy(x).unsqueeze(0)
                    p, v = self.nnet(xt)
                self.Es[s] = [v, -v]
        if self.Es[s] != 0:
            return self.Es[s]
        
        if s not in self.Ns:
            values = [0,0]
            #sTime = time.time()
            for i in range(2,8):
                # leaf node
                #get p, v from nnet
                x = make_input(state,i-2)
                with torch.no_grad():
                    xt = torch.from_numpy(x).unsqueeze(0)
                    p, v = self.nnet(xt)
                    self.Ps[s, i] = np.squeeze(p)
                    #print(self.Ps[s, i])

                #values are average of 3 outputs
                if i in [2,3,4]: values[0] += (v/3.0)
                else: values[1] += (v/3.0)

                valids = get_legal_actions_single(state, i)
                self.Ps[s, i] = [a*b for a,b in zip(self.Ps[s, i],valids)] # masking invalid moves 
                #print(self.Ps[s, i])
                sum_Ps_s = np.sum(self.Ps[s, i])
                #print(valids, self.Ps[s, i])
                if sum_Ps_s > 0:
                    self.Ps[s, i] /= sum_Ps_s  # renormalize
                else:
                    #no valid moves
                    self.Ps[s, i] = [1 / len(self.Ps[s, i])]*get_action_size()

                self.Vs[s, i] = valids
                self.Ns[s] = 0

            #print("time:",time.time()-sTime)
            return values

        best_acts = [None] * 6
        for i in range(2,8):
            valids = self.Vs[s, i]
            cur_best = -float('inf')
            best_act = 0

            # pick the action with the highest upper confidence bound
            for a in range(get_action_size()):
                if valids[a]:
                    if (s, i, a) in self.Qsa:
                        u = self.Qsa[(s, i, a)] + self.cpuct * self.Ps[s, i][a] * math.sqrt(self.Ns[s]) / (
                                1 + self.Nsa[(s, i, a)])
                    else:
                        u = self.cpuct * self.Ps[s,i][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = a
            
            best_acts[i-2] = best_act

        next_s = get_successor(state, best_acts)

        values = self.search(next_s)

        for i in range(2,8):
            a = best_acts[i-2]
            if i in [2,3,4]: v = values[0]
            else: v = values[1]

            if (s, i, a) in self.Qsa:
                self.Qsa[(s, i, a)] = (self.Nsa[(s, i, a)] * self.Qsa[
                    (s, i, a)] + v) / (self.Nsa[(s, i, a)] + 1)
                self.Nsa[(s, i, a)] += 1

            else:
                self.Qsa[(s, i, a)] = v
                self.Nsa[(s, i, a)] = 1
                #print(s,i,a)

        self.Ns[s] += 1
        return values

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

        return (p,v)

def make_input(state, player):
    def get_snake_directions(snake):
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

    NUM_AGENTS = 6
    BOARD_WIDTH = state["board_width"]
    BOARD_HEIGHT = state["board_height"]

    b = np.zeros((NUM_AGENTS * 8 + 1, BOARD_WIDTH * BOARD_HEIGHT), dtype=np.float32)

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
        # Previous head position
        for pos in snake[1:2]:
            b[42 + (p - player) % NUM_AGENTS, pos[0] * BOARD_WIDTH + pos[1]] = 1

        directions = get_snake_directions(snake)

        # Direction Up
        for pos in directions[0]:
            b[18 + (p - player) % NUM_AGENTS, pos[0] * BOARD_WIDTH + pos[1]] = 1
        # Direction Down
        for pos in directions[1]:
            b[24 + (p - player) % NUM_AGENTS, pos[0] * BOARD_WIDTH + pos[1]] = 1
        # Direction Left
        for pos in directions[2]:
            b[30 + (p - player) % NUM_AGENTS, pos[0] * BOARD_WIDTH + pos[1]] = 1
        # Direction Right
        for pos in directions[3]:
            b[36 + (p - player) % NUM_AGENTS, pos[0] * BOARD_WIDTH + pos[1]] = 1
        #print(directions)

    # Food
    food_positions = state_copy[1]
    for pos in food_positions:
        b[48, pos[0] * BOARD_WIDTH + pos[1]] = 1

    return b.reshape(-1, BOARD_HEIGHT, BOARD_WIDTH)

model = SnakeNet()
model_path = os.path.dirname(os.path.abspath(__file__)) + '/latest.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

def my_controller(observation, action_space, is_act_continuous=False):   
    
    #st = time.time()     
    state = observation.copy()
    #global count
    #print(state["controlled_snake_index"],int(count/3))
    #count += 1

    #print((time.time() - st)*1000)

    tree = MCTS(model, 0.5)

    probs = tree.getActionProb(state)
    #print(probs)

    action_index = np.argmax(probs)

    if action_index == 0: action = [[1,0,0,0]]
    if action_index == 1: action = [[0,1,0,0]]
    if action_index == 2: action = [[0,0,1,0]]
    if action_index == 3: action = [[0,0,0,1]]

    return action

#return string representation, only using snake and beans location to represent state
def string_representation(state):
    s = ""
    for i in range(1,8):
        s += str(state[i])
    return s

#return [2,3,4] or [5,6,7]
def get_team_list(index):
    team = [2,3,4]
    if index in team: return team
    else: return [5,6,7]
def get_enemy_list(index):
    enemy = [2,3,4]
    if not index in enemy: return enemy
    else: return [5,6,7]

#return a list of dead snakes given the state
def are_snakes_dead(state):
    dead_snakes = []
    for i in range(2,8):
        head = state[i][0]
        for j in range(2,8):
            if not i == j and head in state[j]:
                dead_snakes.append(i)
                break
            elif head in state[j][1:]:
                dead_snakes.append(i)
                break
    return dead_snakes

#return evaluation of the state in range [-1,+1]
def evaluation(state, dead_snakes, score_factor=0.25):
    #a parameter. The higher it is, the more sensitive the evaluation is to length difference
    score = 0
    for i in range(2,5):
        if not i in dead_snakes:
            score += len(state[i])
        else:
            score += 3
    for i in range(5,8):
        if not i in dead_snakes:
            score -= len(state[i])
        else:
            score -= 3

    score = math.tanh(score_factor*score)
    return [score,-1*score]

#return a list of legal actions, namely if the space is empty
#this is for single agent
def get_legal_actions_single(state, index):
    legal_actions = [1,1,1,1]
    head = state[index][0]
    for j in range(4):
        next_pos = get_next_pos(head,j,state['board_width'],state['board_height'])

        #check for collisions
        for k in range(2,8):
            if next_pos in state[k][:-1]:
                legal_actions[j] = 0

    return legal_actions

def get_action_size(single=True):
    if single: return 4
    else: return 64
        
#return coordinate after going 0:up 1:down 2:left 3:right
def get_next_pos(pos, dir, width, height):
    y, x = pos
    if dir == 0: return [(y-1)%height, x]
    if dir == 1: return [(y+1)%height, x]
    if dir == 2: return [y, (x-1)%width]
    if dir == 3: return [y, (x+1)%width]

#get successor of the current state-action
def get_successor(state, actions):
    successor = state.copy()
    width = successor['board_width']
    height = successor['board_height']
    for i in range(6):
        new_snake = []
        new_head = get_next_pos(successor[i+2][0],actions[i], width, height)
        new_snake.append(new_head)
        for j in successor[i+2][:-1]:
            new_snake.append(j)
        if new_head in successor[1]:
            beans = successor[1].copy()
            beans.remove(new_head)
            successor[1] = beans
            new_snake.append(successor[i+2][-1])
        successor[i+2] = new_snake

    if "steps" in successor:
        successor["steps"] += 1
    else:
        successor["steps"] = 1
    
    return successor

#return a grid map
def get_grid_map(state):
    def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
        snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
        for index, pos in snakes_positions.items():
            for p in pos:
                snakes_map[p[0]][p[1]][0] = index

        for bean in beans_positions:
            snakes_map[bean[0]][bean[1]][0] = 1

        return snakes_map

    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state_ = np.squeeze(state_, axis=2)

    return state_

#greedy policy as a placeholder to nnet
def greedy(state, index):
    #return distance to closest bean
    def dist_closest_bean(beans_positions, head_position, width, height):
        dist = 99999
        hy, hx = head_position
        for by, bx in beans_positions:
            dx = min(abs(hx-bx), min(hx,bx)+width-max(hx,bx))
            dy = min(abs(hy-by), min(hy,by)+height-max(hy,by))
            d = dx+dy
            #print(dx,dy)
            if d < dist: dist = d
        return dist

    #return if a and b is neighbor or same
    def is_neighbor(a, b, width, height):
        ay, ax = a
        by, bx = b

        dx = abs(ax-bx)
        if dx == width - 1: dx = 1
        dy = abs(ay-by)
        if dy == height -1: dy = 1

        return dx + dy <= 1

    team = get_team_list(index)
    enemy = get_enemy_list(index)

    head_pos = state[index][0]  
    self_length = len(state[index])
    board_width = state["board_width"]
    board_height = state["board_height"]
    beans_positions = state[1]

    action = [0,0,0,0]
    action_index = 1
    min_dist = 99999

    for i in range(4):
        y, x = get_next_pos(head_pos,i,board_width,board_height)
        
        skip_flag = False

        if (y,x) in state[index][1:]:
            continue

        for j in enemy:
            enemy_head = state[j][0]
            enemy_len = len(state[j])
            if (y,x) in state[j]:
                skip_flag = True
                break
            if self_length > enemy_len and is_neighbor((y,x),enemy_head, board_width, board_height):
                skip_flag = True
                break
        for j in team:
            if j == index: continue
            friend_head = state[j][0]
            friend_len = len(state[j])
            if (y,x) in state[j]:
                skip_flag = True
                break
            if is_neighbor((y,x),friend_head, board_width, board_height):
                if self_length > friend_len:
                    skip_flag = True
                    break
                if self_length == friend_len and j < index:
                    skip_flag = True
                    break

        if skip_flag: continue

        if (y,x) in state[1]:
            action[i] = 1
            continue;
        d = dist_closest_bean(beans_positions,(y,x),board_width,board_height)
        action[i] = 1.0 / (1 + d)

    return action
