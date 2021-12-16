#utils function for running MCTS
import math
import numpy as np

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
def evaluation(state, dead_snakes):
    #a parameter. The higher it is, the more sensitive the evaluation is to length difference
    score_factor = 0.25
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

        