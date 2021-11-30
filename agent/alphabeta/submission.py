import numpy as np
import itertools
from time import time

#NOT USED
def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map

#return a grid map (NOT USED)
def get_state(state):
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
    #print(state_)
    #print("===============================")

    return state_

#return coordinate after going 0:up 1:down 2:left 3:right
def get_next_pos(pos, dir, width, height):
    y, x = pos
    if dir == 0: return [(y-1)%height, x]
    if dir == 1: return [(y+1)%height, x]
    if dir == 2: return [y, (x-1)%width]
    if dir == 3: return [y, (x+1)%width]

#return distance to closest bean
def dist_closest_bean(beans_positions, head_position, height, width):
    dist = 99999
    hy, hx = head_position
    for by, bx in beans_positions:
        dx = min(abs(hx-bx), min(hx,bx)+width-max(hx,bx))
        dy = min(abs(hy-by), min(hy,by)+height-max(hy,by))
        d = dx+dy
        #print(dx,dy)
        if d < dist: dist = d
    return dist

#get successor of the current state-action
def get_successor(state, indexes, actions):
    successor = state.copy()
    width = successor['board_width']
    height = successor['board_height']
    for index, action in zip(indexes, actions):
        new_snake = []
        new_head = get_next_pos(successor[index][0],action, width, height)
        new_snake.append(new_head)
        for i in successor[index][:-1]:
            new_snake.append(i)
        if new_head in successor[1]:
            beans = successor[1].copy()
            beans.remove(new_head)
            successor[1] = beans
            new_snake.append(successor[index][-1])
        successor[index] = new_snake
    
    return successor

#return a list of dead snakes given the state
def is_snake_dead(state):
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

#score = total team length - total enemy length + sum(0.99/distant from head to closest bean)
def evaluation(state, team, enemy, dead_snakes):
    height = state['board_height']
    width = state['board_width']
    beans_positions = state[1]

    score = 0
    for i in team:
        if not i in dead_snakes:
            score += len(state[i])
            dist = dist_closest_bean(beans_positions, state[i][0], height, width)
            if dist <= 0: dist = 1
            score += 0.99/dist
        #else:
            #score += 3
    for i in enemy:
        if not i in dead_snakes:
            score -= len(state[i])
        #else:
            #score -= 3

    #print(score)
    return score

#return a list of legal actions, namely if the space is empty
def get_legal_actionss(state, team, enemy):
    legal_sets = []
    for i in team:
        legal_set = []
        head = state[i][0]
        #neck = state[i][1]
        for j in range(4):
            next_pos = get_next_pos(head,j,state['board_width'],state['board_height'])

            skip_flag = False
            #check for collisions
            for k in team:
                if next_pos in state[k][:-1]:
                    skip_flag = True
                    break
            if skip_flag: continue
            for k in enemy:
                if next_pos in state[k][:-1]:
                    skip_flag = True
                    break
            if skip_flag: continue
            legal_set.append(j) 
        if len(legal_set) == 0: legal_set = [0]
        legal_sets.append(legal_set)

    return list(itertools.product(*legal_sets))

def alphaBeta(obs, team, enemy):
    def max_value(state, depth, alpha, beta):
        ts = time()
        #check if at leaf
        dead_snakes = is_snake_dead(state)
        if dead_snakes or depth == 0:
            return (0, evaluation(state,team,enemy,dead_snakes))

        best_value = -99999
        best_action = [0,0,0]

        legal_actions = get_legal_actionss(state, team, enemy)

        #print(len(legal_actions))

        for action in legal_actions:
            next_state = get_successor(state,team,action)
            next_value = min_value(next_state,depth,alpha,beta)

            if next_value >= beta:
                #print("pruned", action)
                return (action, next_value)
            if next_value > best_value:
                best_value = next_value
                best_action = action
            if next_value > alpha:
                alpha = next_value
        
        return (best_action, best_value)

    def min_value(state, depth, alpha, beta):
        #only check for leaves at max_value

        best_value = 999999

        legal_sets = []

        legal_actions = get_legal_actionss(state, enemy, team)

        #print(legal_actions)

        for action in legal_actions:
            next_state = get_successor(state,enemy,action)
            next_value = max_value(next_state,depth-1,alpha,beta)[1]

            if next_value <= alpha:
                #print("pruned", next_value, alpha)
                return next_value
            if next_value < best_value:
                best_value = next_value
            if next_value < beta:
                beta = next_value

        #return min value, no need to extract action
        return best_value

    m = max_value(obs, 2, -99999, 99999)
    print(m)
    return m[0]
            

def my_controller(observation, action_space, is_act_continuous=False):   
    obs = observation.copy()

    #agent id and group it belongs in
    index = obs["controlled_snake_index"]
    team = [2,3,4]
    enemy = [5,6,7]
    if index in enemy:
        team = [5,6,7]
        enemy = [2,3,4]

    pos_in_team = team.index(index)
    action_index = alphaBeta(obs,team,enemy)[pos_in_team]

    if action_index == 0: action = [[1,0,0,0]]
    if action_index == 1: action = [[0,1,0,0]]
    if action_index == 2: action = [[0,0,1,0]]
    if action_index == 3: action = [[0,0,0,1]]

    #print(get_state(obs))
    #print("--------")
    # print(c_index, action)
    # print(dist_closest_bean(beans_positions,head_pos,board_height,board_width))
    # print("=====================")

    return action