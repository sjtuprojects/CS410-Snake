import numpy as np

def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map

#return a grid map
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
    if dir == 0: return ((y-1)%height, x)
    if dir == 1: return ((y+1)%height, x)
    if dir == 2: return (y, (x-1)%width)
    if dir == 3: return (y, (x+1)%width)

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

#return if a and b is neighbor or same
def is_neighbor(a, b, height, width):
    ay, ax = a
    by, bx = b

    dx = abs(ax-bx)
    if dx == width - 1: dx = 1
    dy = abs(ay-by)
    if dy == height -1: dy = 1

    return dx + dy <= 1

def my_controller(observation, action_space, is_act_continuous=False):   
    obs = observation.copy()
    #agent id
    index = obs["controlled_snake_index"]
    #agent team
    team = [2,3,4]
    enemy = [5,6,7]
    if not index in team:
        team = [5,6,7]
        enemy = [2,3,4]

    head_pos = obs[index][0]  
    self_length = len(obs[index])
    state = get_state(obs)
    board_width = obs['board_width']
    board_height = obs['board_height']
    beans_positions = obs[1]

    action_index = 0
    action = [[1,0,0,0]]

    #surrounding = get_surrounding(state,board_width,board_height, head_pos[0], head_pos[1])

    min_dist = 99999
    for i in range(4):
        y, x = get_next_pos(head_pos,i,board_width,board_height)

        if state[y][x] > 1: continue
        
        skip_flag = False

        for j in enemy:
            enemy_head = obs[j][0]
            enemy_len = len(obs[j])
            if self_length > enemy_len and is_neighbor((y,x),enemy_head, board_height, board_width):
                skip_flag = True
                break
        for j in team:
            if j == index: continue
            friend_head = obs[j][0]
            friend_len = len(obs[j])
            if is_neighbor((y,x),friend_head, board_height, board_width):
                if self_length > friend_len:
                    skip_flag = True
                    break
                if self_length == friend_len and j < index:
                    skip_flag = True
                    break

        if skip_flag: continue

        if state[y][x] == 1:
            action_index = i
            break;
        d = dist_closest_bean(beans_positions,(y,x),board_height,board_width)
        if state[y][x] == 0 and d < min_dist:
            action_index = i
            min_dist = d

    if action_index == 0: action = [[1,0,0,0]]
    if action_index == 1: action = [[0,1,0,0]]
    if action_index == 2: action = [[0,0,1,0]]
    if action_index == 3: action = [[0,0,0,1]]

    # print(state)
    # print("--------")
    # print(c_index, action)
    # print(dist_closest_bean(beans_positions,head_pos,board_height,board_width))
    # print("=====================")

    return action