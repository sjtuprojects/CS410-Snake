import numpy as np
from operator import itemgetter


# flood fill will start with distance 0 at seeds and only flow where state_map[x][y] == 0
def flood_fill(state, seed):
    snakes_positions = {key: state[key] for key in state.keys() & {2, 3, 4, 5, 6, 7}}
    state_map = make_grid_map(20, 10, snakes_positions)

    # stop floodfill if legal move has a snake cell on it
    if state_map[seed[0]][seed[1]] > 0:
        print("Snake cell!")
        return []

    size_x = 10
    size_y = 20

    field_dist = np.full(fill_value=-1, shape=(size_x, size_y))

    frontier = [seed]

    frontiers = [frontier]

    field_dist[seed] = 0

    dist = 1

    while frontier:
        new_frontier = []
        for x, y in frontier:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = (x + dx) % size_x
                new_y = (y + dy) % size_y
                if state_map[new_x][new_y] == 0 and field_dist[new_x, new_y] == -1:
                    field_dist[new_x, new_y] = dist
                    new_frontier.append((new_x, new_y))
        frontier = new_frontier
        frontiers.append(frontier)
        dist += 1

    return frontiers


# get the distance to the closest frontier cell that matches test_func requirement
def get_dist(frontiers, test_func):
    for dist, frontier in enumerate(frontiers):
        for pos in frontier:
            if test_func(pos):
                return dist

    return None


# create game grid map for flood fill
def make_grid_map(board_width, board_height, snakes_positions:dict):
    snakes_map = [[0 for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]] = index

    return snakes_map


def my_controller(observation, action_space, is_act_continuous=False, min_length=10):   
    obs = observation.copy()
    index = obs["controlled_snake_index"]
    team = [2,3,4]
    enemy = [5,6,7]
    if not index in team:
        team = [5,6,7]
        enemy = [2,3,4]

    self_length = len(obs[index])
    beans_positions = obs[1]

    if self_length < min_length:
        result = goto(obs, lambda pos:[pos[0], pos[1]] in beans_positions) # chase bean
    else:
        result = goto(obs, lambda pos:[pos[0], pos[1]] in obs[index][-1:])      # chase tail

    action = [[1,0,0,0]]
    if result == (-1, 0): action = [[1,0,0,0]]
    elif result == (1, 0): action = [[0,1,0,0]]
    elif result == (0, -1): action = [[0,0,1,0]]
    elif result == (0, 1): action = [[0,0,0,1]]

    return action


# get all possible legal action tuples
def legal_actions(state):
    snake_position = state[state["controlled_snake_index"]]
    snake_head = snake_position[0]
    snake_neck = snake_position[1]

    size_x = 10
    size_y = 20

    poses = []

    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_x = (snake_head[0] + dx) % size_x
        new_y = (snake_head[1] + dy) % size_y
        if new_x != snake_neck[0] or new_y != snake_neck[1]:
            poses.append((dx, dy))

    return poses
            

def goto(state, test_func):
    snake_position = state[state["controlled_snake_index"]]
    snake_head = snake_position[0]

    size_x = 10
    size_y = 20

    result = None
    
    action_dists = {}

    for action in legal_actions(state):
        seed = ((snake_head[0] + action[0]) % size_x, (snake_head[1] + action[1]) % size_y)
        frontiers = flood_fill(state, seed)
        dist = get_dist(frontiers, test_func)
        if dist is not None:
            action_dists[action] = dist

    if action_dists:
        closest_action, _ = min(action_dists.items(), key=itemgetter(1))

        result = closest_action
            
    return result