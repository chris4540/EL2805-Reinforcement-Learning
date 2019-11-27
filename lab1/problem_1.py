#%%
import numpy as np
import os
from enum import IntEnum

#%%
# actions
class Action(IntEnum): 
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAND = 4

# maze setting
class Maze:
    HEIGHT = 7
    WIDTH = 8
    ENTRANCE = (0,0)
    EXIT = (6,5)
    WALL = [(0,2),(1,2),(2,2),(3,2),\
        (1,5),(2,5),(3,5),(2,6),(2,7),\
        (5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(6,4)]
    SIZE = HEIGHT*WIDTH
    STATESIZE = HEIGHT*WIDTH-len(WALL)
    PerInit = (0,0)
    MinoInit = (6,5)
    EXITIDX = np.ravel_multi_index(EXIT, dims=(HEIGHT, WIDTH))
  
def init_maze(wall):
    maze = np.ones([Maze.HEIGHT,Maze.WIDTH])
    for i in wall:
        maze[i] = 0
    return np.pad(maze, 1)

# state index -> position
def idx_to_state(idx):
    ret = np.unravel_index(idx, shape=(Maze.HEIGHT, Maze.WIDTH))
    return ret

# position -> state index
def state_to_idx(state):
    ret = np.ravel_multi_index(state, dims=(Maze.HEIGHT, Maze.WIDTH))
    return ret

"""
ajacent_states:  a MazeSize*ActionSize 2d array for people/minotaur:
    ajacent_states[cur_state, action] = next_state (None for unavailable actions)
"""
def adjacent_states(maze, has_still):
    adj_states = []
    for i in range(Maze.HEIGHT):
        for j in range(Maze.WIDTH):
            if maze[i+1,j+1]==0:
                adj_states.append([])
                continue
            
            next_states = []
            
            if maze[i,j+1]==0:
                next_states.append(None)
            else:
                next_states.append(state_to_idx((i-1,j)))
            
            if maze[i+2,j+1]==0:
                next_states.append(None)
            else:
                next_states.append(state_to_idx((i+1,j)))

            if maze[i+1,j]==0:
                next_states.append(None)
            else:
                next_states.append(state_to_idx((i,j-1)))
            
            if maze[i+1,j+2]==0:
                next_states.append(None)
            else:
                next_states.append(state_to_idx((i,j+1)))

            if has_still:
                next_states.append(state_to_idx((i,j)))
            
            adj_states.append(next_states)

    assert(len(adj_states)==Maze.SIZE)

    return adj_states


# def get_reward(per_state, mino_state):
#     if (per_state == Maze.EXITIDX) and (mino_state != Maze.EXITIDX):
#         ret = 10
#     elif per_state == mino_state:
#         ret = 0
#     else:
#         ret = -1
#     return ret

# valfun: mino_state * per_state * T
def init_valfun(T):
    v_fun = -np.ones((Maze.SIZE,Maze.SIZE,T))
    v_fun[:,:,T-1] = 0
    v_fun[:, Maze.EXITIDX, T-1] = 1
    v_fun[Maze.EXITIDX, Maze.EXITIDX,T-1] = 0
    for wall in Maze.WALL:
        idx = state_to_idx(wall)
        v_fun[idx,:,T-1] = -1
    return v_fun

def value_iteration(T, init_per, init_mino, per_adj_states, mino_adj_states):
    v_fun = init_valfun(T)
    opt_actions = -np.ones((Maze.SIZE, Maze.SIZE, T),dtype=int)
    for t in np.arange(T-1,0,-1):
        for mino_state in range(Maze.SIZE):
            for per_state in range(Maze.SIZE):
                opt_act = -1
                if not per_adj_states[per_state]:
                    continue
                elif (per_state == Maze.EXITIDX) and (per_state != mino_state):
                    update_v = 10
                elif per_state == mino_state:
                    update_v = 0
                else:
                    q_list = []
                    for per_act in Action:
                        per_next_state = per_adj_states[per_state][per_act]
                        if per_next_state is None:
                            q_list.append(-1)
                            continue
                        q_sum = 0
                        count = 0
                        for mino_next_state in mino_adj_states[mino_state]:
                            if mino_next_state is None:
                                continue
                            q_sum += v_fun[mino_next_state, per_next_state, t]
                            count += 1
                        q_sum = q_sum / count
                        q_list.append(q_sum)
                    update_v = np.max(q_list)
                    opt_actions[mino_state, per_state, t-1] = np.argmax(q_list)
                v_fun[mino_state, per_state, t-1] = update_v
    return v_fun, opt_actions

def optimal_policy(opt_actions,T,init_per,init_mino,mino_adj_states,per_adj_states):
    person_states = [init_per]
    mino_states = [init_mino]
    cur_per_state = init_per
    cur_mino_state = init_mino
    for t in range(T):
        opt_act = opt_actions[cur_mino_state,cur_per_state,t]
        next_per_state = per_adj_states[cur_per_state][opt_act]
        person_states.append(next_per_state)

        if next_per_state == Maze.EXITIDX:
            break

        next_mino_state = np.random.choice(mino_adj_states[cur_mino_state])
        while next_mino_state is None:
            next_mino_state = np.random.choice(mino_adj_states[cur_mino_state])
        mino_states.append(next_mino_state)

        cur_per_state = next_per_state
        cur_mino_state = next_mino_state

    return person_states, mino_states


#%%
if __name__ == "__main__":
    # initialize
    T = 20
    maze_per = init_maze(Maze.WALL)
    maze_mino = init_maze([])
    init_per = state_to_idx(Maze.PerInit)
    init_mino = state_to_idx(Maze.MinoInit)
    mino_has_still = True # control minotaur actions

    per_adj_states = adjacent_states(maze_per, True)
    mino_adj_states = adjacent_states(maze_mino, mino_has_still)
    # set the exit state absorbing
    for i in range(len(Action)):
        if per_adj_states[Maze.EXITIDX][i] is not None:
            per_adj_states[Maze.EXITIDX][i] = Maze.EXITIDX

    # value iteration
    v_fun, opt_actions = value_iteration(T,init_per,init_mino,per_adj_states,mino_adj_states)

    # generate an optimal policy
    person_states, mino_states = optimal_policy(opt_actions,T,init_per,init_mino,mino_adj_states,per_adj_states)
    print("person states: " + person_states)
    print("mino states: " + mino_states)