
import numpy as np
import os
from enum import IntEnum
import matplotlib.pyplot as plt

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
    return np.pad(maze, 1, mode='constant')

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
    ajacent_states[cur_state][action] = next_state (None for unavailable actions)
"""
def adjacent_states(maze, can_still):
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

            if can_still:
                next_states.append(state_to_idx((i,j)))

            adj_states.append(next_states)

    assert(len(adj_states)==Maze.SIZE)

    return adj_states


def get_reward(per_state, mino_state, maze):
    # per_pos = idx_to_state(per_state)
    if (per_state == Maze.EXITIDX) and (mino_state != Maze.EXITIDX):
        ret = 10
    elif per_state == mino_state:
        ret = -1
    # elif maze[per_pos[0]+1,per_pos[1]+1]==0 :
    #     ret = -10
    else:
        ret = 0
    return ret

# valfun: mino_state * per_state * T
def init_valfun(T):
    if T > 1:
        v_fun = -np.ones((Maze.SIZE,Maze.SIZE,T))
        v_fun[:,:,T-1] = 0
        v_fun[:, Maze.EXITIDX, T-1] = 1
        v_fun[Maze.EXITIDX, Maze.EXITIDX,T-1] = 0
        for wall in Maze.WALL:
            idx = state_to_idx(wall)
            v_fun[idx,:,T-1] = -1
    elif T == 1:
        v_fun = np.zeros((Maze.SIZE,Maze.SIZE))
        v_fun[:, Maze.EXITIDX] = 1
        v_fun[Maze.EXITIDX, Maze.EXITIDX] = 0
        for wall in Maze.WALL:
            idx = state_to_idx(wall)
            v_fun[idx,:] = -1
    else:
        raise TypeError('Wrong range: T < 1!')
    return v_fun

def dynamic_programming(T, init_per, init_mino, per_adj_states, mino_adj_states):
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

def value_iteration(gamma, epsilon, iter_num, person_adj_states, mino_adj_states, maze):
    q_fun = np.zeros((Maze.SIZE, Maze.SIZE, len(Action)))
    v_fun = init_valfun(1)
    vb_fun = np.zeros((Maze.SIZE, Maze.SIZE))
    opt_policy = np.zeros((Maze.SIZE, Maze.SIZE), dtype=int)
    t = 0
    diff = np.linalg.norm(v_fun - vb_fun)
    while diff > epsilon and t < iter_num:
        vb_fun = np.copy(v_fun)
        for m in range(Maze.SIZE):
            for p in range(Maze.SIZE):
                if not per_adj_states[p]:
                    continue
                for act in Action:
                    next_per_state = person_adj_states[p][act]
                    if next_per_state is None:
                        continue
                    reward = get_reward(next_per_state, m, maze)
                    v_sum = 0
                    count = 0
                    for next_mino_state in mino_adj_states[m]:
                        if next_mino_state is None:
                            continue
                        v_sum += vb_fun[next_mino_state, next_per_state]
                        count += 1
                    v_sum = v_sum / count
                    q_fun[m,p,act] = reward + gamma * v_sum
        v_fun = np.max(q_fun,-1)
        t += 1
        diff = np.linalg.norm(v_fun - vb_fun)
        #print(diff)
    opt_policy = np.argmax(q_fun,-1)
    return v_fun, opt_policy

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

def simulate(opt_actions,init_per,init_mino,mino_adj_states,per_adj_states):
    person_states = [init_per]
    mino_states = [init_mino]
    cur_per_state = init_per
    cur_mino_state = init_mino
    while (cur_per_state != Maze.EXITIDX) and (cur_per_state != cur_mino_state):
        opt_act = opt_actions[cur_mino_state,cur_per_state]
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
    ## initialize
    T = 20
    maze_per = init_maze(Maze.WALL)
    maze_mino = init_maze([])
    init_per = state_to_idx(Maze.PerInit)
    init_mino = state_to_idx(Maze.MinoInit)
    # mino_can_still: if minotaur is allowed to stand still
    mino_can_still = True

    per_adj_states = adjacent_states(maze_per, True)
    mino_adj_states = adjacent_states(maze_mino, mino_can_still)
    # set the exit state absorbing
    for i in range(len(Action)):
        if per_adj_states[Maze.EXITIDX][i] is not None:
            per_adj_states[Maze.EXITIDX][i] = Maze.EXITIDX

    ## bellman dynamic programming
    v_fun, opt_actions = dynamic_programming(T,init_per,init_mino,per_adj_states,mino_adj_states)

    ## generate an optimal policy
    person_states, mino_states = optimal_policy(opt_actions,T,init_per,init_mino,mino_adj_states,per_adj_states)
    print("person states: ")
    print(person_states)
    print("mino states: ")
    print(mino_states)

    ## calculate the maximal probability
    exit_probs = []
    T_list = list(range(1,31))
    round_num = 10000
    for t in T_list:
        win_count = 0
        v_fun, opt_actions = dynamic_programming(t,init_per,init_mino,per_adj_states,mino_adj_states)
        for i in range(round_num):
            p_states, m_states = optimal_policy(opt_actions,t,init_per,init_mino,mino_adj_states,per_adj_states)
            if (p_states[-1]==Maze.EXITIDX) and (m_states[-1]!=Maze.EXITIDX):
                win_count += 1
        exit_probs.append(win_count/round_num)
    print("exit probability without stand still: ")
    print(exit_probs)

    plt.scatter(T_list, exit_probs)
    plt.xlabel('T')
    plt.ylabel('maximal probability of exiting')
    if mino_can_still:
        plt.title('Minotaur can stand still')
        plt.savefig('report/stand_still.pdf')
    else:
        plt.title('Minotaur can not stand still')
        plt.savefig('report/no_stand_still.pdf')
    # plt.show()

    ## infinite horizon MDP
    gamma = 0.97
    eps = 0.001
    iter_num = 500
    inf_v_fun, inf_opt_policy = value_iteration(gamma,eps,iter_num,per_adj_states,mino_adj_states,maze_per)

    # generate 10000 games
    win_num = 0
    for i in range(10000):
        inf_person_states, inf_mino_states = simulate(inf_opt_policy,init_per,init_mino,mino_adj_states,per_adj_states)
        if (inf_person_states[-1]==Maze.EXITIDX) and (inf_mino_states[-1]!=Maze.EXITIDX):
            win_num += 1
    print("survive probability: " + str(win_num/10000))
    print("infiite horizon person states: ")
    print(inf_person_states)
    print("infinite horizon mino states: ")
    print(inf_mino_states)
