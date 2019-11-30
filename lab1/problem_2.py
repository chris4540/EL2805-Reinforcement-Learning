"""
Use value iteration to solve the discounted infinite time horizon problem

https://github.com/CarlosLing/KTH_Reinforcement_Learning/blob/master/Lab1/Problem_2.py

https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb

https://github.com/bsridatta/ReinforcementLearning/blob/master/maze.py

https://github.com/OleguerCanal/KTH_RL-EL2805
"""
from enum import IntEnum
import numpy as np
from tqdm import trange
from utils.csvlogger import CustomizedCSVLogger as CSVLogger
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


class Town:
    """
    Environment config
    """

    # world height
    HEIGHT = 3

    # world width
    WIDTH = 6

    # Size of the world
    SIZE = HEIGHT * WIDTH

    # The location of the banks
    BANKS = [
        (0, 0),
        (2, 0),
        (0, 5),
        (2, 5),
    ]

    # The location of the police station
    POLICE_STATION = (1, 2)

    # The rob init place
    ROB_INIT = (0, 0)


class State:
    r"""
    Abstractized the state S:
    .. math::
        S: S_p \times S_r
    """

    def __init__(self, police_state=None, rob_state=None):
        if police_state is None:
            self.police_state = Town.POLICE_STATION
        else:
            self.police_state = police_state

        if rob_state is None:
            self.rob_state = Town.ROB_INIT
        else:
            self.rob_state = rob_state

    def unpack(self):
        return self.police_state, self.rob_state

    def to_idx(self):
        indices = (*self.police_state, *self.rob_state)
        ret = np.ravel_multi_index(indices, dims=(
            Town.HEIGHT, Town.WIDTH, Town.HEIGHT, Town.WIDTH))
        return ret

    @classmethod
    def from_idx(cls, idx):
        """
        Constrcut the state from index
        Example:
            s = State.from_idx(100)
            print(s.police_state)   # (0, 5)
            print(s.rob_state)      # (1, 4)
            print(s.to_idx)         # 100
        """
        # indices = (*police_state, *rob_state)
        indices = np.unravel_index(idx, shape=(
            Town.HEIGHT, Town.WIDTH, Town.HEIGHT, Town.WIDTH))

        # unpack indices
        police_state = (indices[0], indices[1])
        rob_state = (indices[2], indices[3])

        # return an instanitation
        return cls(police_state, rob_state)

    @staticmethod
    def size():
        """
        Return the size of State
        """
        ret = Town.SIZE ** 2
        return ret

    def at_the_same_pos(self):
        ret = self.police_state == self.rob_state
        return ret

    def __repr__(self):
        ret = "Police: {}; Rob: {}".format(self.police_state, self.rob_state)
        return ret

    def apply_action(self, police_act, rob_act):
        next_p_state = next_state(self.police_state, police_act)
        next_r_state = next_state(self.rob_state, rob_act)
        return State(next_p_state, next_r_state)


class Action(IntEnum):
    STAND = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


def preparation():
    state_size = State.size()
    # trans.shape = [# of S', # of S, # of A]
    trans = np.zeros((state_size, state_size, len(Action)))
    # trans.shape = [# of S, # of A]
    rewards = np.zeros((state_size, len(Action)))

    # Loop over the all possible states
    for state_idx in range(state_size):
        s = State.from_idx(state_idx)
        # consider a situaion when police and rob in the same grid
        if s.at_the_same_pos():
            s_pi = State()  # go back to inital cond
            trans[s_pi.to_idx(), s.to_idx(), :] = 1.0
            # The reward is indep of actions
            rewards[s.to_idx(), :] = -50
            continue    # skip loop over actions

        # consider reward: r(s, a)
        if s.rob_state in Town.BANKS:
            rewards[s.to_idx(), :] = 10  # The reward is indep of actions

        for a in Action:
            for s_pi, p in get_trans_prob(s, a).items():
                trans[s_pi.to_idx(), s.to_idx(), a] = p

    # print(trans.shape)
    # print(np.sum(trans[], axis=0))
    ret = {
        'trans': trans,
        'rewards': rewards
    }
    return ret


def next_police_actions(state):
    # unpack the state first
    police_state, rob_state = state.unpack()

    # consider the random action of the police
    police_row, police_col = police_state
    rob_row, rob_col = rob_state

    actions = list()

    if police_row == rob_row and police_col == rob_col:
        raise ValueError("Goto inital state")

    # vertical movement
    if police_row > rob_row:  # police is on rob's south
        actions.append(Action.UP)
    elif police_row < rob_row:
        actions.append(Action.DOWN)
    else:  # same row
        actions.extend([Action.UP, Action.DOWN])

    # horizontal movement
    if police_col > rob_col:
        actions.append(Action.LEFT)
    elif police_col < rob_col:
        actions.append(Action.RIGHT)
    else:
        actions.extend([Action.LEFT, Action.RIGHT])

    assert len(actions) <= 3

    return actions


def get_trans_prob(state, action):
    """
    Return a dictionary of state and probability as a represtation of
        p(s' | s, a)
    Args:
        state ([type]): [description]
        action ([type]): [description]

    Returns:
        dict:  dictionary of probabilities as a function of next state s'
    """
    ret = dict()
    # check the actions the police will take
    police_actions = next_police_actions(state)

    # As the police takes actions randomly and the actions are distributed uniformly.
    prob = 1.0 / len(police_actions)

    police_state, rob_state = state.unpack()

    # check the next rob state
    next_rob_state = next_state(rob_state, action)

    # check the next police states
    for a in police_actions:
        next_police_state = next_state(police_state, a)
        s_pi = State(next_police_state, next_rob_state)
        ret[s_pi] = prob

    return ret


def next_state(state, action):
    """[summary]

    Args:
        state (tuple): the current state of the player (either police or rob)
        action (enum of Action): [description]

    Raises:
        ValueError: if action is not valid

    Returns:
        tuple: a new state according to the action
    """
    row, col = state
    # --------------------------------------------
    # Consider the next state given the action
    # --------------------------------------------
    if action == Action.UP:
        next_state = (max(row - 1, 0), col)
    elif action == Action.LEFT:
        next_state = (row, max(col - 1, 0))
    elif action == Action.RIGHT:
        next_state = (row, min(col + 1, Town.WIDTH - 1))
    elif action == Action.DOWN:
        next_state = (min(row + 1, Town.HEIGHT - 1), col)
    elif action == Action.STAND:
        next_state = (row, col)
    else:
        raise ValueError("Action for stepping is not valid")
    # --------------------------------------------
    return next_state


def value_iteration(lambda_=0.5):
    csv_fname = 'prob2_results/lab2_v_fun_0_lbda_{:.0E}.csv'.format(lambda_)
    logger = CSVLogger(csv_fname)
    state_size = State.size()
    v_func = np.zeros(state_size)
    max_iters = 500
    prep = preparation()
    trans = prep['trans']
    rewards = prep['rewards']

    eps = 1e-4
    theta = eps * (1 - lambda_) / lambda_
    delta = 0

    pbar_desc = "ITERATION - delta {:.2f} - V_0: {:.2f};"
    pbar = trange(max_iters)
    for t in pbar:
        # clone it
        v_func_old = np.array(v_func)

        # loop over state
        for s_idx in range(state_size):
            # s = State.from_idx(state_idx)
            expect_val_func = np.sum(
                trans[:, s_idx, :] * v_func[:, np.newaxis], axis=0)
            q_fun = rewards[s_idx, :] + lambda_ * expect_val_func

            update = np.max(q_fun)
            action = Action(np.argmax(q_fun))
            v_func[s_idx] = update

        # check the norm of the value fun
        delta = np.linalg.norm(v_func - v_func_old)
        pbar.desc = pbar_desc.format(delta, v_func[0])
        logger.log(iter=t, v_fun_0=v_func[0])

        if delta < theta:
            break

    # calculate the optimal policy
    opt_policy = np.zeros(state_size)
    for s_idx in range(state_size):
        expect_val_func = np.sum(
            trans[:, s_idx, :] * v_func[:, np.newaxis], axis=0)
        q_fun = rewards[s_idx, :] + lambda_ * expect_val_func
        action = Action(np.argmax(q_fun))
        opt_policy[s_idx] = action

    return v_func, opt_policy


def generate_game(v_func, policy, max_time=50):
    s = State()
    states = list()

    while len(states) < max_time:
        # record the state at time t
        states.append(s)

        # police action
        police_actions = next_police_actions(s)
        police_act = np.random.choice(police_actions)
        rob_act = policy[s.to_idx()]

        output_fmt = "Step {}: {}, Robber Action: {}"
        print(
            "Step {}: {}, Robber Action: {}".format(
                len(states), repr(s), Action(rob_act).name))

        s_pi = s.apply_action(police_act, rob_act)

        if s_pi.at_the_same_pos():
            states.append(s_pi)
            print(
                "Step {}: {}, Robber Action: {}".format(
                    len(states), repr(s_pi), 'RESTART'))
            s = State()
        else:
            s = s_pi

    return states


def animate_solution(states, lambda_):
    # Some colours
    LIGHT_RED = '#FFC4CC'
    LIGHT_GREEN = '#95FD99'
    BLACK = '#000000'
    WHITE = '#FFFFFF'
    LIGHT_PURPLE = '#E8D0FF'
    LIGHT_ORANGE = '#FAE0C3'

    # Map a color to each cell in the maze

    # Size of the maze
    rows = Town.HEIGHT
    cols = Town.WIDTH

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    title_fmt = 'λ = {}, Policy simulation at t = {}'
    ax.set_title(title_fmt.format(lambda_, 0))
    ax.set_xticks([])
    ax.set_yticks([])

    # color town

    # Give a color to each cell
    colored_maze = [[WHITE for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)

    def update(i):
        for g in grid.get_celld().values():
            g.set_facecolor(WHITE)
            g.get_text().set_text('')

        s = states[i]
        police_state, rob_state = s.unpack()
        grid[rob_state[0], rob_state[1]].get_text().set_text('robber')
        grid[rob_state[0], rob_state[1]].set_facecolor(LIGHT_GREEN)
        grid[police_state[0], police_state[1]].get_text().set_text('police')
        grid[police_state[0], police_state[1]].set_facecolor(LIGHT_RED)
        ax.set_title(title_fmt.format(lambda_, i))

    ani = animation.FuncAnimation(
        fig, update, interval=500, frames=len(states))

    gif_name = 'prob2_results/lab2_game_lbda_{:.0E}.gif'.format(lambda_)
    ani.save(gif_name, writer='imagemagick')


if __name__ == "__main__":

    v_func_0 = dict()
    for lbda in [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
        v_func, opt_policy = value_iteration(lbda)
        v_func_0[lbda] = v_func[0]
        states = generate_game(v_func, opt_policy, max_time=50)
        animate_solution(states, lbda)

    # Save v_0
    df = pd.Series(v_func_0).to_frame()
    df.columns = ['V_0']
    df.index.name = 'lambda'
    df.to_csv('prob2_results/lab2_v0.csv')
