"""
Use value iteration to solve the discounted infinite time horizon problem

https://github.com/CarlosLing/KTH_Reinforcement_Learning/blob/master/Lab1/Problem_2.py

https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb

https://github.com/bsridatta/ReinforcementLearning/blob/master/maze.py

https://github.com/OleguerCanal/KTH_RL-EL2805
"""
from enum import IntEnum
import numpy as np


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
    ROB_INIT = BANKS[0]


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


class Action(IntEnum):
    STAND = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


def prepare_trans():
    state_size = State.size()
    # trans.shape = [# of S', # of S, # of A]
    trans = np.zeros((state_size, state_size, len(Action)))

    # Loop over the all possible states
    for state_idx in range(state_size):
        s = State.from_idx(state_idx)
        # consider a situaion when police and rob in the same grid
        if s.at_the_same_pos:
            s_pi = State()  # go back to inital cond
            trans[s_pi.to_idx(), s.to_idx(), :] = 1.0
            continue    # skip loop over actions

        for a in Action:
            trans = get_trans_prob(s, a)

    return trans


def next_police_actions(state):
    # unpack the state first
    police_state, rob_state = state.unpack

    # consider the random action of the police
    police_row, police_col = police_state
    rob_row, rob_col = rob_state

    actions = list()

    if police_row == rob_row and police_col == rob_col:
        raise ValueError("Go to inital state")

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
    # check the actions the police will take
    next_police_actions = next_police_actions(state)


def next_state_of_a_role(state, action):
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


def value_iter_once(v_fun):
    return v_fun


if __name__ == "__main__":
    pass
