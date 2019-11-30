"""
Use value iteration to solve the discounted infinite time horizon problem

https://github.com/CarlosLing/KTH_Reinforcement_Learning/blob/master/Lab1/Problem_2.py

https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb

https://github.com/bsridatta/ReinforcementLearning/blob/master/maze.py
"""
from enum import Enum
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

    #
    ROB_INIT = BANKS[0]


class State:
    """
    Abstractized the state S: S_p \times S_r
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
        As this
        """
        ret = Town.SIZE ** 2
        return ret


class Action(Enum):
    STAND = 'S'
    UP = 'U'
    DOWN = 'D'
    LEFT = 'L'
    RIGHT = 'R'


def prepare_trans():
    state_size = State.size()
    trans = np.zeros((state_size, state_size, len(Action)))

    # Loop over the all possible states

    return trans


def get_trans_prob(state, action):
    """
    Return a dictionary of state and probability as a represtation of
        p(s' | s, a)
    """
    pass


def value_iter_once(v_fun):

    return v_fun


if __name__ == "__main__":
    pass
