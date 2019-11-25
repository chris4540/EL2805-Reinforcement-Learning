"""
Style:
    <book_code>/chapter06/cliff_walking.py
    https://git.io/JePKr
"""
"""
The environment for Problem 3: Bank Robbing (Reloaded)
A  .  .  .  .
.  B  .  .  .
.  .  .  .  .
.  .  .  .  .
The player (us) starts at (0, 0) and the police starts at (4, 4)
The reward for each round (staying or moving?) in the bank results in a reward of +1
The reward for getting caught by the police results in a reward of -10
(Same cell as the police)
Actions are performed uniformly at random (up, down, left, right, stay).
We observer the position of the robber and the police at each time step.
a) Solve the problem by implementing the Q-learning algorithm exploring actions uniformly at
   random. Create a plot of the value function over time (in particular, for the initial state),
   showing the convergence of the algorithm. Note: Expect the value function to converge
   after roughly 10 000 000 iterations (for step size 1/n(s, a) 2/3 , where n(s, a) is the number of
   updates of Q(s, a))
"""
from enum import Enum
import numpy as np


class Town:
    """
    Environment config
    """

    # world height
    HEIGHT = 4

    # world width
    WIDTH = 4


class Action(Enum):
    UP = 'U'
    DOWN = 'D'
    LEFT = 'L'
    RIGHT = 'R'
    STAND = 'S'


def step(state, action):
    """
    Function to take one step forward in the environment given the environment,
    state and action.
    Returns a new state. This also checks if the new position is inside the grid.
    """
    # unpack it first
    i, j = state

    # ----------------------------------------
    # Consider the next state given the action
    # ----------------------------------------
    if action == Action.UP:
        next_state = (max(i - 1, 0), j)
    elif action == Action.LEFT:
        next_state = (i, max(j - 1, 0))
    elif action == Action.RIGHT:
        next_state = (i, min(j + 1, Town.WIDTH - 1))
    elif action == Action.DOWN:
        next_state = (min(i + 1, Town.HEIGHT - 1), j)
    elif action == Action.STAND:
        next_state = (i, j)
    else:
        raise ValueError("Action for stepping is not valid")

    # ----------------------------------------
    # consider the reward
    # ----------------------------------------
    reward = 0
    # if (action == ACTION_DOWN and i == 2 and 1 <= j <= 10) or (
    #         action == ACTION_RIGHT and state == START):
    #     reward = -100
    #     next_state = START

    return next_state, reward


def police_policy():
    pass


def rob_policy():
    pass


if __name__ == "__main__":
    num_actions = len(Action)
    num_states = 16
    # q_fun.shape = rob state, police state, action
    q_fun = np.zeros((num_states, num_states, num_actions))
