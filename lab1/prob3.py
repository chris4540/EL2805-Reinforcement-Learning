"""
Style and ref:
    <book_code>/chapter06/cliff_walking.py
    https://git.io/JePKr
"""
"""
The environment for Problem 3: Bank Robbing (Reloaded)
A  .  .  .  .
.  B  .  .  .
.  .  .  .  .
.  .  .  .  .
The player (us) starts at (0, 0) and the police starts at (3, 3)
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
from enum import IntEnum
import numpy as np
from tqdm import tqdm
from tqdm import trange
from utils.csvlogger import CustomizedCSVLogger as CSVLogger


class Town:
    """
    Environment config
    """

    # world height
    HEIGHT = 4

    # world width
    WIDTH = 4

    # Size of the world
    SIZE = HEIGHT * WIDTH

    # The location of the bank
    BANK = (1, 1)


class Action(IntEnum):
    STAND = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class InitialCond:
    police_state = (3, 3)
    rob_state = (0, 0)


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

    return next_state


def get_reward(rob_state, police_state):
    if rob_state == police_state:
        ret = -10
    elif rob_state == Town.BANK:
        ret = 1
    else:
        ret = 0
    return ret


def random_policy(state):
    # employ random policy
    action = np.random.choice(list(Action))
    return action


# def rob_policy(state):
#     # employ random policy
#     action = np.random.choice(list(Action))
#     return action


def state_to_idx(state):
    """
    Map state tuple to index

    Example:
    >> state_to_idx((2,3))
    11
    >> idx_to_state(11)
    (2, 3)
    """
    ret = np.ravel_multi_index(state, dims=(Town.HEIGHT, Town.WIDTH))
    return ret


def idx_to_state(idx):
    ret = np.unravel_index(idx, shape=(Town.HEIGHT, Town.WIDTH))
    return ret


def get_lr(n_visit):
    lr = (1. / n_visit) ** (0.66667)  # 2 / 3 = 0.66667
    return lr


if __name__ == "__main__":
    num_actions = len(Action)
    num_states = Town.SIZE
    # n_iters = int(1E7)
    n_iters = int(1E6)
    discount = 0.8  # the lambda coeff
    log_freq = 10000

    logger = CSVLogger('prob3a.csv')

    # q_fun.shape = (rob state, police state, action)
    # init. with zeros
    q_fun = np.zeros((num_states, num_states, num_actions))
    n_visits = np.zeros((num_states, num_states, num_actions))

    # initial condition
    rob_state = InitialCond.rob_state
    police_state = InitialCond.police_state

    init_r_i = state_to_idx(rob_state)
    init_p_i = state_to_idx(police_state)
    print(init_r_i, init_p_i)

    q_fun_ref = np.array(q_fun)
    sqsum_delta_q = 0

    pbar_desc = "ITERATION - v_fun_0: {:.2f}; sqsum(q - q'): {:.2f}"
    pbar = trange(n_iters)
    for t in pbar:
        # Make an action according to the policy
        police_act = random_policy(police_state)
        rob_act = random_policy(rob_state)

        # Make a move according to the action (s_{t+1})
        new_police_state = step(police_state, police_act)
        new_rob_state = step(rob_state, rob_act)
        new_r_i = state_to_idx(new_rob_state)
        new_p_i = state_to_idx(new_police_state)

        # Mark down n_visits
        r_i = state_to_idx(rob_state)
        p_i = state_to_idx(police_state)
        n_visits[r_i, p_i, rob_act] += 1
        n_visit = n_visits[r_i, p_i, rob_act]

        # calculate reward
        reward = get_reward(new_rob_state, new_police_state)

        # calculate learning rate
        alpha = get_lr(n_visit)

        # update
        update = reward
        # consider the discounted future
        update += (discount * np.max(q_fun[new_r_i, new_p_i, :]))
        update += -q_fun[r_i, p_i, rob_act]

        delta_q = alpha * update
        q_fun[r_i, p_i, rob_act] += delta_q

        # Calculate the value function at begining point
        v_fun_0 = np.max(q_fun[init_r_i, init_p_i, :])

        # replace the state
        police_state = new_police_state
        rob_state = new_rob_state

        # logging and display
        pbar.desc = pbar_desc.format(v_fun_0, sqsum_delta_q)

        if t % log_freq == 0:
            sqsum_delta_q = np.sum((q_fun - q_fun_ref)**2)
            pbar.desc = pbar_desc.format(v_fun_0, sqsum_delta_q)
            q_fun_ref = np.array(q_fun)
            logger.log(v_fun_0=v_fun_0, delta_q=delta_q,
                       sqsum_q_q_pi=sqsum_delta_q)
