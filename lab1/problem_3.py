# -*- coding: utf-8 -*-
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
from enum import Enum
import numpy as np
from tqdm import tqdm
from tqdm import trange
from utils.csvlogger import CustomizedCSVLogger as CSVLogger
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path


class Algorithm(Enum):
    q_learn = 'Q_learn'
    sarsa = 'SARSA'


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


def random_policy(state=None):
    # employ random policy
    action = np.random.choice(list(Action))
    return action


def eps_greedy_policy(rob_state, police_state, q_fun, eps=0.1):
    """
    Perform the ε-greedy policy to map a state to an action

    Args:
        rob_state (tuple[idx]): The rob state
        police_state (tuple[idx]): The police state
        q_fun (np.ndarry): The current estimation of the q-function
                           (i.e. the state-action value function)
        eps (float): The prob. of chossing randomized action in this policy

    """

    if np.random.binomial(1, eps) == 1:
        # Prob(take randomize policy)
        return random_policy()
    else:
        # Map their state to index
        r_i = state_to_idx(rob_state)
        p_i = state_to_idx(police_state)
        action_idx = np.argmax(q_fun[r_i, p_i, :])
        action = Action(action_idx)
        return action


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
    # parser
    parser = ArgumentParser(description='Lab 3 run script',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eps', default=0.1, type=float,
                        help="The ε in ε-greedy policy.")
    parser.add_argument('--algo', type=str, choices=[a.value for a in Algorithm],
                        required=True, help="The name of algorithm")
    parser.add_argument('--outfolder', default=str(Path('./')),
                        help="The name of algorithm")
    args = parser.parse_args()

    # Config
    num_actions = len(Action)
    num_states = Town.SIZE
    n_iters = int(1E7)
    discount = 0.8  # the lambda coeff
    log_freq = 10000
    eps = args.eps  # the ε in ε-greedy policy
    algo = Algorithm(args.algo)
    outfolder = Path(args.outfolder)
    print("---------------------------------------")
    print("Output folder = ", args.outfolder)
    if algo == Algorithm.sarsa:
        print("ε value in ε-greedy policy = ", eps)
    print("Algorithm = ", algo.value)
    print("---------------------------------------")
    # -------------------------------------
    if algo == Algorithm.q_learn:
        csv_fname = "{}.csv".format(algo.value)
    else:
        csv_fname = "{}-eps-{:.0E}.csv".format(algo.value, eps)
    logger = CSVLogger(outfolder / csv_fname)

    # q_fun.shape = (rob state, police state, action)
    # init. with zeros
    q_fun = np.zeros((num_states, num_states, num_actions))
    n_visits = np.zeros((num_states, num_states, num_actions))

    # initial condition
    rob_state = InitialCond.rob_state
    police_state = InitialCond.police_state

    init_r_i = state_to_idx(rob_state)
    init_p_i = state_to_idx(police_state)

    # For logging
    q_fun_ref = np.array(q_fun)
    sqsum_delta_q = 0
    acc_reward = 0  # accumulate rewards

    pbar_desc = "ITERATION - V_0: {:.2f}; (q - q')**2: {:.2f}; reward: {:.2f};"
    pbar = trange(n_iters)
    for t in pbar:
        # Make an action according to the policy
        police_act = random_policy()
        if algo == Algorithm.sarsa:
            rob_act = eps_greedy_policy(
                rob_state, police_state, q_fun, eps=0.1)
        else:
            # q-learning
            rob_act = random_policy()

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
        acc_reward += reward

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

        if t % log_freq == 0:
            sqsum_delta_q = np.sum((q_fun - q_fun_ref)**2)
            pbar.desc = pbar_desc.format(v_fun_0, sqsum_delta_q, acc_reward)
            q_fun_ref = np.array(q_fun)
            logger.log(iter=t, v_fun_0=v_fun_0, delta_q=delta_q,
                       sqsum_q_q_pi=sqsum_delta_q, acc_reward=acc_reward)
