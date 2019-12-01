import numpy as np


theta = 0.5
repair_cost = 8
cont_cost = 6  # see the problem
# ----------------------------------------------
# Transition probs
trans_r = np.array(
    [[1, 0, 0],
     [1, 0, 0],
     [1, 0, 0]]
)
trans_c = np.array(
    [[1 - theta, theta, 0],
     [0, 1 - theta, theta],
     [0, 0, 1]]
)
trans = {
    'replace': trans_r,
    'continue': trans_c
}
# ----------------------------------------------
# Rewards
reward_r = np.array([-repair_cost] * 3).reshape(-1, 1)
reward_c = np.array([0, -cont_cost / 2, -cont_cost]).reshape(-1, 1)
reward = {
    'replace': reward_r,
    'continue': reward_c
}
# ----------------------------------------------
# Do the calculation
u3 = np.array([0, 0, 0]).reshape(-1, 1)

# t = 2
u2_r = reward['replace'] + trans['replace'].dot(u3)
u2_c = reward['continue'] + trans['continue'].dot(u3)
u2 = np.maximum(u2_r, u2_c)
print("==========================")
print("T = 2: ")
print("Repair value func: ", u2_r.T)
print("Cont.  value func: ", u2_c.T)
print("Value function:    ", u2.T)
print("==========================")
# t = 1
u1_r = reward['replace'] + trans['replace'].dot(u2)
u1_c = reward['continue'] + trans['continue'].dot(u2)
u1 = np.maximum(u1_r, u1_c)
print("==========================")
print("T = 1: ")
print("Repair value func: ", u1_r.T)
print("Cont.  value func: ", u1_c.T)
print("Value function:    ", u1.T)
print("==========================")

# t = 0
u0_r = reward['replace'] + trans['replace'].dot(u1)
u0_c = reward['continue'] + trans['continue'].dot(u1)
u0 = np.maximum(u0_r, u0_c)
print("==========================")
print("T = 0: ")
print("Repair value func: ", u0_r.T)
print("Cont.  value func: ", u0_c.T)
print("Value function:    ", u0.T)
print("==========================")
