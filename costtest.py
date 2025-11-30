import numpy as np
from transform import *
def cost_function(x, u, target):
    Q = np.diag(np.array([10.0,10.0, 10.0, 0.0, 0.0, 8.0]))  # State costs
    # R = np.diag(np.array([0.0000000000001,0.0000000000001, 0.0000000000001, 0.0000000000001, 0.0000000000001, 0.0000000000001]))  # Input costs
    R = np.zeros((6,6))

    x_des = np.array([target[0], target[1], target[2], target[3], target[4], target[5]])
    state_diff = x_des - x
    for i in range(3, 6, 1):
        state_diff[i] = wrap_angle(state_diff[i])
    
    state_cost = np.dot(state_diff.T, np.dot(Q,state_diff))

    cost = state_cost + np.dot(u.T, np.dot(R, u))
    return cost


target = np.array([0.0,0.0,5.0,0.0,0.0, -1.57079633])
x = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 1])
u = np.zeros(6)

print(cost_function(x, u, target))