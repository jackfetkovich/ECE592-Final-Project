import numpy as np
from numba import int64, float64, boolean
from numba.experimental import jitclass
from numba import njit
from transform import *
from dynamics import *

@njit
def mppi(x_init, traj, time, K, T, lam, dt, m, Ix, Iy, Iz, W, B, params):
    means = np.array([0, 0, 1000, 0, 0, 0])
    sigmas = np.array([600, 600, 300, 25, 25, 25])
    X_calc = np.zeros((K, T + 1, 12))

    U = gen_normal_control_seq(means, sigmas, K, T)

    targets = np.zeros((T, 6)) # Discretize path for computation
    for i in range(T):
        targets[i] = traj.sample_trajectory(time + i * dt)

    for k in range(K):
        X_calc[k, 0, :] =  x_init # Initialize all trajectories with the current state
            
    costs = np.zeros(K) # initialize all costs
    for k in range(K):
        for t in range(len(targets)-1):
            u_nom = U[k,t]
            X_calc[k, t + 1, :] = forward_dynamics(X_calc[k, t, 0:6], X_calc[k, t, 6:], u_nom, dt, m, Ix, Iy, Iz, W, B, params)
                   
            current_target = targets[t]
            cost = cost_function(X_calc[k, t+1, 0:6], u_nom, current_target)
            costs[k] += cost
        
        final_target = targets[-1]    
        terminal_cost_val = terminal_cost(X_calc[k, T, 0:6], final_target) #Terminal cost of final state
        costs[k] += terminal_cost_val
        
        
    # Calculate weights for each trajectory
    weights = np.exp(-(costs - np.min(costs)) / lam)
    sum_weights = np.sum(weights)
    if sum_weights < 1e-10:
        weights = np.ones_like(weights) / len(weights)  # fallback to uniform
    else:
        weights /= sum_weights
    
    traj_weight_single = np.zeros(K)
    traj_weight_single[:] = weights

    # Compute the weighted sum of control inputs
    u_star = np.sum(weights[:, None, None] * U, axis=0)
    return u_star[0]

# Cost function
@njit
def cost_function(x, u, target):
    Q = np.diag(np.array([10.0, 10.0, 300.0, 2.0, 2.0, 2.0]))  # State costs
    R = np.diag(np.array([0.001,0.001, 0.001, 0.001, 0.001, 0.0001]))  # Input costs

    x_des = np.array([target[0], target[1], target[2], target[3], target[4], target[5]])
    state_diff = x_des - x
    for i in range(3, 6, 1):
        state_diff[i] = wrap_angle(state_diff[i])
    
    state_cost = np.dot(state_diff.T, np.dot(Q,state_diff))

    cost = state_cost + np.dot(u.T, np.dot(R, u))
    return cost

# Terminal Cost Function
@njit
def terminal_cost(x, target):
    Q = np.diag(np.array([12.0, 12.0, 300.0, 2.0, 2.0, 2.0]))  # State costs
    x_des = np.array([target[0], target[1], target[2], target[3], target[4], target[5]])
    state_diff = x_des - x
    for i in range(3, 6, 1):
        state_diff[i] = wrap_angle(state_diff[i])
    terminal_cost = np.dot(state_diff.T, np.dot(Q,state_diff))
    return terminal_cost 

@njit
def gen_normal_control_seq(means, sigmas, K, T):
    return np.dstack((
        np.random.normal(loc=means[0], scale=sigmas[0], size=(K, T)), #Fx
        np.random.normal(loc=means[1], scale=sigmas[1], size=(K, T)), #Fy
        np.random.normal(loc=means[2], scale=sigmas[2], size=(K, T)), #Fz
        np.random.normal(loc=means[3], scale=sigmas[3], size=(K, T)), #phi_dot
        np.random.normal(loc=means[4], scale=sigmas[4], size=(K, T)), #theta_dot
        np.random.normal(loc=means[5], scale=sigmas[5], size=(K, T)), #psi_dot
    )) 
