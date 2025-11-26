import numpy as np
from numba import int64, float64, boolean
from numba.experimental import jitclass
from numba import njit


@njit
def mppi(sub, traj, time, params, K, T):
    means = np.array([3, 0, 0, 0, 0, 0])
    sigmas = np.array([5, 5, 5])
    X_calc = np.zeros((K, params.T + 1, 12))

    # x = [eta, eta_dot]: configuration in the global frame


    U = gen_normal_control_seq(0.3, 1, 0, params.max_w, params.K, params.T) #

    targets = np.zeros((params.T, 6)) # Discretize path for computation
    for i in range(params.T):
        targets[i] = traj.sample_trajectory(time + i * params.dt)

    for k in range(params.K):
        X_calc[k, 0, :] = x  # Initialize all trajectories with the current state
            
    costs = np.zeros(params.K) # initialize all costs
    last_u = np.zeros(2)
    for k in range(params.K):
        for t in range(len(targets)-1):
            u_nom = U[k,t]
            u_safe = u_nom
            X_calc[k, t + 1, :] = sub.forward_dynamics(X_calc[k, t, :], u_safe, params)
            next_x = X_calc[k, t+1, :]
                   
            current_target = targets[t]
            cost = cost_function(X_calc[k, t+1, :], u_safe, current_target)
            costs[k] += cost
            last_u = u_safe
        
        final_target = targets[-1]    
        terminal_cost_val = terminal_cost(X_calc[k, params.T, :], final_target) #Terminal cost of final state
        costs[k] += terminal_cost_val
        
        last_u = np.zeros(2)
        
    # Calculate weights for each trajectory
    weights = np.exp(-(costs - np.min(costs)) / params.lambda_)
    sum_weights = np.sum(weights)
    if sum_weights < 1e-10:
        weights = np.ones_like(weights) / len(weights)  # fallback to uniform
    else:
        weights /= sum_weights
    
    traj_weight_single = np.zeros(params.K)
    traj_weight_single[:] = weights

    # Compute the weighted sum of control inputs
    u_star = np.sum(weights[:, None, None] * U, axis=0)
    return u_star[0]

# Cost function
@njit
def cost_function(x, u, target):
    Q = np.diag(np.array([16, 16, 3, 0.00, 0.00]))  # State costs
    R = np.diag(np.array([0.0005,0.0001]))  # Input costs

    x_des = np.array([target[0], target[1], target[2], 0, 0])
    state_diff = x_des - x
    state_diff[2] = (state_diff[2] + np.pi) % (2 * np.pi) - np.pi
    state_cost = np.dot(state_diff.T, np.dot(Q,state_diff))

    cost = state_cost + np.dot(u.T, np.dot(R, u))
    return cost

# Terminal Cost Function
@njit
def terminal_cost(x, target):
    Q = np.diag(np.array([18, 18, 0.5, 0.00, 0.00]))
    x_des= np.array([target[0], target[1], target[2], 0, 0])
    state_diff = x_des - x
    state_diff[2] = (state_diff[2] + np.pi) % (2 * np.pi) - np.pi
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