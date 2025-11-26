import numpy as np
from numba import int64, float64, boolean
from numba.experimental import jitclass
from numba import njit


@njit
def mppi(sub, traj, time, params, K, T, lam, dt):
    means = np.array([300, 0, 0, 0, 0, 0])
    sigmas = np.array([400, 400, 400, 0.1, 0.1, 0.4])
    X_calc = np.zeros((K, T + 1, 12))

    U = gen_normal_control_seq(means, sigmas, K, T)

    targets = np.zeros((T, 6)) # Discretize path for computation
    for i in range(T):
        targets[i] = traj.sample_trajectory(time + i * dt)

    for k in range(K):
        X_calc[k, 0, :] =  sub.telemetry.x() # Initialize all trajectories with the current state
            
    costs = np.zeros(K) # initialize all costs
    for k in range(K):
        for t in range(len(targets)-1):
            u_nom = U[k,t]
            X_calc[k, t + 1, :] = sub.forward_dynamics(X_calc[k, t, 0:6], u_nom)
                   
            current_target = targets[t]
            cost = cost_function(X_calc[k, t+1, :], u_nom, current_target)
            costs[k] += cost
            last_u = u_nom
        
        final_target = targets[-1]    
        terminal_cost_val = terminal_cost(X_calc[k, T, :], final_target) #Terminal cost of final state
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