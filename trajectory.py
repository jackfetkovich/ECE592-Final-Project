import numpy as np
from numba import int64, float64, boolean
from numba.experimental import jitclass
from numba import njit



class Trajectory:
    def __init__(self, waypoints):
        # Waypoint should look like [x, y, z, phi, theta, psi]
        self.waypoints = waypoints

    def sample_trajectory(self, t):
        mask = self.waypoints[:, 6] >= t
        if np.any(mask):
            smallest_idx = np.argmax(mask) # Find the first waypoint with larger time than requested
            if smallest_idx == 0:
                return self.waypoints[0, 0:6] # Return first waypoint

            return lin_interpolate(
                self.waypoints[smallest_idx - 1, 0:6],  # previous
                self.waypoints[smallest_idx, 0:6],      # next
                (t - self.waypoints[smallest_idx - 1, 6]) / (self.waypoints[smallest_idx,  6] - self.waypoints[smallest_idx - 1, 6])
            )
        else:
            return self.waypoints[-1, 0:6]
        
def lin_interpolate(x1, x2, pct):
    return x1 + pct*(x2-x1)