from transform import *
from dynamics import *
import numpy as np

Fw = np.array([0,0,1])
phi = 3.14/2
theta = 0
psi = 0
B_to_N = r_b_to_n(phi, theta, psi)

Fb = B_to_N.T @ Fw
print(Fb)