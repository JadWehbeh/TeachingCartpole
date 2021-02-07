import numpy as np
import pickle
from scipy.signal import cont2discrete as c2d


# Class for storing parameters
class Params:
    def __init__(self, theta_span, thetad_span, x_span, v_span, a_span, dt, Ad,
                 Bd, gamma):
        self.theta_span = theta_span
        self.thetad_span = thetad_span
        self.x_span = x_span
        self.v_span = v_span
        self.a_span = a_span
        self.dt = dt
        self.Ad = Ad
        self.Bd = Bd
        self.gamma = gamma


# Define sampling range for state and control variables
theta_span = np.linspace(-3.0/180.0*np.pi,3.0/180.0*np.pi,71)
thetad_span = np.linspace(-0.8, 0.8, 21)
x_span = np.linspace(-0.3, 0.3, 21)
v_span = np.linspace(-0.3, 0.3, 11)
a_span = np.linspace(-2, 2, 31)

# Define prediction time step and number of value iterations
dt = 0.005
gamma = 0.7

# Define simulation constants
m = 0.5
M = 0.5
l = 0.5
b = 0.1
g = 9.82

# Build state prediction matrices
A = np.array(
    [[0, 1, 0, 0], [0, -4 * b / (4 * M + m), 0, 3 * m * g / (4 * M + m)],
     [0, -3 * b / (l * (4 * M + m)), 0,
      6 * (m + M) * g / (l * (4 * M + m))], [0, 0, 1, 0]])

B = np.array([[0, 4 / (4 * M + m), 0, 3 / (l * (4 * M + m))]])
B.shape = (4, 1)
C = np.identity(4)
D = np.zeros((4, 1))

Ad, Bd, Cd, Dd, dt_act = c2d((A, B, C, D), dt, method='zoh')

# Save parameters to file
params = Params(theta_span, thetad_span, x_span, v_span, a_span, dt, Ad, Bd,
                gamma)
f = open('trained_data/params.p', 'wb')
pickle.dump(params, f)
f.close()
