import numpy as np
import pickle

# Define epsilon error for comparisons
eps = 0.0001


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


# Return reward of 10 if the goal is achieved
def reward(s):
    if abs(s[3]) < np.pi/1800 + eps:
        error = 10.0*np.sum(np.abs(s))
        return 10.0 - error
    else:
        return 0.0

# Return indices of nearest discrete neighbour to predicted state s.
# Return -1 for i[0] if any of the indices are out of the bounds considered.
def nearest_neighbour(s, params):
    i = np.zeros((4))
    i[0] = nearest_index(s[0], params.x_span)
    i[1] = nearest_index(s[1], params.v_span)
    i[2] = nearest_index(s[2], params.thetad_span)
    i[3] = nearest_index(s[3], params.theta_span)
    if not (i > -1).all:
        i[0] = -1
    return i

# Finds the index corresponding to the nearest element of array to val.
# Returns -1 if val is outside the bounds of array
def nearest_index(val, array):
    i = np.searchsorted(array, val)
    if i < 1 or i > (array.size - 2):
        if val < array[0] or val > array[array.size - 1]:
            return -1
        else:
            return i
    if np.abs((val - array[i])) > np.abs((val - array[i + 1])):
        return (i + 1)
    else:
        return i


# Fetch parameters for prediction
g0 = open('trained_data/params.p', 'rb')
params = pickle.load(g0)
g0.close()

g1 = open('trained_data/pred_array.p', 'rb')
pred_array = pickle.load(g1)
g1.close()

# Initialize arrays to be used
val_pred = np.zeros(params.a_span.size)
val_array = - np.ones((params.x_span.size, params.v_span.size,
                      params.thetad_span.size, params.theta_span.size))
pol_array = np.zeros((params.x_span.size, params.v_span.size,
                      params.thetad_span.size, params.theta_span.size))

# Set terminal state value to zero
i_goal = nearest_neighbour(np.array((0,0,0,0)),params).astype(int)
val_array[i_goal[0],i_goal[1],i_goal[2],i_goal[3]] = 0


while True:
    delta = 0               # Reset value for delta

    # Iterate over all states
    for i0 in range(0, params.x_span.size - 1):
        for i1 in range(0, params.v_span.size - 1):
            for i2 in range(0, params.thetad_span.size - 1):
                for i3 in range(0, params.theta_span.size - 1):
                    # Find maximum value over all actions
                    for i4 in range(0, params.a_span.size - 1):
                        # Fetch index of predicted state
                        i_pred = pred_array[i0, i1, i2, i3, i4].astype(int)
                        # Check if predicted state is within discreitzation
                        if abs(i_pred[0] + 1) > eps:
                            s_pred = np.array((params.x_span[i_pred[0]],
                                               params.v_span[i_pred[1]],
                                               params.thetad_span[i_pred[2]],
                                               params.theta_span[i_pred[3]]))
                            # Update predicted value function
                            val_pred[i4] = reward(
                                s_pred
                            ) + params.gamma * val_array[i_pred[0], i_pred[1],
                                                         i_pred[2], i_pred[3]]
                        # Set predicted value function to -10 if not
                        else:
                            val_pred[i4] = -10.0
                    # Update value array and policy with max over actions
                    v = val_array[i0, i1, i2, i3]
                    val_array[i0, i1, i2, i3] = np.max(val_pred)
                    pol_array[i0, i1, i2, i3] = params.a_span[np.argmax(
                        val_pred, axis=0)]
                    delta = max(delta, abs(v - val_array[i0, i1, i2, i3]))

    # Exit once converged sufficiently
    if delta < 0.01:
        break

# Save arrays  obtained
f1 = open('trained_data/val_array.p', 'wb')
pickle.dump(val_array, f1)
f1.close()

f2 = open('trained_data/pol_array.p', 'wb')
pickle.dump(pol_array, f2)
f2.close()
