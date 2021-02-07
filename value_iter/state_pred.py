import numpy as np
import pickle


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


# Predict next state from current state s and action a based on discrete
# linear dynamics
def lin_pred(s, a, params):
    return np.dot(params.Ad, s) + params.Bd * a


# Return indices of nearest discrete neighbour to predicted state s.
# Return -1 for i[0] if any of the indices are out of the bounds considered.
def nearest_neighbour(s, params):
    i = np.zeros((4))
    i[0] = nearest_index(s[0], params.x_span)
    i[1] = nearest_index(s[1], params.v_span)
    i[2] = nearest_index(s[2], params.thetad_span)
    i[3] = nearest_index(s[3], params.theta_span)
    if not (i > -0.5).all():
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
g = open('trained_data/params.p', 'rb')
params = pickle.load(g)
g.close()

pred_array = np.zeros(
    (params.x_span.size, params.v_span.size, params.thetad_span.size,
     params.theta_span.size, params.a_span.size, 4))

# Iterate over states and actions
i0 = 0
for s0 in params.x_span:
    print(i0)
    i1 = 0
    for s1 in params.v_span:
        print(i1)
        i2 = 0
        for s2 in params.thetad_span:
            i3 = 0
            for s3 in params.theta_span:
                i4 = 0
                for s4 in params.a_span:
                    # Find index of nearest neighbour of prediction and store it
                    s = np.array((s0, s1, s2, s3))
                    s.shape = (4, 1)
                    i_pred = nearest_neighbour(lin_pred(s, s4, params), params)
                    pred_array[i0,i1,i2,i3,i4,:] = i_pred
                    i4 = i4 + 1
                i3 = i3 + 1
            i2 = i2 + 1
        i1 = i1 + 1
    i0 = i0 + 1

f = open('trained_data/pred_array.p', 'wb')
pickle.dump(pred_array, f)
f.close()
