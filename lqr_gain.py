# Calculate lqr gain for cartpole problem

import scipy.linalg
import numpy as np

def lqr( A, B, Q, R ):
    # Returns lqr gain given required matrices
    x = scipy.linalg.solve_continuous_are( A, B, Q, R )
    k = np.linalg.inv(R) * np.dot( B.T, x )
    return k

# Define simulation constants
m = 0.5
M = 0.5
l = 0.5
b = 0.1
g = 9.82

# Define and print lqr Matrices
A = np.array([[ 0,  1,                  0, 0                        ],
              [ 0,  -4*b/(4*M+m),       0, 3*m*g/(4*M+m)           ],
              [ 0,  -3*b/(l*(4*M+m)),   0, 6*(m+M)*g/(l*(4*M+m))   ],
              [ 0,  0,                  1, 0                        ]] )

B = np.array( [[0, 4/(4*M+m), 0, 3/(l*(4*M+m))]] )
B.shape = (4,1)

Q =  100*np.array([[ 1, 0, 0, 0 ],
                [ 0, 1, 0, 0 ],
                [ 0, 0, 1, 0 ],
                [ 0, 0, 0, 1 ]] )

R = np.array([[1]])

print( "A holds:",A)
print( "B holds:",B)
print( "Q holds:",Q)
print( "R holds:",R)

# Compute and print lqr gain
k = lqr( A, B, Q, R )
print( "k holds:",k)
