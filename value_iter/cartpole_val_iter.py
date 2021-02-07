'''
This file holds a cartpole simulator using physics functions borrowed from a previous
research project. Those are:
Copyright (c) 2017, Juan Camilo Gamboa Higuera, Anqi Xu, Victor Barbaros, Alex Chatron-Michaud, David Meger

The GUI is new in 2020 and was started from the pendulum code of Wesley Fernandes
https://pastebin.com/zTZVi8Yv
python simple pendulum with pygame

The rest of the file and instructions are written by David Meger for the purposes of supporting
his teaching in RL and Robotics. Please use this freely for any purpose, but acknowledgement of sources
is always welcome.
'''

# Simulate cartpole under LQR feedback

import pygame
import math
import pickle
import numpy as np
from scipy.integrate import ode

# The very basic code you should know and interact with starts here. Sets some variables that you
# might change or add to, then defines a function to do control that is currently empty. Add
# more logic in and around that function to make your controller work/learn!

x0 = [0.0,0,0,np.pi+np.pi/10000]           # This specifies the average starting state

                                        # change this if you want to do swing-up.
                                        # The meaning of the state dimensions are
                                        # state[0] : cart position (x)
                                        # state[1] : cart velocity (x_dot)
                                        # state[2] : pole angular velocity (theta_dot)
                                        # state[3] : pole angle (theta)

goal = np.array([ 0, 0, 0, np.pi ])     # This is where we want to end up. Perfectly at the centre
                                        # with the pole vertical.


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

# TODO: Fill in this function
def computeControl( x ):

    i = nearest_neighbour(x-goal,params).astype(int)
    control = pol_array[i[0],i[1],i[2],i[3]]
    print(i)
    print(control)
    return control

# Load parameters and policy array
f1 = open('trained_data/params.p','rb')
params = pickle.load(f1)
f1.close()

f2 = open('trained_data/pol_array.p','rb')
pol_array = pickle.load(f2)
f2.close

# After this is all the code to run the cartpole physics, draw it on the screen, etc.
# You should not have to change anything below this, but are encouraged to read and understand
# as much as possible.

# VARIABLES FOR GUI/INTERACTION
screen_width, screen_height = 800, 400   # set the width and height of the window
                           # (you can increase or decrease if you want to, just remind to keep even numbers)
Done = False                # if True,out of while loop, and close pygame
Pause = False               # when True, freeze the pendulum. This is
                            # for debugging purposes

#COLORS
white = (255,255,255)
black = (0,0,0)
gray = (150, 150, 150)
Dark_red = (150, 0, 0)
radius = 7
cart_width = 30
cart_height = 15
pole_length = 100
cart_x_to_screen_scaling = 100.0

#BEFORE STARTING GUI
pygame.init()
background = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# A simple class to simulate cartpole physics using an ODE solver
class CartPole(object):

    # State holds x, x_dot, theta_dot, theta (radians)
    def __init__(self, X0):
        self.g = 9.82
        self.m = 0.5
        self.M = 0.5
        self.l = 0.5
        self.b = 1.0

        self.X0 = self.x = np.array(x0,dtype=np.float64).flatten()
        self.x = self.X0
        self.t = 0

        self.u = 0

        # This is a key line that makes this class an accurate version of cartpole dynamics.
        # The ODE solver is connected with our instantaneous dynamics equations so it can do
        # the hard work of computing the motion over time for us.
        self.solver = ode(self.dynamics).set_integrator('dopri5', atol=1e-12, rtol=1e-12)
        self.set_state(self.x)

    # For internal use. This connects up the local state in the class
    # with the variables used by our ODE solver.
    def set_state(self, x):
        if (self.x is None or np.linalg.norm(x-self.x) > 1e-12):
            self.x = np.array(x,dtype=np.float64).flatten()
        self.solver = self.solver.set_initial_value(self.x)
        self.t = self.solver.t

    # Convenience function. Allows for quickly resetting back to the initial state to
    # get a clearer view of how the control works.
    def reset(self):
        self.x = self.X0
        self.t = 0
        self.set_state(self.x)

    # Draw the cart and pole
    def draw(self, bg):
        cart_centre = (int(screen_width/2+self.x[0]*cart_x_to_screen_scaling), int(screen_height/2))
        pole_end = (int(cart_centre[0] + pole_length * math.sin(self.x[3])), int(cart_centre[1]+ pole_length*math.cos(self.x[3])))
        pygame.draw.rect(bg, black, [cart_centre[0]-cart_width/2, cart_centre[1]-cart_height/2, cart_width, cart_height])
        pygame.draw.lines(bg, black, False, [cart_centre, pole_end], 2)
        pygame.draw.circle(bg, Dark_red, cart_centre, radius - 2)
        pygame.draw.circle(bg, Dark_red, pole_end, radius)

    # These equations are simply typed in from the dynamics
    # on the assignment document. They have been derived
    # for a pole of uniform mass using the Lagrangian method.
    def dynamics(self,t,z):

        f = np.array([self.u])

        sz = np.sin(z[3])
        cz = np.cos(z[3])
        cz2 = cz*cz

        a0 = self.m*self.l*z[2]*z[2]*sz
        a1 = self.g*sz
        a2 = f[0] - self.b*z[1]
        a3 = 4*(self.M+self.m) - 3*self.m*cz2

        dz = np.zeros((4,1))
        dz[0] = z[1]                                                            # x
        dz[1] = (  2*a0 + 3*self.m*a1*cz + 4*a2 )/ ( a3 )                       # dx/dt
        dz[2] = -3*( a0*cz + 2*( (self.M+self.m)*a1 + a2*cz ) )/( self.l*a3 )   # dtheta/dt
        dz[3] = z[2]                                                            # theta

        return dz

    # Takes the command, u, and applies it to the system for dt seconds.
    # Note that the solver has already been connected to the dynamics
    # function in the constructor, so this function is effectively
    # evaluating the dynamics. The solver does this in an "intelligent" way
    # that is more accurate than dt * accel, rather it evaluates the dynamics
    # at several points and correctly integrates over time.
    def step(self,u,dt=None):

        self.u = u

        if dt is None:
            dt = 0.005
        t1 = self.solver.t + dt
        while self.solver.successful and self.solver.t < t1:
            self.solver.integrate(self.solver.t+ dt)
        self.x = np.array(self.solver.y)
        self.t = self.solver.t
        return self.x

    def get_state(self):
        return self.x

# The next two are just helper functions for the display.
# Draw a grid behind the cartpole
def grid():
    for x in range(50, screen_width, 50):
        pygame.draw.lines(background, gray, False, [(x, 0), (x, screen_height)])
        for y in range(50, screen_height, 50):
            pygame.draw.lines(background, gray, False, [(0, y), (screen_width, y)])

# Clean up the screen and draw a fresh grid and the cartpole with its latest state coordinates
def redraw():
    background.fill(white)
    grid()
    pendulum.draw(background)
    pygame.display.update()

# Starting here is effectively the main function.
# It's a simple GUI drawing loop that calls to your code to compute the control, sets it to the
# cartpole class and loops the GUI to show what happened.
pendulum = CartPole(x0)
state = pendulum.get_state()

while not Done:
    clock.tick(240)             # GUI refresh rate

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Done = True
        if event.type == pygame.KEYDOWN:    # "r" key resets the simulator
            if event.key == pygame.K_r:
                pendulum.reset()
            if event.key == pygame.K_p:     # holding "p" key freezes time
                Pause = True
        if event.type == pygame.KEYUP:      # releasing "p" makes us live again
            if event.key == pygame.K_p:
                Pause = False

    if not Pause:

        control = computeControl( state )  # This is the call to the code you write
        state = pendulum.step(control)

        redraw()

pygame.quit()
