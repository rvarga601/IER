#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:25:29 2021

@author: rolandvarga
"""

"""
SARSA learning for a spherical robot system implemented by Roland Varga
The code is based on an already derived model by another group, 
for credits see the paper summarizing the methods and results.
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class SphericalRobotEnv(gym.Env):
    """
    Description:
        The spherical robot is able to roll on a flat, 2D surface by driving
        2 inner wheels which are connected to the outer shell. The robot 
        is initialized in a random position and its goal is to reach the 
        origin as soon as possible.
    Source:
        See the attached paper for the derivation of the discrete-time model
        from an existing continuous-time one.
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       x position                -Inf                    Inf
        1       y position                -Inf                    Inf
        2       Wheel angle 1             -Inf                    Inf
        3       Wheel angle 2             -Inf                    Inf
    Actions:
        Type: MultiDiscrete([3,3])
        Num   Action
        0     Turn the inner wheel in the defined negative direction
        1     Do not turn the inner wheel
        2     Turn the inner wheel in the defined positive direction
        Note: These 3 actions are defined for both inner wheels, so at every 
              time-step the input is a combination of 2 values (9 possible 
              input combinations)
    Reward:
        Reward is -1 if the step is taken in the direction of the origin 
        (with a given minimum step size) and it is -2 otherwise. Additionally,
        if the robot goes out the [-5,5] range in either the x or the y 
        position coordinates the reward is -5000. If the robot reaches a 
        given range around the origin the reward is +1000.
    Starting State:
        The positions are assigned a uniform random value in [-5,5], except 
        the area around the origin. The inner wheel angles are initialized to
        also uniformly distributed in the range [-0.01,0.01] radians.
    Episode Termination:
        Area around the origin is reached.
        Episode length is greater than 300.
    Solved Requirements:
        Not defined yet.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        
        self.dt = 0.1   # Discretization time-step
        
        self.rho = 0.3
        # the distance between the center of the sphere and the 
        # horizontal plane defined by the contact points of the wheels
        self.h = 0.75
        self.r = 1
        self.w = 0.8    # track width
        self.IsdivJ = 0.2   # ratio of the shell and inner part inertia
        self.c = self.rho * self.IsdivJ / (2*self.w*(self.IsdivJ+1))
        
        self.pos_threshold = 0.5
        self.dist_threshold = 0.1
        
        high = np.array([np.finfo(np.float32).max,
                          np.finfo(np.float32).max,
                          np.finfo(np.float32).max,
                          np.finfo(np.float32).max],
                        dtype=np.float32)
        
        self.action_space = spaces.MultiDiscrete([3,3])
        self.AVAIL_SPEED = [-10.0, 0., +10.0]
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        
        self.desired_pos = (0, 0) # Origin as desired position

        self.steps_beyond_done = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x1, x2, phi1, phi2 = self.state
        
        # Choose input
        phi1dot = self.AVAIL_SPEED[action[0]]
        phi2dot = self.AVAIL_SPEED[action[1]]
        
        # Simulate system with the chosen input
        k1 = self.r*self.rho/(2*self.h)*(phi1dot+phi2dot)
        k2 = self.c*(phi1-phi2)
        k3 = self.c*(phi1dot-phi2dot)
        if k3 == 0:
            x1_ = x1 + k1*(-math.sin(k2))*self.dt
            x2_ = x2 + k1*(math.cos(k2))*self.dt
        else:
            x1_ = x1 + k1/k3*(math.cos(k2+k3*self.dt)-math.cos(k2))
            x2_ = x2 + k1/k3*(math.sin(k2+k3*self.dt)-math.sin(k2))
        
        phi1_ = angle_normalize(phi1 + phi1dot * self.dt, 1/self.c)
        phi2_ = angle_normalize(phi2 + phi2dot * self.dt, 1/self.c)

        self.state = (x1_, x2_, phi1_, phi2_)
        
        done = bool(math.sqrt((x1_-self.desired_pos[0])**2+(x2_-self.desired_pos[1])**2) < self.pos_threshold or
                    x1_ > 5 or x1_ < -5 or x2_ > 5 or x2_ < -5 )
        
        if not done:
            # Calculate the change in the distances from the origin
            dist = (x1-self.desired_pos[0])**2+(x2-self.desired_pos[1])**2
            dist_ = (x1_-self.desired_pos[0])**2+(x2_-self.desired_pos[1])**2
            
            # Set reward based on the change in the distance
            if dist_-dist < -self.dist_threshold:
                reward = -1     # Step was made toward the origin
            else:
                reward = -2
        elif (x1_ > 5 or x1_ < -5 or x2_ > 5 or x2_ < -5):
            # The robot is out of the soft bounds
            reward = -5000
        elif self.steps_beyond_done is None:
            # Location just reached!
            self.steps_beyond_done = 0
            reward = 1000.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
       
        high = np.array([5, 5, 0.01, 0.01], dtype=np.float32)
        
        self.state = self.np_random.uniform(-high, high)
        
        # If the initialized location is in the area of the origin reinitilaze
        while np.sqrt(self.state[0]**2+self.state[1]**2) < 0.5:
            self.state = self.np_random.uniform(-high, high)
        self.steps_beyond_done = None
        return np.array(self.state)

    # TODO for the visualization
    # def render(self, mode='human'):

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
def angle_normalize(x, scaling):
    return (((x+np.pi*scaling) % (2*np.pi*scaling)) - np.pi*scaling)