#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:54:16 2021

@author: rolandvarga
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import savgol_filter
import pickle

#%matplotlib qt
#%matplotlib inline

# Set to 1 to repeat SARSA learning (With Intel Core i7-8750H it takes 
# around 70 minutes), 0 for loading previous result
REPEAT_LEARNING = 0

# Parameter to set which tests to do
DO_TEST1 = 1    # Simulate the system once and plot the trajectory
DO_TEST2 = 0    # Simulate the system 1000 times and plot success-rate

# Set to 1 to plot a projection of the state-value function V
PLOT_STATEVALUE = 1


#%% Load previous result
if REPEAT_LEARNING == 0:
    
    filename='train_6x6x20x60000.pickle'
    	
    with open(filename, 'rb') as f:
        cell_nums, dhat, durations, Q, reward_set, rhat, start_time, end_time, states_high, max_steps = pickle.load(f)
    

#%% SARSA learning

env = gym.make('SphericalRobot-v0')

#Function to choose the next action
def choose_action(state, eps):
    action=0
    if np.random.uniform(0, 1) < eps:
        # Select a random action
        action = env.action_space.sample()
    else:
        # Choose greedy action
        action = np.array(np.unravel_index(np.argmax(Q[state], axis=None), Q[state].shape))
        # action = np.argmax(Q[state])
    return action

    
#Convert continuous state-space to discrete
def discretize_state(observation_c, low, high, cell_nums):
    # Initialize the discretized observation
    observation_d = []
    
    # Loop through and discretize all 3 states
    for state,low_val,high_val,c_num in zip(observation_c,low,high,cell_nums):
        # Define intervals for the possible values
        bins = np.linspace(low_val,high_val,c_num+1,endpoint=True)
        
        # Discretize with NumPy function
        state_d = np.digitize(state, bins, right=True)
        
        # Check if the discrete values are valid
        assert state_d > 0 and state_d <= c_num
        
        observation_d.append(state_d-1)     # -1 to have values start at 0
        
    return observation_d
    
if REPEAT_LEARNING == 1:
    
    # Learning parameters
    epsilon = 0.3   # For start
    total_episodes = 100
    max_steps = 300
    alpha = 0.1
    gamma = 0.99
    
    # The discretization of the states
    states_high = np.array([6,6,2*np.pi/env.c])     # Set boundaries for the values
    cell_nums = np.array([6,6,20])      # Set the number of discrete cells
    
    #Initializing the Q-matrix
    Q = np.ones(np.append(cell_nums,[3,3]))
    #Function to update the Q-value
    def update(state, state2, reward, action, action2):
        predict = Q[state][action]
        target = reward + gamma * Q[state2][action2]
        Q[state][action] = Q[state][action] + alpha * (target - predict)
    
    #Initializing the reward
    # reward=0
    
    reward_set = []
    durations = []
    
    start_time = time.time()
    
    # Starting the SARSA learning
    for episode in range(total_episodes):
        t = 0
        cumm_reward = 0
        state1 = env.reset()
        state1_d = discretize_state(state1, -states_high, states_high, cell_nums)
        action1 = choose_action(tuple(state1_d), epsilon)
        
        states = [state1]
        
      
        while t < max_steps:
            # Visualizing the training, TODO
            # env.render()
              
            # Getting the next state
            state2, reward, done, info = env.step(action1)
            
            # Note: The 3rd state is the difference between the wheel angles
            state1_d = discretize_state(np.array([state1[0],state1[1], state1[2]-state1[3]]),
                                        -states_high, states_high, cell_nums)
            state2_d = discretize_state(np.array([state2[0],state2[1], state2[2]-state2[3]]), 
                                        -states_high, states_high, cell_nums)
      
            # Choosing the next action
            action2 = choose_action(tuple(state2_d), epsilon)
            
            # Updating the Q-value
            update(tuple(state1_d), tuple(state2_d), reward, tuple(action1), tuple(action2))
      
            # Update variables for next iteration
            state1 = state2
            action1 = action2
            
            # Save state to be able to plot trajectories
            states.append(state2)
              
            #Updating the respective vaLues
            t += 1
            cumm_reward += reward
              
            #If at the end of learning process
            if done:
                break
            
        reward_set.append(cumm_reward)
        durations.append(t)
        
        # plt.figure(0)   
        # x = np.array(states)[:,0]
        # y = np.array(states)[:,1]   
        # plt.scatter(x, y)
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)
        # plt.show()
    
    # Print time it took to run the learning
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    
    # Plot the filtered rewards during the learning
    plt.figure(1)
    #plt.plot(reward_set)
    rhat = savgol_filter(reward_set, 501, 3) # window size 501, polynomial order 3
    plt.plot(rhat)
    #plt.ylim(-500, 500)
    plt.xlabel(r"Episode [-]")
    plt.ylabel(r"Reward [-]")
    plt.legend()
    plt.savefig('reward_learning.eps', format='eps', bbox_inches='tight')
    plt.show()
    
    # Plot the filtered episode lengths during the learning
    plt.figure(2)
    #plt.plot(durations)
    dhat = savgol_filter(durations, 51, 3) # window size 51, polynomial order 3
    plt.plot(dhat)
    plt.show()
    

#%% Test 1: Generate one trajectory
if DO_TEST1 == 1:
    t = 0
    cumm_reward = 0
    state1 = env.reset()
    state1_d = discretize_state(state1, -states_high, states_high, cell_nums)
    action1 = choose_action(tuple(state1_d), 0.0)
    
    states = [state1]
    actions = [action1]
      
    while t < max_steps:
        #Visualizing the training
        # env.render()
          
        #Getting the next state
        state2, reward, done, info = env.step(action1)
        
        state1_d = discretize_state(np.array([state1[0],state1[1], state1[2]-state1[3]]), 
                                    -states_high, states_high, cell_nums)
        state2_d = discretize_state(np.array([state2[0],state2[1], state2[2]-state2[3]]), 
                                    -states_high, states_high, cell_nums)
      
        #Choosing the next action
        action2 = choose_action(tuple(state2_d), 0.0)
        
        #Learning the Q-value
        #update(tuple(state1_d), tuple(state2_d), reward, tuple(action1), tuple(action2))
      
        state1 = state2
        action1 = action2
        
        states.append(state2)
        actions.append(action2)
          
        #Updating the respective vaLues
        t += 1
        cumm_reward += reward
          
        #If at the end of learning process
        if done:
            break
    
    print(reward)
    
    # Plot trajectory on 2D plot
    plt.figure(3)
    x = np.array(states)[:,0]
    y = np.array(states)[:,1]
    plt.scatter(x, y)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xticks(np.arange(-5, 6, 1))
    plt.yticks(np.arange(-5, 6, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(r"$x_1$ [m]")
    plt.ylabel(r"$x_2$ [m]")
    plt.legend()
    plt.savefig('trajectory.eps', format='eps', bbox_inches='tight')
    plt.show()
    
    # Plot position states separately
    plt.figure(4)
    plt.plot(x, label="x1")
    plt.plot(y, label="x2")
    plt.xlabel(r"Time step [-]")
    plt.ylabel(r"Coordinate [m]")
    plt.legend()
    plt.savefig('trajectory_plot.eps', format='eps', bbox_inches='tight')
    plt.show()


#%% Test 2: Successful-unsuccessful tries
if DO_TEST2 == 1:
    cumm_rewards = []
    for k in range(1000):
        t = 0
        cumm_reward = 0
        state1 = env.reset()
        state1_d = discretize_state(state1, -states_high, states_high, cell_nums)
        action1 = choose_action(tuple(state1_d), 0.0)
    
        while t < max_steps:
            #Visualizing the training
            # env.render()
              
            #Getting the next state
            state2, reward, done, info = env.step(action1)
    
            
            state1_d = discretize_state(np.array([state1[0],state1[1], state1[2]-state1[3]]), 
                                        -states_high, states_high, cell_nums)
            state2_d = discretize_state(np.array([state2[0],state2[1], state2[2]-state2[3]]), 
                                        -states_high, states_high, cell_nums)
          
            #Choosing the next action
            action2 = choose_action(tuple(state2_d), 0.0)
            
            #Learning the Q-value
            #update(tuple(state1_d), tuple(state2_d), reward, tuple(action1), tuple(action2))
          
            state1 = state2
            action1 = action2
            
            #states.append(state2)
            #actions.append(action2)
              
            #Updating the respective vaLues
            t += 1
            cumm_reward += reward
              
            #If at the end of learning process
            if done:
                break
            
        cumm_rewards.append(cumm_reward)
        
    print("Average reward out of 1000 try: " + str(np.average(np.array(cumm_rewards))))
    
    plt.figure(5)
    plt.hist(cumm_rewards,np.array([-1000,0,1000]))
    plt.show()

#%% Additional plot: State-value function
if PLOT_STATEVALUE == 1:
    V = np.zeros([cell_nums[0],cell_nums[1]])
    
    for k in range(V.shape[0]):
        for l in range(V.shape[1]):
            V[k,l]=np.amax(Q[k,l,:])
    
    plt.figure(6)
    plt.imshow(V, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.savefig('state_value.eps', format='eps', bbox_inches='tight')
    plt.show()
