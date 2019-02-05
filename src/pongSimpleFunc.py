#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

""" Majority of this code was copied directly from Andrej Karpathy's gist:
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5 """

"""
Created on Fri Feb    1 23:17:17 2019

@author: wei

functions used to get a simple solution for the atari games, which include
    image process, 
    discount reward, 
    forward policy, 
    backward policy
"""

# pong game parameters
# useful pixels: 34:194,:,:
# bar (left):,16:20,: length 16 = 78-62
# bar (right):,140:144,: length 16 = 61-45
# ball : length 4X2 = (99-95)X(62-60) or (97-93)X(60-58)

# break game parameters
# useful pixels: 34:194,:,:
# wall (left):,0:8,: length 8
# bar (right):,152:160,: length 8 = 
# bar bottom: 189:193 width 4 length 16 = 115-99
# ball : length 4X2 = (99-95)X(62-60) or (97-93)X(60-58)
import cv2


import numpy as np
import pickle
import gym
#from gym import wrappers
import queue
#import matplotlib.pyplot as plt

# hyperparameters to tune
#H = 200 # number of hidden layer neurons
#batch_size = 10 # used to perform a RMS prop param update every batch_size steps
#learning_rate = 1e-3 # learning rate used in RMS prop
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

# Config flags - video output and res
#resume = True # resume training from previous checkpoint (from save.p  file)?
#render = True # render video output?

# model initialization
#D = 75 * 80 # input dimensionality: 75x80 grid

def SizeImg(observation):
    "resize a figure from pong game and make it look like from the breakout game"
    obs = np.zeros_like(observation)
    obs[63:207, 8:152,0] = cv2.resize(np.rot90(observation[34:194,:,0],k=3),(144,144))
    return obs

#observation2 = SizeImg(pongbr)
#plt.imshow(observation2[:,:,0])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
    I = I[35:185] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
    I = I[::2,::2,0] # downsample by factor of 2.
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
    return I.astype(np.float).ravel() # ravel flattens an array and collapses it into a column vector

def prepro7580(I):
    """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
    I = I[35:185] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
    I = I[::2,::2,0] # downsample by factor of 2.
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
    return I.astype(np.float).ravel() # ravel flattens an array and collapses it into a column vector

def prepro8075(I):
    """ prepro 210x160x3 uint8 frame into 6000 (80x75) 1D float vector """
    I = I[34:194,1:151,:] # crop - remove 34px from start & 16px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
    I = I[::2,::2,0] # downsample by factor of 2.
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
    return I.astype(np.float).ravel() # ravel flattens an array and collapses it into a column vector

def prepro8075x(I):
    """ prepro 210x160x3 uint8 frame into 6000 (80x75) 1D float vector """
    I = I[50:210,1:151,:] # crop - remove 34px from start & 16px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
    I = I[::2,::2,0] # downsample by factor of 2.
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
    return I.astype(np.float).ravel() # ravel flattens an array and collapses it into a column vector


def prepro8080(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[34:194] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
    I = I[::2,::2,0] # downsample by factor of 2.
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
    return I.astype(np.float).ravel() # ravel flattens an array and collapses it into a column vector


def preprocess_img(I):
    """ prepro 210x160x3 uint8 frame into 1600 (40x40) 1D float vector """
    # downsample by factor of 4.
    I = I[34:194] 
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I = I[::4,::4,0]+I[::4,1::4,0]+I[::4,2::4,0]+I[::4,3::4,0]+ \
    I[1::4,::4,0]+I[1::4,1::4,0]+I[1::4,2::4,0]+I[1::4,3::4,0]+ \
    I[2::4,::4,0]+I[2::4,1::4,0]+I[2::4,2::4,0]+I[2::4,3::4,0]+ \
    I[3::4,::4,0]+I[3::4,1::4,0]+I[3::4,2::4,0]+I[3::4,3::4,0] 
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
    return I.astype(np.float).ravel() # ravel flattens an array and collapses it into a column vector


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    """ this function discounts from the action closest to the end of the completed game backwards
    so that the most recent action has a greater weight """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)): # xrange is no longer supported in Python 3
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(model, x):
    """This is a manual implementation of a forward prop"""
    h = np.dot(model['W1'], x) # (H x D) . (D x 1) = (H x 1) (200 x 1)
    h[h<0] = 0 # ReLU introduces non-linearity
    logp = np.dot(model['W2'], h) # This is a logits function and outputs a decimal.     (1 x H) . (H x 1) = 1 (scalar)
    p = sigmoid(logp)    # squashes output to    between 0 & 1 range
    return p, h # return probability of taking action 2 (UP), and hidden state

def policy_backward(model, eph, epx, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    """ Manual implementation of a backward prop"""
    """ It takes an array of the hidden states that corresponds to all the images that were
    fed to the NN (for the entire episode, so a bunch of games) and their corresponding logp"""
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}


def rotate_clockwise(W):
    """
    for a matrix (W) with a certain shape
    return a matrix (R) that (R* W.ravel()).reshape(np.flip(W.shape)) 
    
    Input:
        W: shape (M, N)    
    Output:
        R: shape (M*N, M*N)
    """
    xlim = W.shape[0]
    ylim = W.shape[1]
    N=xlim*ylim
    I=np.zeros((N,N))
    for xx in np.arange(xlim):
        for yy in np.arange(ylim):
            index1 = xx*ylim+yy
            index2 = yy*xlim+xx
 #           I[index1][index2]=1
            I[index2][index1]=1
    return I

#R = rotate_clockwise(np.zeros( (75,80) )) 
 

def worker(env, q, model, NSims=1, option="4040", rot=False, return_hidden=True):
    """
    Return states and rewards from a stack of openai simulations.
    
    Input: 
        env: a openai environment
        q: queue object to catch the result
        NSims: number of simulations
        option: XXYY -- figure size XXxYY  
        rot: rotate 90 degree or not rotate
        return_hidder: return hidden states
    Output
        None
    """
    TotalSims = NSims
    observation = env.reset()
    prev_x = None # used in computing the difference frame
    xs,hs,dlogps,drs = [],[],[],[]
    #running_reward = None
    reward_sum = 0
    episode_number = 0
    XLIM, YLIM = int(option[0:2]) , int(option[2:4]) 
    if (rot==True):
        R = rotate_clockwise(np.zeros( (XLIM, YLIM) )) 
    while True:
    # preprocess the observation, set input to network to be difference image
        if ( option=="7580" ):
            cur_x = prepro7580(observation)
        elif ( option=="8075" ):
            cur_x = prepro8075(observation)
        elif ( option=="8080" ):
            cur_x = prepro8080(observation)
        elif ( option=="4040" ):
            cur_x = preprocess_img(observation)
        else:
            print ("error in image resizing. options are '4040', '8080', '7580', '8075' ")
            break
        
    # we take the difference in the pixel input, since this is more likely to account for interesting information
    # e.g. motion
        x = cur_x - prev_x if prev_x is not None else np.zeros_like(cur_x)
        prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
        if (rot==False):
            aprob, h = policy_forward(model, x)
        else:
            aprob, h = policy_forward(model, np.dot(R,x))
    # The following step is randomly choosing a number which is the basis of making an action decision
    # If the random number is less than the probability of UP output from our neural network given the image
    # then go down.    The randomness introduces 'exploration' of the Agent
        action = 2 if np.random.uniform() < aprob else 3 # roll the dice! 2 is UP, 3 is DOWN, 0 is stay the same

    # record various intermediates (needed later for backprop).
    # This code would have otherwise been handled by a NN library
        xs.append(x) # observation
        hs.append(h) # hidden state
#        h0s.append(h0) # hidden state
        y = 1 if action == 2 else 0 # a "fake label" - this is the label that we're passing to the neural network
    # to fake labels for supervised learning. It's fake because it is generated algorithmically, and not based
    # on a ground truth, as is typically the case for Supervised learning

        dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done: # an episode finished
            episode_number += 1
            print ("%d simulation finished! Left %d simuations to do."%(TotalSims - NSims+1, NSims-1))
            observation = env.reset() 
            NSims-=1
            if NSims<=0:
                break
            
    if return_hidden:
        q.put([xs, drs, dlogps, hs,])
        return [xs, drs, dlogps, hs,]
    else:
        q.put([xs, drs,])
        return [xs, drs,]




def worker_breakout(env, q, model, NSims=1, option="4040", rot=False, return_hidden=True):
    """
    Return states and rewards from a stack of openai simulations.
    
    Input: 
        env: a openai environment
        q: queue object to catch the result
        NSims: number of simulations
        option: XXYY -- figure size XXxYY  
        rot: rotate 90 degree or not rotate
        return_hidder: return hidden states
    Output
        None
    """
    lives = 5 # store how many lives left
    TotalSims = NSims
    observation = env.reset()
    prev_x = None # used in computing the difference frame
    xs,hs,dlogps,drs = [],[],[],[]
    #running_reward = None
    reward_sum = 0
    episode_number = 0
    XLIM, YLIM = int(option[0:2]) , int(option[2:4]) 
    if (rot==True):
        R = rotate_clockwise(np.zeros( (XLIM, YLIM) )) 
    observation, reward, done, info = env.step(1) # Fire the game
    finish = True 
    while True:
#        print ("lives: ", lives)
#        print ("info['ale.lives']", info['ale.lives'])
        if (info['ale.lives']!=lives):
            print ("lives: ", lives)
            print ("info['ale.lives']", info['ale.lives'])
            observation, reward, done, info = env.step(1) # Fire the game  
            lives = info['ale.lives']
            print ("Restart game. Left %d lives"%(info['ale.lives']))
    # preprocess the observation, set input to network to be difference image
        if ( option=="7580" ):
            cur_x = prepro7580(observation)
        elif ( option=="8075" ):
            cur_x = prepro8075(observation)
        elif ( option=="8075x" ):
            cur_x = prepro8075x(observation)
        elif ( option=="8080" ):
            cur_x = prepro8080(observation)
        elif ( option=="4040" ):
            cur_x = preprocess_img(observation)
        else:
            print ("error in image resizing. options are '4040', '8080', '7580', '8075' ")
            break
        
    # we take the difference in the pixel input, since this is more likely to account for interesting information
    # e.g. motion
        x = cur_x - prev_x if prev_x is not None else np.zeros_like(cur_x)
        prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
        if (rot==False):
            aprob, h = policy_forward(model, x)
        else:
            aprob, h = policy_forward(model, np.dot(R,x))
    # The following step is randomly choosing a number which is the basis of making an action decision
    # If the random number is less than the probability of UP output from our neural network given the image
    # then go down.    The randomness introduces 'exploration' of the Agent
        action = 3 if np.random.uniform() < aprob else 2 # 3 is Left, 2 is Right, 0 is stay the same, 1 is start the game
#        print ("action", action)
    # record various intermediates (needed later for backprop).
    # This code would have otherwise been handled by a NN library
        xs.append(x) # observation
        hs.append(h) # hidden state
#        h0s.append(h0) # hidden state
        y = 1 if action == 3 else 0 # a "fake label" - this is the label that we're passing to the neural network
    # to fake labels for supervised learning. It's fake because it is generated algorithmically, and not based
    # on a ground truth, as is typically the case for Supervised learning

        dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done: # an episode finished
            episode_number += 1
            print ("%d simulation finished! Left %d simuations to do."%(TotalSims - NSims+1, NSims-1))
            observation = env.reset() 
            NSims-=1
            finish = finish and (info['ale.lives'] ==0 )
            print (finish)
            if NSims<=0:
                break
            
    if return_hidden:
        q.put([xs, drs, dlogps, hs, finish])
        return [xs, drs, dlogps, hs, finish]
    else:
        q.put([xs, drs,])
        return [xs, drs,]   