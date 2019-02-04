#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Created on Sat Feb  2 00:37:50 2019

@author: wei
"""


import sys
import numpy as np
import pickle
import gym
from gym import wrappers

import time
#import matplotlib.pyplot as plt
import threading
import queue
import pongSimpleFunc as psfunc

import os
workdir = "./pongRot90down14/pongRotSol/"
try:
    os.stat(workdir)
except:
    os.makedirs(workdir)   


TotalSims=20 # batch size : How many simulations to accumulate before update the neuron network
TotalBatch=10000 # number of batch to calculate
option = '8075' # resize the figure 75X80 options are '4040', '7580','8080'
gameoption = 'rot90down14' # gameoptions are 'rgb_array', 'rot90', 'rot90down14', 'human'


if (len(sys.argv) >=2 ):
    TotalSims = sys.argv[1]
if (len(sys.argv) >=3 ):
    TotalBatch = sys.argv[2]

resume = True # resume training from known sol
resumecheck = False # resume training from previous checkpoint (from save.p  file)?
render = False # render video output?
rot= True # Add a rotation matrix at the last layer

# model initialization
H = 200 # number of hidden layer neurons
#batch_size = 10 # used to perform a RMS prop param update every batch_size steps
learning_rate = 1e-3 # learning rate used in RMS prop
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

XLIM, YLIM = int(option[0:2]) , int(option[2:4]) 
D = XLIM* YLIM # input dimensionality: 40x40 grid

R = psfunc.rotate_clockwise(np.zeros( (XLIM, YLIM) )) 

if resume:  
    if resumecheck:
        model = pickle.load(open(workdir+'/save.protnt', 'rb'))
        timelist =         np.load( workdir+"/timelist.npy")
        batchlist =      np.load( workdir+"/batchlist.npy")
        rewardlist =      np.load( workdir+"/rewardlist.npy")
    else:
        model = pickle.load(open('save.p', 'rb'))
        print ("Create new files to record time/batch/reward of each training.")
        timelist = np.zeros((0))
        batchlist = np.zeros((0))
        rewardlist = np.zeros((0))        
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization - Shape will be H x D
    model['W2'] = np.random.randn(H) / np.sqrt(H) # Shape will be H
    print ("Create new files to record time/batch/reward of each training.")
    timelist = np.zeros((0))
    batchlist = np.zeros((0))
    rewardlist = np.zeros((0))   

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory


env = gym.make("Pong-v0")
env._set_obs_type(gameoption )
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
offsettime=0
offsetepisode=0
if len(timelist)>0:
    offsettime = timelist[-1]
    offsetepisode = batchlist[-1]

q= queue.Queue()
starttime=time.time()
#timelist=[]
#batchlist=[]
#rewardlist=[]

batchnum = 0
while batchnum < TotalBatch:
    batchnum+=1
    if render: env.render()

    xs, drs, dlogps, hs = psfunc.worker(env, q, model, NSims=TotalSims, option=option, rot=rot)
    while not (q.empty() ):
        q.get()

    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    
    epx = np.vstack(xs)
    epx0 = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    
    if rot:    
        epx = np.dot(R, epx.transpose()).transpose()

    # compute the discounted reward backwards through time
    discounted_epr = psfunc.discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (Policy Grad magic happens right here.)
    grad = psfunc.policy_backward(model, eph, epx, epdlogp)
    
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
#    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
#    print ('resetting env. episode reward total was %f. running mean: %f'%(reward_sum, running_reward) )
    
    endtime=time.time()
    timelist = np.concatenate( (timelist, np.array([endtime-starttime+offsettime])) )
    batchlist = np.concatenate( ( batchlist, np.array([TotalSims*episode_number+offsetepisode])) )
    rewardlist = np.concatenate( ( rewardlist, np.array([np.array(drs).sum()])) )
    
 #   print (timelist)
 #   print (batchlist)
    print (rewardlist)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory
    
    if episode_number % 10 == 0: #save results
        pickle.dump(model, open(workdir+'/save.protnt', 'wb'))
        np.save( workdir+"/timelist", np.array(timelist))
        np.save( workdir+"/batchlist", np.array(batchlist))
        np.save( workdir+"/rewardlist", np.array(rewardlist))
        
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None


#--- Without MultiThred    
env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
