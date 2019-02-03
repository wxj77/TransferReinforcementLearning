#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Created on Fri Feb  1 23:31:42 2019

@author: wei

compare time that takes to do N simulation without multi-threading and with multi-threading
"""

import sys
import numpy as np
import pickle
import gym
from gym import wrappers

import time
import matplotlib.pyplot as plt
import threading
import queue
import pongSimpleFunc as psfunc
from multiprocessing import Process, Pool, Queue

TotalSims=20
SepSims=4
if (len(sys.argv) >=2 ):
    TotalSims = sys.argv[1]
if (len(sys.argv) >=3 ):
    SepSims = sys.argv[2]

resume = False # resume training from previous checkpoint (from save.p  file)?
render = False # render video output?

# model initialization
H = 200 # number of hidden layer neurons
#batch_size = 10 # used to perform a RMS prop param update every batch_size steps
#learning_rate = 1e-3 # learning rate used in RMS prop
D = 75 * 80 # input dimensionality: 75x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization - Shape will be H x D
  model['W2'] = np.random.randn(H) / np.sqrt(H) # Shape will be H

#--- Without MultiThred    
env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

starttime=time.time()
q= queue.Queue()
xs, drs, dlogps, hs = psfunc.worker(env, q, model, TotalSims, '7580')
endtime=time.time()

print ("No threading. %d simlations cost %.3f s"%(TotalSims, (endtime-starttime)))

#--- With MultiThred

qlist=[] # to capture results
envlist=[] # list of simulation agents
for i in range(int (SepSims) ):
    q= queue.Queue()
    qlist.append(q)
    env = gym.make("Pong-v0")
    envlist.append(env)

starttime=time.time() 
xs, drs, dlogps, hs = [],[],[],[]

tlist = []
for i in range(int (SepSims) ):    
    w = threading.Thread(target=psfunc.worker, args=(envlist[i],qlist[i], model.copy(), int (TotalSims/SepSims), '7580') )
    tlist.append(w)
    w.start()

for i in range(int (SepSims) ):
    tlist[i].join() 

for i in range(int (SepSims) ):
    xs0, drs0, dlogps0, hs0  =  qlist[i].get()  
    xs = xs+xs0
    drs = drs+drs0
    dlogps = dlogps+dlogps0
    hs = hs + hs0


endtime=time.time()

print ("With multi threading. %d simlations cost %.3f s"%(TotalSims, (endtime-starttime)))


#--- With MultiProcess

qlist=[] # to capture results
envlist=[] # list of simulation agents
for i in range(int (SepSims) ):
    q= Queue.Queue()
    qlist.append(q)
    env = gym.make("Pong-v0")
    envlist.append(env)

starttime=time.time() 
xs, drs, dlogps, hs = [],[],[],[]

tlist = []
for i in range(int (SepSims) ):    
    w = Process(target=psfunc.worker, args=(envlist[i], qlist[i], model.copy(), int (TotalSims/SepSims), '7580') )
    tlist.append(w)
    w.start()
    
for i in range(int (SepSims) ):
    tlist[i].join() 

for i in range(int (SepSims) ):
    xs0, drs0, dlogps0, hs0  =  qlist[i].get()  
    xs = xs+xs0
    drs = drs+drs0
    dlogps = dlogps+dlogps0
    hs = hs + hs0


endtime=time.time()

print ("With multi process. %d simlations cost %.3f s"%(TotalSims, (endtime-starttime)))