# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 06:40:07 2023

@author: Hruthik V S
"""

import gym
import numpy as np
from block_transfer_vision_SAC import dVRKBlockVisionEnv

from stable_baselines3 import SAC
 

14
print('yes')


model_class =  SAC

MAX_EP_LEN = 100

env = dVRKBlockVisionEnv(maxsteps=MAX_EP_LEN)

model = model_class.load('models/SAC-1675794854/20000.zip', env=env)

obs = env.reset()
while True:
    action, _states = model.predict(obs)

    obs, rewards, done, info = env.step(action) 
    #print(obs)
    env.render()
    if done:
        print('Episode Done')
        env.reset()