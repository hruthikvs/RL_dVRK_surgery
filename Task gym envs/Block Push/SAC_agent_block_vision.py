# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:35:57 2023

@author: Hruthik V S
"""
 

import gym
import numpy as np
from block_transfer_vision_SAC import dVRKBlockVisionEnv
import os
from stable_baselines3 import SAC
import time
from tensorflow.keras.callbacks import TensorBoard
 
 

model_class =  SAC

MAX_EP_LEN = 100

models_dir = f"models/SAC-{int(time.time())}"
logdir = f"logs/SAC-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = dVRKBlockVisionEnv(maxsteps=MAX_EP_LEN)


# Time limit for the episodes
max_episode_length = MAX_EP_LEN





model = model_class("CnnPolicy", env, buffer_size=50000, verbose=1, tensorboard_log= logdir)


 


#epochs 
TIMESTEPS = 10000
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, log_interval=1,tb_log_name='DDPG_HER')    #model reset set to false: prevent model reset for every learn
    model.save(f"{models_dir}/{TIMESTEPS*i}")
env = model.get_env()



del model # remove to demonstrate saving and loading

model = model_class.load(f"{models_dir}/{TIMESTEPS*50}")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    
    env.render()
    if done:
        print('Episode Done')
        env.reset()
