# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 22:45:23 2022

@author: Hruthik V S
"""

import gym
import numpy as np
from peg_picknplace_RL import dVRKCopeliaPicknPlaceEnv
import os
from stable_baselines3 import DDPG,HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import time
from tensorflow.keras.callbacks import TensorBoard
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

 

model_class =  DDPG

MAX_EP_LEN = 200

models_dir = f"models/DDPG_HER-PnP-{int(time.time())}"
logdir = f"logs/DDPG_HER-PnP-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = dVRKCopeliaPicknPlaceEnv(maxsteps=MAX_EP_LEN)
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = MAX_EP_LEN


# The noise objects for DDPG
print(env.action_space)
#n_actions = env.action_space.shape[-1]
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))




model = DDPG("MultiInputPolicy",env, buffer_size=1000000, replay_buffer_class=HerReplayBuffer,
             replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,),
             verbose=1, 
             tensorboard_log= logdir)


 


#epochs 
TIMESTEPS = 10000
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, log_interval=1,tb_log_name='DDPG_HER')    #model reset set to false: prevent model reset for every learn
    model.save(f"{models_dir}/{TIMESTEPS*i}")
env = model.get_env()



del model # remove to demonstrate saving and loading

model = DDPG.load(f"{models_dir}/{TIMESTEPS*50}")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    
    env.render()
    if done:
        print('Episode Done')
        env.reset()
