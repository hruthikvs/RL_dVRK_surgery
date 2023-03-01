# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:35:57 2023

@author: Hruthik V S
"""
 
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:41:18 2023

@author: Hruthik V S
"""

import gym
import numpy as np
from block_randomiseStart_HER import dVRKBlockVisionEnv
import os
from stable_baselines3 import SAC,HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import time
from tensorflow.keras.callbacks import TensorBoard
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

 

model_class =  SAC

#Set TRUE if pretrained model exist
pretrained = False
pretrained_model_dir = "models/SAC_HER-1677651951/100.zip"
pretrained_log_dir = "models/SAC_HER-1677651951/SAC_HER_0"

MAX_EP_LEN = 200

models_dir = f"models/SAC_HER-{int(time.time())}"
logdir = f"logs/SAC_HER-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = dVRKBlockVisionEnv(maxsteps=MAX_EP_LEN)
 
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = MAX_EP_LEN


# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))


if not pretrained:
    
    model = model_class("MultiInputPolicy",env,learning_rate=0.0003,buffer_size=50000, verbose=1,
                        learning_starts=max_episode_length, batch_size=256, tau=0.005, 
                        gamma=0.99, train_freq=(5,'steps'), gradient_steps=1,
                        replay_buffer_class=HerReplayBuffer, 
                        replay_buffer_kwargs=dict( n_sampled_goal=4,
                                        goal_selection_strategy=goal_selection_strategy,
                                        online_sampling=online_sampling,
                                        max_episode_length=max_episode_length,) 
                                        ,action_noise=action_noise, 
                                              tensorboard_log= logdir)

#if pretrained==True, Load existing model
else:
    model = model_class.load(pretrained_model_dir, env=env, tensorboard_log=pretrained_log_dir)
    model.set_env(env)
    
#epochs 
TIMESTEPS = 100
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, log_interval=1,tb_log_name='SAC_HER')    #model reset set to false: prevent model reset for every learn
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
