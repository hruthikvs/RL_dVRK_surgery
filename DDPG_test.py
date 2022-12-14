import gym
import numpy as np
from peg_transferRL_script import dVRKCopeliaEnv

from stable_baselines3 import DDPG,HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise



print('yes')


model_class =  DDPG

MAX_EP_LEN = 100

env = dVRKCopeliaEnv(maxsteps=MAX_EP_LEN)

model = model_class.load('models/DDPG_HER-1666089129/100000.zip', env=env)

obs = env.reset()
while True:
    action, _states = model.predict(obs)

    obs, rewards, done, info = env.step(action) 
    #print(obs)
    env.render()
    if done:
        print('Episode Done')
        env.reset()