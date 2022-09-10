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
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = MAX_EP_LEN


# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))




model = DDPG("MultiInputPolicy",env, buffer_size=100000, replay_buffer_class=HerReplayBuffer,
             replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
),
             verbose=1)


print(model.set_env(env))

model.learn(total_timesteps=50000, log_interval=1)
model.save("ddpg_dVRK")
env = model.get_env()


del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_dVRK")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    
    env.render()
    if done:
        print('Episode Done')
        env.reset()
