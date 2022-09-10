# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 22:36:45 2022

@author: Hruthik V S
"""
import random
from typing import Any, Dict, Union 
from collections import OrderedDict
import numpy as np
import time
from zmqRemoteApi import RemoteAPIClient
import sys
MAX_INT = sys.maxsize



import gym
import gym.spaces
print(gym.__version__)
from gym import spaces
# from gym.utils import colorize, seeding

class dVRKCopeliaEnv(gym.GoalEnv):
    def __init__(self,maxsteps=100):
        
        self.distance_threshold = 0.02
        
        
        self.max_steps = maxsteps 
        self.num_steps = 0
               
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')
        self.sim.startSimulation()
        
        # getting handles
        self.toolTip =  self.sim.getObjectHandle("/L3_dx_respondable_TOOL2")
        self.targetIDR =  self.sim.getObjectHandle("/TargetPSMR")
        self.peg = self.sim.getObjectHandle("/Peg")
        
        #getting positions
        self.cylinder = []
        self.endPos = []
        for i in range(6):
            cyl_tag = "/Cylinder[{0}]".format(i)
            self.cylinder.append(self.sim.getObjectHandle(cyl_tag))
            self.endPos.append(self.sim.getObjectPosition(self.cylinder[i],-1))
        
        self.startPos = self.sim.getObjectPosition(self.peg,-1)
        self.currentgoal =  random.sample(self.endPos, 1)
        #print('TimeStep:',self.sim.getSimulationTimeStep())
        
        
        print('(dVRKVREP) initialized')

        obs = np.inf
        act = 1

        self.action_space = spaces.Box(-act,act, shape = (3,))
        self.observation_space = spaces.Dict(
            dict(
                observation= spaces.Box(-obs,obs,shape=(6,), dtype=np.float32),
                desired_goal= spaces.Box(-obs,obs,shape=(3,), dtype=np.float32),
                achieved_goal= spaces.Box(-obs,obs,shape=(3,), dtype=np.float32),
            )
        )
         
        self.velocity = np.zeros([3])
        self.self_observe()
 
    
    
    
    
    
    def self_observe(self):
        """
        Helper to create the observation.
        :return: The current observation.
        """
        
        
        targetpos = self.sim.getObjectPosition(self.targetIDR,-1)
        
        
        
        current_pos = np.array([
            targetpos[0], targetpos[1],targetpos[2],
            ]).astype('float32')
        
        current_state = np.append(current_pos,self.velocity)
        
        self.observation = OrderedDict([ ("observation",  current_state), ("achieved_goal", current_state[:3]),
                ("desired_goal", np.array(self.currentgoal))])
        
        

    
        
        
    
    def step(self,actions):
        
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
        scaling_factor = 0.005
        
        #Calculating velocity
        dt = self.sim.getSimulationTimeStep()
        self.velocity = np.array( (actions*scaling_factor) / dt )
        
    
        finalpos = self.observation['observation'][:3] + actions*scaling_factor
        
        #print('final pos=',finalpos)
        print('action=',actions)
        
        # step
        stat = self.sim.setObjectPosition(self.targetIDR,-1,finalpos.tolist())
        
        
        # observe again
        self.self_observe()

        #Calculating reward
        dist_goal = np.linalg.norm(np.array(finalpos) - np.array(self.currentgoal))
        
        
        reward = -1 if dist_goal>self.distance_threshold else 0
        
        #setting done to True
        self.num_steps += 1       
        print(self.num_steps)
        if self.num_steps > self.max_steps or dist_goal<self.distance_threshold:
            print('yes')
            done = True
            if dist_goal<self.distance_threshold:
                print('Goal Achieved!!')
                #Random Sampling of Goal
                self.currentgoal = random.sample(self.endPos,1)
               
        else :
            done = False
             
        if not (self.observation['observation'][0]>-3.3 and self.observation['observation'][0]<-3.1 and self.observation['observation'][1]>-1 and self.observation['observation'][1]<1 and self.observation['observation'][2]>1.35 and self.observation['observation'][2]<1.6):
            reward = self.num_steps - 100 - 1
            #print(self.num_steps)
            done=True
          
        print(reward) 
        return self.observation, reward, done, {}

    def render(self):
        pass
    
    def reset(self):
        #stop Simulation    
        self.sim.stopSimulation()
        print('end')
        
        #TODO
        time.sleep(0.2)
        
        
        #start Simulation 
        self.sim.startSimulation()
        print('start')
        
        #set start position
        self.sim.setObjectPosition(self.targetIDR,-1,self.startPos)
        
        
        
        self.num_steps = 0  
        
        self.self_observe()
        return self.observation

     

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = np.linalg.norm(np.array(achieved_goal) - np.array(desired_goal))
        return -np.array(d > self.distance_threshold, dtype=np.float64)
    







print(__name__)
if __name__ == '__main__':
    env = dVRKCopeliaEnv()
    
    done = None
    for k in range(5):
        
        done = False
        print('This is epidode',k)
        observation = env.reset()
        while not done:
          #env.render()
          
          action = env.action_space.sample() 
          #print(action)
          # your agent here (this takes random actions)
          observation, reward, done, info = env.step(action)
          print(reward)
          
      
    
    
    
    print('simulation ended. leaving in 5 seconds...')
    time.sleep(5)