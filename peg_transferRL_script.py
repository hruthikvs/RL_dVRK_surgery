# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 22:36:45 2022

@author: Hruthik V S
"""

 
 
import numpy as np
import time
from zmqRemoteApi import RemoteAPIClient
import sys
MAX_INT = sys.maxsize



import gym
from gym import spaces
# from gym.utils import colorize, seeding

class dVRKCopeliaEnv(gym.Env):
    def __init__(self):
        
        self.num_steps = 100 
               
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')
               
        # getting handles
        self.targetIDR =  self.sim.getObjectHandle("/TargetPSMR")
        self.cylinder1 = self.sim.getObjectHandle("/Cylinder[0]")
        self.cylinder5 = self.sim.getObjectHandle("/Cylinder[5]")
        
        #getting positions
        self.startPos = self.sim.getObjectPosition(self.cylinder1,-1)
        self.endPos = self.sim.getObjectPosition(self.cylinder5,-1)
        
        
        
        print('(dVRKVREP) initialized')

        obs = np.array([np.inf]*3)
        act = np.array([1.]*3)

        self.action_space = spaces.Box(-act,act)
        self.observation_space = spaces.Box(-obs,obs)
        
        
        self.self_observe()
 
    
    def self_observe(self):
        # observe then assign
        targetpos = self.sim.getObjectPosition(self.targetIDR,-1)

        self.observation = np.array([
            targetpos[0], targetpos[1],targetpos[2],
            ]).astype('float32')
        
        
    
    def step(self,actions):
        
        actions = np.clip(actions, -1, 1)
        
        scaling_factor = 0.01
        finalpos = self.observation + actions*scaling_factor
        
        # step
        stat = self.sim.setObjectPosition(self.targetIDR,-1,finalpos.tolist())
        
        
        # observe again
        self.self_observe()

        
        #Calculating reward
        cost = np.linalg.norm(np.array(self.observation) - np.array(self.endPos))
        
        #setting done to True
        
        self.num_steps -= 1
        
    
        if self.num_steps <=0:
            done = True
        else :
            done = False
            
            
        return self.observation, -cost, done, {}

    def render(self):
        pass
    
    def reset(self):
        #stop Simulation    
        self.sim.stopSimulation()
        print('end')
        
        #TODO
        time.sleep(0.1)
        
        
        #start Simulation 
        self.sim.startSimulation()
        print('start')
        
        #set start position
        self.sim.setObjectPosition(self.targetIDR,-1,self.startPos)
        
        
        
        self.num_steps = 100 
        
        self.self_observe()
        return self.observation

     

print(__name__)
if __name__ == '__main__':
    env = dVRKCopeliaEnv()
    done = None
    for k in range(5):
        
        done = False
        print('This is epidode',k)
        observation = env.reset()
        while not done:
          env.render()
          
          action = env.action_space.sample() 
          print(action)
          # your agent here (this takes random actions)
          observation, reward, done, info = env.step(action)
          print(reward)
          
      
    
    
    
    print('simulation ended. leaving in 5 seconds...')
    time.sleep(5)