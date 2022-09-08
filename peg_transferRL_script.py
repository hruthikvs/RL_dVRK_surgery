# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 22:36:45 2022

@author: Hruthik V S
"""

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
    def __init__(self,numsteps=100):
        
        self.distance_threshold = 0.02
        
        
        self.num_steps = numsteps 
               
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')
        self.sim.startSimulation()
        
        # getting handles
        self.toolTip =  self.sim.getObjectHandle("/L3_dx_respondable_TOOL2")
        self.targetIDR =  self.sim.getObjectHandle("/TargetPSMR")
        self.peg = self.sim.getObjectHandle("/Peg")
        
        self.cylinder = []
        for i in range(6):
            cyl_tag = "/Cylinder[{0}]".format(i)
            self.cylinder.append(self.sim.getObjectHandle(cyl_tag))
        
        
        #getting positions
        self.startPos = self.sim.getObjectPosition(self.peg,-1)
        self.endPos = self.sim.getObjectPosition(self.cylinder[5],-1)
        
        #print('TimeStep:',self.sim.getSimulationTimeStep())
        
        
        print('(dVRKVREP) initialized')

        obs = np.array([np.inf]*3)
        act = np.array([1.]*3)

        self.action_space = spaces.Box(-act,act)
        self.observation_space = spaces.Dict(
            dict(
                observation= spaces.Box(-obs,obs, dtype=np.float32),
                desired_goal= spaces.Box(-obs,obs, dtype=np.float32),
                achieved_goal= spaces.Box(-obs,obs, dtype=np.float32),
            )
        )
         
        
        self.self_observe()
 
    
    
    
    
    
    def self_observe(self):
        """
        Helper to create the observation.
        :return: The current observation.
        """
        targetpos = self.sim.getObjectPosition(self.targetIDR,-1)
        
        
        
        current_state = np.array([
            targetpos[0], targetpos[1],targetpos[2],
            ]).astype('float32')
        
        self.observation = OrderedDict([ ("observation",  current_state), ("achieved_goal", current_state),
                ("desired_goal", np.array(self.endPos))])

    
        
        
    
    def step(self,actions):
        
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
        scaling_factor = 0.01
        
        
        finalpos = self.observation['observation'] + actions*scaling_factor
        
        #print('final pos=',finalpos)
        print('action=',actions)
        
        # step
        stat = self.sim.setObjectPosition(self.targetIDR,-1,finalpos.tolist())
        
        
        # observe again
        self.self_observe()

        
        #Calculating reward
        dist_goal = np.linalg.norm(np.array(self.observation['observation']) - np.array(self.endPos))
        
        
        reward = -1 if dist_goal>self.distance_threshold else 0
        
        #setting done to True
        self.num_steps -= 1       
        print(self.num_steps)
        if self.num_steps <=0 or dist_goal<self.distance_threshold:
            print('yes')
            done = True
            if dist_goal<self.distance_threshold:
                print('Goal Achieved!!')
                #Random Sampling of Goal
                endPosLst = []
                for i in range(6):
                    endPosLst.append(self.sim.getObjectPosition(self.cylinder[i],-1))
               
        else :
            done = False
             
        if not (self.observation['observation'][0]>-3.3 and self.observation['observation'][0]<-3.1 and self.observation['observation'][1]>-1 and self.observation['observation'][1]<1 and self.observation['observation'][0]<-2 and self.observation['observation'][2]>1.35 and self.observation['observation'][2]<1.6):
            reward = -1
            #print(self.num_steps)
            #done=True
          
            
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
        
        
        
        self.num_steps = 100 
        
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