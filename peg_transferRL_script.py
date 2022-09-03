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
    def __init__(self,numsteps=100):
        
        self.distance_threshold = 0.01
        
        
        self.num_steps = numsteps 
               
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')
               
        # getting handles
        self.toolTip =  self.sim.getObjectHandle("/L3_dx_respondable_TOOL2")
        self.targetIDR =  self.sim.getObjectHandle("/TargetPSMR")
        self.peg = self.sim.getObjectHandle("/Peg")
        self.cylinder1 = self.sim.getObjectHandle("/Cylinder[0]")
        self.cylinder5 = self.sim.getObjectHandle("/Cylinder[5]")
        
        #getting positions
        self.startPos = self.sim.getObjectPosition(self.peg,-1)
        self.endPos = self.sim.getObjectPosition(self.cylinder5,-1)
        
        print('TimeStep:',self.sim.getSimulationTimeStep())
        
        
        print('(dVRKVREP) initialized')

        obs = np.array([np.inf]*3)
        act = np.array([1.]*3)

        self.action_space = spaces.Box(-act,act)
        self.observation_space = spaces.Box(-obs,obs)
        
        
        self.self_observe()
 
    
    
    
    
    
    def self_observe(self) -> Dict[str, Union[int, np.ndarray]]:
        """
        Helper to create the observation.
        :return: The current observation.
        """
        targetpos = self.sim.getObjectPosition(self.targetIDR,-1)
        
        
        
        current_state = np.array([
            targetpos[0], targetpos[1],targetpos[2],
            ]).astype('float32')
        
        self.observation = OrderedDict(
            [
                ("observation", , current_state
                ("achieved_goal", self.convert_if_needed(self.state.copy())),
                ("desired_goal", self.convert_if_needed(self.desired_goal.copy())),
            ]
        )

    
        
        
    
    def step(self,actions):
        
        actions = np.clip(actions, self.action_space.low, self.action_space.low)
        
        scaling_factor = 0.01
        
        
        finalpos = self.observation + actions*scaling_factor
        
        # step
        stat = self.sim.setObjectPosition(self.targetIDR,-1,finalpos.tolist())
        
        
        # observe again
        self.self_observe()

        
        #Calculating reward
        dist_goal = np.linalg.norm(np.array(self.observation) - np.array(self.endPos))
        reward = -1 if dist_goal>self.distance_threshold else 0
        
        #setting done to True
        self.num_steps -= 1       
    
        if self.num_steps <=0 or dist_goal<1e-3:
            reward = 10
            done = True
        else :
            done = False
             
        if not (self.observation[0]>-3.3 and self.observation[0]<-3.1 and self.observation[1]>-1 and self.observation[1]<1 and self.observation[0]<-2 and self.observation[2]>1.35 and self.observation[2]<1.6):
            reward = -100
            print(self.num_steps)
            done=True
          
            
        return self.observation, reward, done, {}

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
          #env.render()
          
          action = env.action_space.sample() 
          #print(action)
          # your agent here (this takes random actions)
          observation, reward, done, info = env.step(action)
          print(reward)
          
      
    
    
    
    print('simulation ended. leaving in 5 seconds...')
    time.sleep(5)