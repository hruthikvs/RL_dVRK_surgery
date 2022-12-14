# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:45:19 2022

@author: Hruthik V S
"""

import PIL.Image as Image
import io
import base64
import cv2
from matplotlib import pyplot as plt


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
from gym import spaces 

class dVRKCopeliaVisionEnv(gym.GoalEnv):
    def __init__(self,maxsteps=100):
        
        self.distance_threshold = 0.02
        
        
        self.max_steps = maxsteps 
        self.num_steps = 0
               
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')
        self.sim.intparam_speedmodifier
        self.sim.startSimulation()
        
        # getting handles
        self.toolTip =  self.sim.getObjectHandle("/L3_dx_respondable_TOOL2")
        self.targetIDR =  self.sim.getObjectHandle("/TargetPSMR")
        self.peg = self.sim.getObjectHandle("/Peg")
        self.goalSphere = self.sim.getObjectHandle("/Goal")
        
        #camera handle
        self.camera_side = self.sim.getObjectHandle("/Camera[1]")
        self.camera_top = self.sim.getObjectHandle("/Vision_sensor_left")
        
        
        #getting positions
        self.cylinder = []
        self.endPos = []
        
        for i in range(0,9):
            cyl_tag = "/Cylinder[{0}]".format(i)
            
            self.cylinder.append(self.sim.getObjectHandle(cyl_tag))
            
            goal_pos = np.array(self.sim.getObjectPosition(self.cylinder[i],-1)) + np.array([0,0,0.02])
            self.endPos.append(goal_pos.tolist())
        #removing start position from  sampling goal array
        self.endPos.pop(0)
        
        self.startPos = self.sim.getObjectPosition(self.peg,-1)
        
        self.currentgoal =  random.sample(self.endPos, 1)[0]
        #setting goal sphere position
        self.sim.setObjectPosition(self.goalSphere,-1,self.currentgoal  )
        
        #self.currentgoal = self.endPos[4]
        #print('TimeStep:',self.sim.getSimulationTimeStep())
        
        
        print('(dVRKVREP) initialized')

        obs = np.inf
        act = 1
        
        image , b = self.sim.getVisionSensorImg(self.camera_top,0)
        img = Image.frombytes("RGB", (250, 250), image)
        img = np.array(img)
        
        
        
        
        self.action_space = spaces.Box(-act,act, shape = (3,))
        self.observation_space = spaces.Dict(
            dict(
                observation =  spaces.Box(0,255,shape=(img.shape[2],img.shape[0],img.shape[1]), dtype=np.uint8),
                desired_goal= spaces.Box(-obs,obs,shape=(3,), dtype=np.float32),
                achieved_goal= spaces.Box(-obs,obs,shape=(3,), dtype=np.float32),
            )
        )
        print(np.shape(self.observation_space['observation'])) 
         
        self.velocity = np.zeros([3])
        self.self_observe()
        
        self.position = self.sim.getObjectPosition(self.targetIDR,-1)
 
    
    
    
    
    
    def self_observe(self):
        """
        Helper to create the observation.
        :return: The current observation.
        """
        #Observe Image at each Timestep
        image , b= self.sim.getVisionSensorImg(self.camera_top,0)
        img = Image.frombytes("RGB", (250, 250), image)
        img = np.array(img)
        cv2.imshow('image',img)
        cv2.waitKey(1)
        #reshaping Image
        img = np.moveaxis(img, -1, 0)
        
        targetpos = self.sim.getObjectPosition(self.targetIDR,-1)
        
        
        
        current_pos = np.array([
            targetpos[0], targetpos[1],targetpos[2],
            ]).astype('float32')
        
        current_state = np.append(current_pos,self.velocity)
         
        self.observation = OrderedDict([ ("observation", img ), ("achieved_goal", current_state[:3]),
                ("desired_goal", np.array(self.currentgoal))])
        
        
        
        

    
        
        
    
    def step(self,actions):
        
        
        
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
        scaling_factor = 0.005
        
        #Calculating velocity
        dt = self.sim.getSimulationTimeStep()
        self.velocity = np.array( (actions*scaling_factor) / dt )
        
    
        finalpos = self.position + actions*scaling_factor
        
        
        
        # step
        stat = self.sim.setObjectPosition(self.targetIDR,-1,finalpos.tolist())
        
        self.position = finalpos
        
        # observe again
        self.self_observe()

        #Calculating reward
        dist_goal = np.linalg.norm(np.array(finalpos) - np.array(self.currentgoal))
        
        
        reward = -1 if dist_goal>self.distance_threshold else 0
        
        #setting done to True
        self.num_steps += 1       
        
        #print(self.num_steps)
        
        if self.num_steps >= self.max_steps or dist_goal<self.distance_threshold:
            print(self.num_steps) 
            done = True
            if dist_goal<self.distance_threshold:
                print('Goal Achieved!!')
                
                #Random Sampling of Goal
                # self.currentgoal = random.sample(self.endPos,1)
               
        else :
            done = False
             
        if not (self.position[0]>-3.3 and self.position[0]<-3.1 and self.position[1]>-1 and self.position[1]<1 and self.position[2]>1.35 and self.position[2]<1.6):
            reward = self.num_steps - self.max_steps-1
            #print(self.num_steps)
            done=True
          
         
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
        
        #sampling Random Goal After each episode
        self.currentgoal =  random.sample(self.endPos, 1)[0]
        
        self.sim.setObjectPosition(self.goalSphere,-1,self.currentgoal  )
        
        self.num_steps = 0  
        
        self.self_observe()
        
        self.position = self.sim.getObjectPosition(self.targetIDR,-1)
 
        
        return self.observation

     

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = np.linalg.norm(np.array(achieved_goal) - np.array(desired_goal))
        return -np.array(d > self.distance_threshold, dtype=np.float64)
    







print(__name__)
if __name__ == '__main__':
    env = dVRKCopeliaVisionEnv()
    
    done = None
    for k in range(10):
        
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
     