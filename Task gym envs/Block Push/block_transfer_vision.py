# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:51:48 2023

@author: Hruthik V S
"""

import random
from typing import Any, Dict, Union 
from collections import OrderedDict
import numpy as np
import time
import cv2
from zmqRemoteApi import RemoteAPIClient
import sys
MAX_INT = sys.maxsize


import gym
import gym.spaces 
from gym import spaces 

class dVRKBlockVisionEnv(gym.GoalEnv):
    def __init__(self,maxsteps=100):
         
        self.distance_threshold = 0.04  
        
        
        self.max_steps = maxsteps 
        self.num_steps = 0
               
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')
        self.sim.intparam_speedmodifier
        self.sim.startSimulation()
        
        # getting handles
        self.toolTip = self.sim.getObjectHandle("/L3_dx_respondable_TOOL2")
        self.targetIDR = self.sim.getObjectHandle("/TargetPSMR")
        self.goalCube = self.sim.getObjectHandle("/goalCube")  #getting object handle for goal cube/ green block
        self.pushCube = self.sim.getObjectHandle("/pushCube")  #getting object handle for push cube/ red block
        self.visionSensor =  self.sim.getObjectHandle("/Vision_sensor_left")
        
        #getting positions
         
        self.endPos = []
        
       
            
        goal_pos = np.array(self.sim.getObjectPosition(self.goalCube,-1))# setting goal position as greenCube
        #self.endPos.append(goal_pos.tolist())
        #removing start position from  sampling goal array
        #self.endPos.pop(0)
        #-3.224,0.053,1.3968e+00
        
        self.startPos = [-3.17,0.08,1.3968] # setting start position as redCube
        
        self.currentgoal =  goal_pos
        #setting goalCube position
        #self.sim.setObjectPosition(self.goalCube,-1,self.currentgoal)
        
        #workspacee boundaries
        self.workspace_limits = [[-3.3 ,-3.1],[-1 ,1],[1.35 ,1.6]]
        
        
        print('(dVRKVREP) initialized')

        obs = np.inf
        act = 1

        
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensor)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        
        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        img = cv2.resize(img,(100,100))
        print(img.shape)
        #Converting to GrayScale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        
        
         
         
        self.action_space = spaces.Box(-act,act, shape = (2,))
        self.observation_space = spaces.Dict(
            dict(
                observation= spaces.Box(0,255,shape=(img.shape[2],img.shape[0],img.shape[1]), dtype=np.uint8),
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
        
        
        self.initpos = self.sim.getObjectPosition(self.targetIDR,-1)
        pushCube_pos = self.sim.getObjectPosition(self.pushCube,-1)
         
        pushCube_pos = np.array([
            pushCube_pos[0], pushCube_pos[1],pushCube_pos[2],
            ]).astype('float32') 
        
        
        
        
         
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensor)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        
        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        img = cv2.resize(img,(100,100))
        # cv2.imshow('', img)
        # cv2.waitKey(1)
        img = img.transpose((-1, 0, 1))
        
        
        
        
        self.observation = OrderedDict([ ("observation",  img), ("achieved_goal", pushCube_pos[:3]),
                ("desired_goal", np.array(self.currentgoal))])
        
        
        
        

    
        
        
    
    def step(self,actions):
        
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
        actions = np.array([actions[0],actions[1],0])
        
        scaling_factor = 0.005
        
        #Calculating velocity
        dt = self.sim.getSimulationTimeStep()
        self.velocity = np.array( (actions*scaling_factor) / dt )
        
        action_vec =  actions*scaling_factor 
        
         
        finalpos = self.initpos + action_vec
        
        
        print('action=',actions)
        
        # step
        stat = self.sim.setObjectPosition(self.targetIDR,-1,finalpos.tolist())
        
        
        # observe again
        self.self_observe()
         
        
        #Getting cube position 
        pushCube_pos = self.observation['achieved_goal']
        
        #Calculating reward
        dist_goal = np.linalg.norm(pushCube_pos[:2] - np.array(self.currentgoal)[:2])
        
        
        reward = -1 if dist_goal>self.distance_threshold else 0
        
        #setting done to True
        self.num_steps += 1       
        
        print(self.num_steps)
        
        if self.num_steps >= self.max_steps or dist_goal<self.distance_threshold:
            print(self.num_steps) 
            done = True
            if dist_goal<self.distance_threshold:
                print('Goal Achieved!!')
                
                #Random Sampling of Goal
                # self.currentgoal = random.sample(self.endPos,1)
               
        else :
            done = False
             
        
        
        if not (self.initpos[0]>-3.3 and self.initpos[0]<-3.1 and self.initpos[1]>-0.08 and self.initpos[1]<0.1 and self.initpos[2]>1.35 and self.initpos[2]<1.6):
            reward = self.num_steps - self.max_steps -1
            print(self.num_steps)
            done=True
        
         
        #when gripper reaches boundary
        # position = self.observation['observation'].copy()
        
        # position[0] = min(max(position[0],self.workspace_limits[0][0]),self.workspace_limits[0][1])
        # position[1] = min(max(position[1],self.workspace_limits[1][0]),self.workspace_limits[1][1])
        # position[2] = min(max(position[2],self.workspace_limits[2][0]),self.workspace_limits[2][1])
        
        # print(position)
        # self.observation['observation'] = position.copy()
        
        
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
        #self.currentgoal =  goal_pos
        
        #self.sim.setObjectPosition(self.goalCube,-1,self.currentgoal  )
        
        self.num_steps = 0  
        
        self.self_observe()
        
        return self.observation

     

    def compute_reward(self, achieved_goal, desired_goal,info):
        d = np.linalg.norm(np.array(achieved_goal) - np.array(desired_goal))
        return -np.array(d > self.distance_threshold, dtype=np.float64)
    







print(__name__)
if __name__ == '__main__':
    env = dVRKBlockVisionEnv(maxsteps=300)
    
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
     
