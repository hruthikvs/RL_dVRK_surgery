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
        self.workspace_limit_high = np.array([ -3.1 ,0.4 ,1.6])
        self.workspace_limit_low =  np.array([-3.3,-0.08,1.35])
          
        
        print('(dVRKVREP) initialized')

        obs = np.inf
        act = 1

        
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensor)
        print(resX,resY)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        
        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        self.H,self.W = 84,84
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        img = cv2.resize(img,(self.H,self.W))
        
        #Converting to GrayScale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        
        
        
        self.frame_stack_len = 3
        self.frame_stack = []
        for i in range(self.frame_stack_len):
            self.frame_stack.append(np.zeros([self.H,self.W]))
        
        self.frame_stack.pop(0)
        self.frame_stack.append(img)
         
        
        self.action_space = spaces.Box(-act,act, shape = (2,))
        self.observation_space = spaces.Box(0,255,shape=(self.frame_stack_len,img.shape[0],img.shape[1]), dtype=np.uint8)
         
        self.velocity = np.zeros([3])
        self.self_observe()
 
    
    
    
    
    
    def self_observe(self):
        """
        Helper to create the observation.
        :return: The current observation.
        """
        
        
        self.initpos = self.sim.getObjectPosition(self.targetIDR,-1)
        pushCube_pos = self.sim.getObjectPosition(self.pushCube,-1)
         
        self.pushCube_pos = np.array([
            pushCube_pos[0], pushCube_pos[1],pushCube_pos[2],
            ]).astype('float32') 
        
        
        
        
         
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensor)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        
        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        img = cv2.resize(img,(self.H,self.W))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        
        
        self.frame_stack.pop(0)
        self.frame_stack.append(img)
         
        # Display Image
        img = np.array(self.frame_stack).transpose((1, -1, 0))
        cv2.imshow('', img)
        cv2.waitKey(1)
        img = np.array(self.frame_stack)
        
        
        
    
        
        
        
        self.observation = img
        
        
        
        

    
        
        
    
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
        pushCube_pos = self.pushCube_pos
        
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
             
        
        
        #Checking for workspace singularity 
        if not np.logical_and(np.array(self.initpos)<= self.workspace_limit_high, np.array(self.initpos)>= self.workspace_limit_low).sum() == self.workspace_limit_high.size:
            reward = self.num_steps - self.max_steps-1
            print('-----collided with workspace boundary-----')
            done=True
        
        
        
        return self.observation, reward, done, {}

    def render(self):
        pass
    
    def reset(self):
        #stop Simulation    
        self.sim.stopSimulation()
        print('end')
        
        #to make sure we really stopped:
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.01)
        
        
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
     