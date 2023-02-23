import random
from typing import Any, Dict, Union 
from collections import OrderedDict
import numpy as np
import time
from zmqRemoteApi import RemoteAPIClient
import sys
MAX_INT = sys.maxsize

import os
import pandas as pd
import gym
import gym.spaces 
from gym import spaces 

class dVRKCopeliaEnv(gym.GoalEnv):
    def __init__(self,maxsteps=100):
        
        self.distance_threshold = 0.03  
        
        
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
        
        #getting positions
         
        self.endPos = []
        
       
            
        goal_pos = np.array(self.sim.getObjectPosition(self.goalCube,-1))# setting goal position as greenCube
        #self.endPos.append(goal_pos.tolist())
        #removing start position from  sampling goal array
        #self.endPos.pop(0)
        #-3.224,0.053,1.3968e+00
        
        self.startPos = [-3.17,0.08,1.3968] # setting start position as redCube
        
        self.currentstart = self.startPos
        
        self.currentgoal =  goal_pos
        #setting goalCube position
        #self.sim.setObjectPosition(self.goalCube,-1,self.currentgoal)
        
        #workspacee boundaries
        self.workspace_limits = [[-3.3 ,-3.1],[-1 ,1],[1.35 ,1.6]]
        
        
        print('(dVRKVREP) initialized')

        obs = np.inf
        act = 1

        self.action_space = spaces.Box(-act,act, shape = (2,))
        self.observation_space = spaces.Dict(
            dict(
                observation= spaces.Box(-obs,obs,shape=(12,), dtype=np.float32),
                desired_goal= spaces.Box(-obs,obs,shape=(3,), dtype=np.float32),
                achieved_goal= spaces.Box(-obs,obs,shape=(3,), dtype=np.float32),
            )
        )
         
        self.velocity = np.zeros([3])
        self.self_observe()
        
        
        ''' csv Save '''
        
        self.timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
 
        self.df = pd.DataFrame()
         
        self.action_arr = []
        self.points_arr = []
        self.ep_len = 0
        self.ep_reward = 0
        self.trajectory_len =0
 
    
    
    
    
    
    def self_observe(self):
        """
        Helper to create the observation.
        :return: The current observation.
        """
        
        
        end_effector_pos = self.sim.getObjectPosition(self.targetIDR,-1)
        pushCube_pos = self.sim.getObjectPosition(self.pushCube,-1)
        pushCube_vel = self.sim.getObjectVelocity(self.pushCube)[0]
         
        pushCube_pos = np.array([
            pushCube_pos[0], pushCube_pos[1],pushCube_pos[2],
            ]).astype('float32')
        pushCube_vel = np.array([
            pushCube_vel[0], pushCube_vel[1],pushCube_vel[2],
            ]).astype('float32')
        
        current_pos = np.array([
            end_effector_pos[0], end_effector_pos[1],end_effector_pos[2],
            ]).astype('float32')
        
        current_state = np.concatenate((current_pos, self.velocity, pushCube_pos, pushCube_vel), axis=0)
        
        
        print('end eff pos=',current_pos)
        print('cube pos=',pushCube_pos)
        
        
        self.observation = OrderedDict([ ("observation",  current_state), ("achieved_goal", pushCube_pos[:3]),
                ("desired_goal", np.array(self.currentgoal))])
        
        
        
        

    
        
        
    
    def step(self,actions):
        
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
        actions = np.array([actions[0],actions[1],0])
        
        scaling_factor = 0.005
        
        #Calculating velocity
        dt = self.sim.getSimulationTimeStep()
        self.velocity = np.array( (actions*scaling_factor) / dt )
        
        action_vec =  actions*scaling_factor 
    
        finalpos = self.observation['observation'][:3] + action_vec
        
        
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
             
        
        
        if not (self.observation['observation'][0]>-3.3 and self.observation['observation'][0]<-3.1 and self.observation['observation'][1]>-1 and self.observation['observation'][1]<1 and self.observation['observation'][2]>1.35 and self.observation['observation'][2]<1.6):
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
        
        '''Save to CSV : Code to save the parameters into a CSV file for graphing'''  
        
        
        self.action_arr.append(list(actions))
        self.points_arr.append(list(self.observation['observation'][0:3]))
        self.trajectory_len += np.linalg.norm(action_vec)
        self.ep_reward += reward
         
        
        if done:
            
            timestr2 = time.strftime("%Y_%m_%d-%H_%M_%S")
            file_path = f"csv_save/"
            
            if not os.path.exists(file_path):
                os.makedirs(file_path)
                
                
            file_name = 'wrapper_reset_train'+'_'+self.timestr+'.csv'
            
            
            savepath=os.path.join(file_path,file_name)
            
            df = self.df
            df = df.append({'timestamp':timestr2, 'start_point':self.currentstart, 'goal_point':self.currentgoal,
                            'total_step':self.num_steps, 'action':self.action_arr, 
                            'path_points':self.points_arr,'trajectory_len':self.trajectory_len,
                            'ep_reward':self.ep_reward, 'goal_reached':reward==0 },
                           ignore_index=True)
         
            if os.path.exists(savepath):
                df.to_csv(savepath,mode='a',header=False)
            else:
                df.to_csv(savepath,mode='a',header=True)
        
        
        return self.observation, reward, done, {}

    def render(self):
        pass
    
    def reset(self):
        #stop Simulation    
        self.sim.stopSimulation()
        print('end')
        
        #to make sure we really stopped:
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.01);
        
        
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
        
        '''CSV saving'''
        
         
        self.action_arr = []
        self.points_arr = []
        self.ep_len = 0
        self.ep_reward = 0
        self.trajectory_len = 0
        
        return self.observation

     

    def compute_reward(self, achieved_goal, desired_goal,info):
        d = np.linalg.norm(np.array(achieved_goal) - np.array(desired_goal))
        return -np.array(d > self.distance_threshold, dtype=np.float64)
    







print(__name__)
if __name__ == '__main__':
    env = dVRKCopeliaEnv(maxsteps=300)
    
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
     
