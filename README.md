# Deep Reinforcement Learning for Autonomous Laparoscopic Surgery

This repository contains my Dual Degree Project work as part of my IDDD (Inter-Disciplinary Dual Degree) Programme in IIT Madras, India. Experiments are performed on the [da Vinci Research Kit Surgical Robot](https://www.intuitive-foundation.org/dvrk/) on the CopeliaSim Robotics Simulator. 

## Simulator - RL Interface 

 
<div align="center">
<img src="https://user-images.githubusercontent.com/52667241/226593143-755e02c3-08c1-4ca5-82b3-ca64cd29761b.png" height="300px" width="407px">
</div>

### Modelled Tasks
* [Multi-Goal Reach Task](#multi-goal-reach) - DDPG+HER algorithm is used to learn optimal trajectories from a start point to multiple queried goal locations.
* [Block Push Task](#block-push) 
    * [Using Simulator Coordinates](#baseline-algorithm) - To push the block from a given start to goal position using the coordinates of the block, end effector, and goal which are taken from the simulator
    * [End-to-End Image Based Training](#end-to-end-image) - To push the block from a given start to goal position by training directly on Images from an endoscopic camera. This is done by using a CNN feature extraction pipeline attached to the RL algorithm.

https://user-images.githubusercontent.com/52667241/222405809-d6c34a8f-68e3-4dd0-aa60-57d91778bab8.mp4 



 
 
# Requirements and Dependencies

* NVIDIA GPU 
* [PyTorch with CUDA](https://pytorch.org/)   
* [Gym](https://github.com/openai/gym) Version 0.21.0
* [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) 
* [CopeliaSim Surgical Simulator](https://www.coppeliarobotics.com/downloads)

Note: All python installations are done using pip on Anaconda environment

Our implementations have been tested on Windows 10 with an NVIDIA Quadro P600 GPU.

## Quick Start

To run our algorithms and implementations, follow the below steps:

1. Clone this repository 

    ```bash
    git clone https://github.com/hruthikvs/RL_dVRK_surgery.git
    ```

2. Launch the required Copelia Environment from the *Copelia_env* folder

3. From the *Task_envs* folder select the respective folders
    
    * The `dVRK_peg_transfer.ttt` environment can be run using Multi-Goal Reach Task, Pick and Place Task (*TODO) and 3D Vision Task (*TODO)
    * The `block.ttt` or `block_small.ttt` can be run using the Task controllers present in the `RL_Coordinates_based/Block push fixed goal` folder as well as the `RL_Vision_based` folder.  These envrionments consists of a small sized peg board with a fixed goal location at one corner indicated by a green block.
    * The `block_randomise.ttt` environment can be run using the Task controllers present in the `RL_Coordinates_based/Block push randomise` folder. This environment supports randomisation of the initial push block (red) position as well as the goal block (green) position.
    * The `block_randomise_vision.ttt`  environment can be run using the Task controllers present in the `RL_Vision_based`. This environment contains a larger peg board, increasing task complexity, and a wider field of view camera to feed images to the Vision based model.

## Training

To train your own model:

1. The Train files are named with the convention `Train_<Algorithm>_<Task_env>.py`

2. Read the `Instructions.txt` file

3. The training logs are stored in the `.\logs' folder` and the trained models are stored with timesteps in the `.\models` folder 

4. To Visualise the Training metrics, launch command prompt in the directory with containing `logs` folder and run the below command 

    ```cmd
    tensorboard --logdir=logs
    ```
Then open the local host URL on the browser to see the training metrics plots

 

## Testing 

To evaluate a trained model:

1. Navigate to the folder containing the test file with naming convention `Test_<Algorithm>_<Task_env>.py`.

2. modify the file path in the code section below to the respective `<model_folder>/<train_timesteps>.zip`

    ```python
    model = model_class.load('models/DDPG_HER-1676442643/50000.zip', env=env)
    ```

 
