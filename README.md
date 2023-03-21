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
* [Torch](http://torch.ch/)   
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

2. Launch the required Copelia Environment 

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-snapshot-10001.t7
    ```

    Direct download link: [suction-based-grasping-snapshot-10001.t7 (450.1 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-snapshot-10001.t7)

3. Run our model on an optional target RGB-D image. Input color images should be 24-bit RGB PNG, while depth images should be 16-bit PNG, where depth values are saved in deci-millimeters (10<sup>-4</sup>m).

    ```bash
    th infer.lua # creates results.h5
    ```

    or

    ```bash
    imgColorPath=<image.png> imgDepthPath=<image.png> modelPath=<model.t7> th infer.lua # creates results.h5
    ```

4. Visualize the predictions in Matlab. Shows a heat map of confidence values where hotter regions indicate better locations for grasping with suction. Also displays computed surface normals, which can be used to decide between robot motion primitives suction-down or suction-side. Run the following in Matlab:

    ```matlab
    visualize; % creates results.png and normals.png
    ```

## Training

To train your own model:

1. Navigate to `arc-robot-vision/suction-based-grasping`

    ```bash
    cd arc-robot-vision/suction-based-grasping
    ```

2. Download our suction-based grasping dataset and save the files into `arc-robot-vision/suction-based-grasping/data`. More information about the dataset can be found [here](http://vision.princeton.edu/projects/2017/arc/#datasets).

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-dataset.zip
    unzip suction-based-grasping-dataset.zip # unzip dataset
    ```

    Direct download link: [suction-based-grasping-dataset.zip (1.6 GB)](http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-dataset.zip)

3. Download the Torch ResNet-101 model pre-trained on ImageNet:

    ```bash
    cd convnet
    wget http://vision.princeton.edu/projects/2017/arc/downloads/resnet-101.t7
    ```

    Direct download link: [resnet-101.t7 (409.4 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/resnet-101.t7)

4. Run training (set optional parameters through command line arguments):

    ```bash
    th train.lua
     ```

    Tip: if you run out of GPU memory (CUDA error=2), reduce batch size or modify the network architecture in `model.lua` to use the smaller [ResNet-50 (256.7 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/resnet-50.t7) model pre-trained on ImageNet.

## Evaluation

To evaluate a trained model:

1. Navigate to `arc-robot-vision/suction-based-grasping/convnet`

    ```bash
    cd arc-robot-vision/suction-based-grasping/convnet
    ```

2. Run our pre-trained model to get affordance predictions for the testing split of our grasping dataset:

    ```bash
    th test.lua # creates evaluation-results.h5
    ```

    or run your own model:

    ```bash
    modelPath=<model.t7> th test.lua # creates evaluation-results.h5
    ```

3. Run the evaluation script in Matlab to compute pixel-level precision against manual annotations from the grasping dataset, as reported in our [paper](https://arxiv.org/pdf/1710.01330.pdf):

    ```matlab
    evaluate;
    ```

## Baseline Algorithm

Our baseline algorithm predicts affordances for suction-based grasping by first computing 3D surface normals of the point cloud (projected from the RGB-D image), then measuring the variance of the surface normals (higher variance = lower affordance). To run our baseline algorithm over the testing split of our grasping dataset:

1. Navigate to `arc-robot-vision/suction-based-grasping/baseline`

    ```bash
    cd arc-robot-vision/suction-based-grasping/baseline
    ```

2. Run the following in Matlab:

    ```matlab
    test; % creates results.mat
    evaluate;
    ```

# Parallel-Jaw Grasping

A Torch implementation of fully convolutional neural networks for predicting pixel-level affordances for parallel-jaw grasping. The network takes an RGB-D heightmap as input, and outputs affordances for **horizontal** grasps. Input heightmaps can be rotated at any arbitrary angle. This structure allows the use of a unified model to predict grasp affordances for any possible grasping angle. 

![parallel-jaw-grasping](images/parallel-jaw-grasping.jpg?raw=true)

**Heightmaps** are generated by orthographically re-projecting 3D point clouds (from RGB-D images) upwards along the gravity direction where the height value of bin bottom = 0 (see [getHeightmap.m](parallel-jaw-grasping/convnet/getHeightmap.m)).

## Quick Start

To run our pre-trained model to get pixel-level affordances for parallel-jaw grasping:

1. Clone this repository and navigate to `arc-robot-vision/parallel-jaw-grasping/convnet`

    ```bash
    git clone https://github.com/andyzeng/arc-robot-vision.git
    cd arc-robot-vision/parallel-jaw-grasping/convnet
    ```

2. Download our pre-trained model for parallel-jaw grasping:

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/parallel-jaw-grasping-snapshot-20001.t7
    ```

    Direct download link: [parallel-jaw-grasping-snapshot-20001.t7 (450.1 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/parallel-jaw-grasping-snapshot-20001.t7)

3. To generate a RGB-D heightmap given two RGB-D images, run the following in Matlab:

    ```matlab
    getHeightmap;
    ```

4. Run our model on an optional target RGB-D heightmap. Input color images should be 24-bit RGB PNG, while height images (depth) should be 16-bit PNG, where height values are saved in deci-millimeters (10<sup>-4</sup>m) and bin bottom = 0.

    ```bash
    th infer.lua # creates results.h5
    ```

    or

    ```bash
    imgColorPath=<image.png> imgDepthPath=<image.png> modelPath=<model.t7> th infer.lua # creates results.h5
    ```

5. Visualize the predictions in Matlab. Shows a heat map of confidence values where hotter regions indicate better locations for horizontal parallel-jaw grasping. Run the following in Matlab:

    ```matlab
    visualize; % creates results.png
    ```

## Training

To train your own model:

1. Navigate to `arc-robot-vision/parallel-jaw-grasping`

    ```bash
    cd arc-robot-vision/parallel-jaw-grasping
    ```

2. Download our parallel-jaw grasping dataset and save the files into `arc-robot-vision/parallel-jaw-grasping/data`. More information about the dataset can be found [here](http://vision.princeton.edu/projects/2017/arc/#datasets).

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/parallel-jaw-grasping-dataset.zip
    unzip parallel-jaw-grasping-dataset.zip # unzip dataset
    ```

    Direct download link: [parallel-jaw-grasping-dataset.zip (711.8 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/parallel-jaw-grasping-dataset.zip)

3. Pre-process input data and labels for parallel-jaw grasping dataset and save the files into `arc-robot-vision/parallel-jaw-grasping/convnet/training`. Pre-processing includes rotating heightmaps into 16 discrete rotations, converting raw grasp labels (two-point lines) into dense pixel-wise labels, and augmenting labels with small amounts of jittering. Either run the following in Matlab:

    ```matlab
    cd convnet;
    processLabels;
    ```

    or download our already pre-processed input:

    ```bash
    cd convnet;
    wget http://vision.princeton.edu/projects/2017/arc/downloads/training-parallel-jaw-grasping-dataset.zip
    unzip training-parallel-jaw-grasping-dataset.zip # unzip dataset
    ```

    Direct download link: [training-parallel-jaw-grasping-dataset.zip (740.0 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/training-parallel-jaw-grasping-dataset.zip)

4. Download the Torch ResNet-101 model pre-trained on ImageNet:

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/resnet-101.t7
    ```

    Direct download link: [resnet-101.t7 (409.4 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/resnet-101.t7)

5. Run training (set optional parameters through command line arguments):

    ```bash
    th train.lua
     ```

    Tip: if you run out of GPU memory (CUDA error=2), reduce batch size or modify the network architecture in `model.lua` to use the smaller [ResNet-50 (256.7 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/resnet-50.t7) model pre-trained on ImageNet.

## Evaluation

To evaluate a trained model:

1. Navigate to `arc-robot-vision/parallel-jaw-grasping/convnet`

    ```bash
    cd arc-robot-vision/parallel-jaw-grasping/convnet
    ```

2. Run the model to get affordance predictions for the testing split of our grasping dataset:

    ```bash
    modelPath=<model.t7> th test.lua # creates evaluation-results.h5
    ```

3. Run the evaluation script in Matlab to compute pixel-level precision against manual annotations from the grasping dataset, as reported in our [paper](https://arxiv.org/pdf/1710.01330.pdf):

    ```matlab
    evaluate;
    ```

## Baseline Algorithm

Our baseline algorithm detects anti-podal parallel-jaw grasps by detecting "hill-like" geometric features (through brute-force sliding window search) from the 3D point cloud of an input heightmap (no color). These geometric features should satisfy two constraints: (1) gripper fingers fit within the concavities along the sides of the hill, and (2) top of the hill should be at least 2cm above the lowest points of the concavities. A valid grasp is ranked by an affordance score, which is computed by the percentage of 3D surface points between the gripper fingers that are above the lowest points of the concavities. To run our baseline algorithm over the testing split of our grasping dataset:

1. Navigate to `arc-robot-vision/parallel-jaw-grasping/baseline`

    ```bash
    cd arc-robot-vision/parallel-jaw-grasping/baseline
    ```
    
2. Run the following in Matlab:

    ```matlab
    test; % creates results.mat
    evaluate;
    ```

# Image Matching

A Torch implementation of two-stream convolutional neural networks for matching observed images of grasped objects to their product images for recognition. One stream computes 2048-dimensional feature vectors for product images while the other stream computes 2048-dimensional feature vectors for observed images. During training, both streams are optimized so that features are more similar for images of the same object and dissimilar otherwise. During testing, product images of both known and novel objects are mapped onto a common feature space. We recognize observed images by mapping them to the same feature space and finding the nearest neighbor product image match.

![image-matching](images/image-matching.jpg?raw=true)

## Training

To train a model:

1. Navigate to `arc-robot-vision/image-matching`

    ```bash
    cd arc-robot-vision/image-matching
    ```

2. Download our image matching dataset and save the files into `arc-robot-vision/image-matching/data`. More information about the dataset can be found [here](http://vision.princeton.edu/projects/2017/arc/#datasets).

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/image-matching-dataset.zip
    unzip image-matching-dataset.zip # unzip dataset
    ```

    Direct download link: [image-matching-dataset.zip (4.6 GB)](http://vision.princeton.edu/projects/2017/arc/downloads/image-matching-dataset.zip)

3. Download the Torch ResNet-50 model pre-trained on ImageNet:

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/resnet-50.t7
    ```

    Direct download link: [resnet-50.t7 (256.7 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/resnet-50.t7)

4. Run training (change variable `trainMode` in `train.lua` depending on which architecture you want to train):

    ```bash
    th train.lua
     ```

## Evaluation

To evaluate a trained model:

1. Navigate to `arc-robot-vision/image-matching`

    ```bash
    cd arc-robot-vision/image-matching
    ```

2. Download our pre-trained models (K-net and N-net) for two-stage cross-domain image matching:

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/k-net.zip
    unzip k-net.zip 
    wget http://vision.princeton.edu/projects/2017/arc/downloads/n-net.zip
    unzip n-net.zip 
    ```

    Direct download links: [k-net.zip  (175.3 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/k-net.zip) and [n-net.zip (174.0 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/n-net.zip)

3. Run our pre-trained models to compute features for the testing split of our image matching dataset (change variable `trainMode` depending on which architecture you want to test):

    ```bash
    trainMode=1 snapshotsFolder=snapshots-with-class snapshotName=snapshot-170000 th test.lua # for k-net: creates HDF5 output file and saves into snapshots folder
    trainMode=2 snapshotsFolder=snapshots-no-class snapshotName=snapshot-8000 th test.lua # for n-net: creates HDF5 output file and saves into snapshots folder
    ```

4. Run the evaluation script in Matlab to compute 1 vs 20 object recognition accuracies over our image matching dataset, as reported in our [paper](https://arxiv.org/pdf/1710.01330.pdf):

    ```matlab
    evaluateTwoStage;
    ```

    or run the following in Matlab for evaluation on a single model (instead of a two stage system):

    ```matlab
    evaluateModel;
    ```

