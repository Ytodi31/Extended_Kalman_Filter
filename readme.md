# Extended Kalman Filter


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
The project recursively estimate the position of a vehicle along a trajectory using available measurements from a LiDAR sensor and a motion model.
The LiDAR measurements contain range and bear measurements corresponding to landmarks in the environment.

The project uses a non-linear motion model and non-linear measurement model. Since Kalman Filter is only good for linear systems, Extended Kalman Filter is used to account for the non-linearity in the model.

## Dependencies/ Libraries
The code is developed in Python with the following libraries:

* pickle
* numpy
* matplotlib

## Pipeline
Motion model :

<p align="center">
  <img src="/data/motion_model.png" width="600" height = "150" title="Detection of Obstacle from LiDAR data">
</p>

Measurement model :

<p align="center">
  <img src="/data/measurement_model.png" width="600" height = "150" title="Detection of Obstacle from LiDAR data">
</p>

Prediction Step :
Uses odometry information and the motion model to produce a state

Correction Step :
Uses sensor measurements to correct the pose estimates

<p align="center">
  <img src="/data/gtruth.png" width="300 height = "150" title="Detection of Obstacle from LiDAR data">
</p>

<p align="center">
  <img src="/data/gtruth2.png" width="300 height = "150" title="Detection of Obstacle from LiDAR data">
</p>

## Results

The images below represent the estimated trajectory using the EKF :

<p align="center">
  <img src="/data/estimated1.png" width="300 height = "150" title="Detection of Obstacle from LiDAR data">
</p>

<p align="center">
  <img src="/data/estimated2.png" width="300 height = "150" title="Detection of Obstacle from LiDAR data">
</p>
