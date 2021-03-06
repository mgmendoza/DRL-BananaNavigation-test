# DRL-BananaNavigation-test

### Background
This project is part of the first assignment of the Udacity Deep Reinforcement Learning Nanodegree. 

Here is a GIF of my trained banana collector: 
![](GIF/trained_banana.gif)

## Introduction
This project trains an agent to navigate and collect bananas in a square world

Goal: Navigate to collect the yellow bananas and avoid blue bananas
Reward:
* +1 is provided for collecting a yellow banana
* -1 is penalized for collecting a blue banana

(Instructions taken and modified from Udacity's original version found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation))

The state space has n dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. 

Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

1. Clone this repository: '<https://github.com/mgmendoza/DRL-BananaNavigation-test.git>'

2. Follow the instructions specified in README of the [Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set your python environment. The instructions will also help you install OpenAI gym, PyTorch, and ML-agents which are packages required to run the project.  

3. Download the environment from one of the links on the Getting Started section found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation))

4. Place the file in the DRLND GitHub repository, in the p1_navigation/ folder, and unzip (or decompress) the file.

## Instructions
Follow the instructions in Banana_Collector.ipynb to get started with training your own agent! 

You should be able to watch your agent collect bananas correctly in the last step. 
