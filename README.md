# Deep RL With MuJoCo

Final project for the 2023-winter's edition of the UPC's *Artificial Intelligence with Deep Learning* postgraduate course.

The goal of this project is to train diverse AI models to perform tasks using (Deep) Reinforcement Learning in the MuJoCo physics simulator.

### Team Members

* Adrià De Angulo

* Daniel Matas

* Hariss Farhan

* Miquel Quesada

### Project Advisor

* JuanJo Nieto

## Index

1. [Introduction To Reinforcement Learning](#intro-to-rl)
2. [MuJoCo](#mujoco)
3. [Computational Resources](#comp-res)
4. [How To Run](#how-to-run)
5. [Proximal Policy Optimization (PPO)](#ppo)
6. [Half Cheetah](#halfcheetah)
    1. [Overview](#overview1)
    2. [Architecture](architecture1)
7. [ANYmal C](#anymal-c)
    1. [Overview](#overview2)
    2. [Architecture](architecture2)



## Introduction To Reinforcement Learning <a name="intro-to-rl"></a>

Reinforcement Learning is an area of Machine Learning concerned with how intelligent agents ought to take actions in an environment based on rewarding desired behaviors and/or punishing undesired ones. 
 
> "Reinforcement learning is learning what to do—how to map situations to actions—so
as to maximize a numerical reward signal. The learner is not told which actions to
take, but instead must discover which actions yield the most reward by trying them. In the most interesting and challenging cases, actions may affect not only the immediate reward but also the next situation and, through that, all subsequent rewards. These two characteristics—trial-and-error search and delayed reward—are the two most important distinguishing features of reinforcement learning." — Richard S. Sutton and Andrew G. Barto [^1]

[^1]: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

## MuJoCo <a name="mujoco"></a>


> [MuJoCo](https://mujoco.org/) is a general purpose physics engine that aims to facilitate research and development in robotics, biomechanics, graphics and animation, machine learning, and other areas that demand fast and accurate simulation of articulated structures interacting with their environment.

Initially developed by *Roboti LLC*, it was acquired and made freely available by *DeepMind* in October 2021, and open sourced in May 2022.

<center><em><img src="https://github.com/danimatasd/MUJOCO-AIDL/blob/main/misc/example_mujoco.gif?raw=true" width=35%></em></center>

## PPO <a name="ppo"></a>


Proximal Policy Optimization (PPO) is considered the state-of-the-art in 
Reinforcement Learning, and it consists of a policy gradient method whose main
goal is to have an algorithm that is as reliable as possible, and is data efficient. 

In Reinforcement Learning, there are many methods that can be classified into 
two main groups: policy-based methods and value-based methods. 

// meter grapho

On one hand, a policy-based method is a type of reinforcement learning algorithm that learns a policy directly without explicitly computing the value function. In this approach, the agent directly learns a policy that maps the state to actions.

The policy is typically represented as a probability distribution over actions, and the agent tries to maximize the expected cumulative reward over time by adjusting its policy based on feedback from the environment.

Policy-based methods can be classified into two main categories: deterministic policy gradient (DPG) methods and stochastic policy gradient (SPG) methods.


 A value-based method is an algorithm that learns to estimate the value function of states or state-action pairs. The value function represents the expected cumulative reward an agent can obtain from a particular state or state-action pair over time.

// meter grapho 

In value-based methods, the agent uses this value function to choose the action that maximizes its expected cumulative reward. The value function can be learned using various techniques, such as Monte Carlo methods, temporal difference learning, or Q-learning.

PPO is a policy-based reinforcement learning algorithm. It directly learns a policy that maps states to actions, rather than estimating a value function like value-based methods such as Q-learning.

In particular, PPO optimizes the policy by maximizing a surrogate objective function that directly estimates the policy's expected performance. This surrogate objective function is optimized using stochastic gradient descent, with the policy being updated to maximize this objective while also satisfying a trust region constraint to ensure the policy update is not too large.

As a policy-based algorithm, PPO is well-suited for tasks with continuous action spaces, and can handle stochastic policies, which can be important in environments with complex dynamics or multiple optimal actions.

Given the complexity of our model, which involves multiple joints and actions, we decided to utilize this algorithm for our project.

## Computational Resources <a name="comp-res"></a>

For the development of this project we have mainly used Google Colab; also a desktop PC with better specs to speed up the trainings of our models.

Here are the specifications of the machines:

* Google Colab
    * Intel Xeon @ 2.20GHz
    * 12GB RAM
    * NVIDIA Tesla K80 12GB GDDR5

* Desktop PC
    * AMD Ryzen 7 5800H @ 3.20 GHz
    * 16GB RAM
    * NVIDIA GeForce RTX 3070 8GB GDDR6

## How To Run <a name="how-to-run"></a>


## Half Cheetah <a name="half-cheetah"></a>


### Overview <a name="overview1"></a>

The HalfCheetah is a 2-dimensional robot consisting of 9 links and 8 joints connecting them (including two paws). The goal is to apply a torque on the joints to make the cheetah run forward (right) as fast as possible, with a positive reward allocated based on the distance moved forward and a negative reward allocated for moving backward. The torso and head of the cheetah are fixed, and the torque can only be applied on the other 6 joints over the front and back thighs (connecting to the torso), shins (connecting to the thighs) and feet (connecting to the shins).

![Half Cheetah](https://github.com/danimatasd/MUJOCO-AIDL/blob/main/misc/half_cheetah.gif?raw=true)

### Architecture <a name="architecture1"></a>

## Anybotics ANYmal C <a name="anymal-c"></a>


### Overview <a name="overview2"></a>

![ANYmal C](https://github.com/danimatasd/MUJOCO-AIDL/blob/main/misc/Anybotics%20ANYmal%20C.jpg?raw=true)
<font size= "2"><center><em>[Anybotics ANYmal C](https://www.anybotics.com/anymal-autonomous-legged-robot/)</em></center></font>

### Architecture <a name="architecture2"></a>