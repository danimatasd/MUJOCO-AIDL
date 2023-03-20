# Deep RL With MuJoCo

Final project for the 2023-winter's edition of the *Artificial Intelligence with Deep Learning* postgraduate course taught by the Universitat Politècnica de Catalunya (UPC).

The goal of this project is to train several AI models to perform tasks using (Deep) Reinforcement Learning in the MuJoCo physics simulator.

### Team Members

* Adrià De Angulo
* Daniel Matas
* Hariss Farhan
* Miquel Quesada

### Project Advisor

* JuanJo Nieto

## Index

1. [How To Run](#how-to-run)
2. [Introduction To Reinforcement Learning](#intro-to-rl)
3. [Proximal Policy Optimization (PPO)](#ppo)
4. [MuJoCo](#mujoco)
    1. [Half Cheetah](#halfcheetah)
        1. [Overview](#overview1)
        2. [Architecture](#architecture1)
        3. [Results](#results1)
        4. [Conclusions](#conclusions1)
    2. [ANYmal C](#anymal-c)
        1. [Overview](#overview2)
        2. [Architecture](#architecture2)
        3. [Results](#results2)
        4. [Conclusions](#conclusions2)
6. [Future Work](#future-work)
7. [Computational Resources](#comp-res)

## How To Run <a name="how-to-run"></a>

In order to speed up the trainings of our models we have separated the video capture part of the code from the actual training code.

The easiest way to run the project is via Google Colab. Here you will find direct links to the notebooks. Open them and execute the cells.

Alternatively we have also included a folder with the python files in case you want to download the code and use it in your machine. Note that prior adaptation of the code to your machine will be needed beforehand.

* Half Cheetah (PPO)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danimatasd/MUJOCO-AIDL/blob/main/notebooks/SWEEP_RL_PPO_HALFCHEETAH.ipynb)

* Half Cheetah (Video Capture)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danimatasd/MUJOCO-AIDL/blob/main/notebooks/RL_PPO_HALFCHEETAH_VIDEO.ipynb)

* ANYmal C (PPO)

        ==PENDING==

* ANYmal C (Video Capture)

        ==PENDING==


## Introduction To Reinforcement Learning <a name="intro-to-rl"></a>

Reinforcement Learning is an area of Machine Learning concerned with how intelligent agents ought to take actions in an environment based on rewarding desired behaviors and/or punishing undesired ones. 
 
> "Reinforcement learning is learning what to do—how to map situations to actions—so
as to maximize a numerical reward signal. The learner is not told which actions to
take, but instead must discover which actions yield the most reward by trying them. In the most interesting and challenging cases, actions may affect not only the immediate reward but also the next situation and, through that, all subsequent rewards. These two characteristics—trial-and-error search and delayed reward—are the two most important distinguishing features of reinforcement learning." — Richard S. Sutton and Andrew G. Barto [^1]

[^1]: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

## PPO <a name="ppo"></a>

Proximal Policy Optimization (PPO) is considered the state-of-the-art in 
Reinforcement Learning, and it consists of a policy gradient method whose main
goal is to have an algorithm that is as reliable as possible, and is data efficient. 

In Reinforcement Learning, there are many methods that can be classified into 
two main groups: policy-based methods and value-based methods. 

<p align="center">
    <img src="./assets/types-of-rl-methods.svg" >
</p>

On one hand, a policy-based method is a type of reinforcement learning algorithm that learns a policy directly without explicitly computing the value function. In this approach, the agent directly learns a policy that maps the state to actions.

The policy is typically represented as a probability distribution over actions, and the agent tries to maximize the expected cumulative reward over time by adjusting its policy based on feedback from the environment.

Policy-based methods can be classified into two main categories: deterministic policy gradient (DPG) methods and stochastic policy gradient (SPG) methods.


 A value-based method is an algorithm that learns to estimate the value function of states or state-action pairs. The value function represents the expected cumulative reward an agent can obtain from a particular state or state-action pair over time.


In value-based methods, the agent uses this value function to choose the action that maximizes its expected cumulative reward. The value function can be learned using various techniques, such as Monte Carlo methods, temporal difference learning, or Q-learning.

PPO is a policy-based reinforcement learning algorithm. It directly learns a policy that maps states to actions, rather than estimating a value function like value-based methods such as Q-learning.

In particular, PPO optimizes the policy by maximizing a surrogate objective function that directly estimates the policy's expected performance. This surrogate objective function is optimized using stochastic gradient descent, with the policy being updated to maximize this objective while also satisfying a trust region constraint to ensure the policy update is not too large.

As a policy-based algorithm, PPO is well-suited for tasks with continuous action spaces, and can handle stochastic policies, which can be important in environments with complex dynamics or multiple optimal actions.

Given the complexity of our model, which involves multiple joints and actions, we decided to utilize this algorithm for our project.

## MuJoCo <a name="mujoco"></a>

 [MuJoCo](https://mujoco.org/) (which stands for Multi-Joint dynamics with Contact) is a general purpose physics engine that aims to facilitate research and development in robotics, biomechanics, graphics and animation, machine learning, and other areas that demand fast and accurate simulation of articulated structures interacting with their environment.

Initially developed by Roboti LLC, it was acquired and made freely available by DeepMind in October 2021, and open sourced in May 2022.

<p align="center">
    <img src="https://github.com/danimatasd/MUJOCO-AIDL/blob/main/misc/example_mujoco.gif?raw=true" width=30%>
</p>

## Half Cheetah <a name="halfcheetah"></a>

### Overview <a name="overview1"></a>

The HalfCheetah is a 2-dimensional robot consisting of 9 links and 8 joints connecting them (including two paws). The goal is to apply a torque on the joints to make the cheetah run forward (right) as fast as possible, with a positive reward allocated based on the distance moved forward and a negative reward allocated for moving backward. The torso and head of the cheetah are fixed, and the torque can only be applied on the other 6 joints over the front and back thighs (connecting to the torso), shins (connecting to the thighs) and feet (connecting to the shins).

<p align="center">
    <img src="https://github.com/danimatasd/MUJOCO-AIDL/blob/main/misc/half_cheetah.gif?raw=true">
</p>

### Architecture <a name="architecture1"></a>

We based the architecture from the lunar lander lab.

Lunar lander es discreto
Half cheetah es un entorno continuo

Tenemos 6 acciones que van de -1 a 1


    self.mlp = nn.Sequential(
        nn.Linear(obs_len, 64),
        nn.Tanh(),
        nn.Linear(64, 128),
        nn.Tanh())

    self.actor = nn.Linear(128, act_len)

    self.critic = nn.Linear(128, 1)

### Results <a name="results1"></a>

### Conclusions <a name="conclusions1"></a>


## Anybotics ANYmal C <a name="anymal-c"></a>

### Overview <a name="overview2"></a>

The ANYmal C is a quadruped robot used for inspection of industrial facilities. With it's 8 joints it is able to navigate through complex environments, such as industrial inspection, search and rescue, and scientific research. It is highly adaptable and can be customized to suit a wide range of tasks and environments, making it a versatile platform for robotics research and development. 

The Anybiotics ANYmal C model has 12 degrees of freedom, enabling it to execute a wide variety of dynamic movements, such as walking, trotting, running, climbing, and crawling. These degrees of freedom provide the robot with the ability to move its body in many different ways, making it highly versatile and adaptable.

<p align="center">
    <img src="https://github.com/danimatasd/MUJOCO-AIDL/blob/main/misc/Anybotics%20ANYmal%20C.jpg?raw=true">
</p>

<font size= "2"><center><em>[Anybotics ANYmal C](https://www.anybotics.com/anymal-autonomous-legged-robot/)</em></center></font>

### Architecture <a name="architecture2"></a>

We have added one hidden layer more than in the half cheetah environment and we use more input parameters.

    self.mlp = nn.Sequential(
        nn.Linear(obs_len, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh())

    self.actor = nn.Sequential(
        nn.Linear(128,64),
        nn.Tanh(),
        nn.Linear(64,act_len))

    self.critic = nn.Sequential(
        nn.Linear(128,64),
        nn.Tanh(),
        nn.Linear(64,1))

The architecture consists of three neural networks: an MLP (multi-layer perceptron), an actor network, and a critic network. Here is a breakdown of each component:

The  multi-layer perceptron  consists of two fully connected layers with 128 neurons each, followed by a hyperbolic tangent activation function.

The output of the multilayer perceptron is passed as input to both the actor and the critic networks.

The actor, is responsible for producing the policy distribution over actions, whereas the critic  is responsible for estimating the state value function, which is the expected return starting from the current state. 

As you can see, the architecture follows a common pattern in reinforcement learning called the actor-critic method. The actor network generates a policy distribution over actions, while the critic network estimates the value of each state or state-action pair. The actor uses the critic's estimates to improve the policy, while the critic learns to predict the expected returns accurately. 

### Results <a name="results2"></a>

<img src="https://github.com/danimatasd/MUJOCO-AIDL/blob/main/misc/Video_Anymal_C_Reward_8876.gif">

<img src="https://github.com/danimatasd/MUJOCO-AIDL/blob/main/misc/Video_Anymal_C_Step_Crash.gif">

<img src="https://github.com/danimatasd/MUJOCO-AIDL/blob/main/misc/Video_Anymal_C_Step_Crash_and_Recovery.gif">

<br/>

<img src="./assets/anybiotics-reward.jpeg" >

<img src="./assets/anybiotics-entropy.jpeg">

<img src="./assets/parameters-tuning-anybiotics.jpeg">

<img src="./assets/parameters-correlation-anybiotics.jpeg">

### Conclusions <a name="conclusions2"></a>

Based on the given information, we draw a series of conclusions:

1. The training algorithm being used is Proximal Policy Optimization (PPO), which is a reinforcement learning algorithm that has been shown to be effective for training robotic control tasks.

2. The algorithm improved the model's walking performance over time, as evidenced by the videos, with a maximum reward of 5925.51 achieved during training.

3. The entropy started at 17.025 and went down to 14.564 in 20.3 hours, which suggests that the policy became more deterministic as the training progressed.

## Future Work <a name="future-work"></a>

## Computational Resources <a name="comp-res"></a>

As the neural networks that we used were pretty small, we didn't find ourselves in the need of using GPUs for the training of our models (we would have spent more time passing the parameters to the GPU than doing the actual calculations in the CPU). 

To add on that, MuJoCo runs only on CPU and has no GPU support currently, so the only use case for GPUs in our project has been the render and capture of the videos.

Given this situation, for the development of this project, we have mainly used Google Colab, and when we needed to do longer trainings, two laptops with better specs to speed up the process a little bit (and also don't get kicked out from the runtime).

Here are the specifications of the machines:

* Google Colab
    * Intel Xeon @ 2.20GHz
    * 12GB RAM
    * NVIDIA Tesla K80 12GB GDDR5

* Laptop #1
    * AMD Ryzen 7 5800H @ 3.20GHz
    * 16GB RAM
    * NVIDIA GeForce RTX 3070 8GB GDDR6

* Laptop #2
    * INTEL Core i7-9750H @ 2.60 GHz
    * 8GB RAM
    * NVIDIA GeForce GTX 1050 4GB GDDR5
