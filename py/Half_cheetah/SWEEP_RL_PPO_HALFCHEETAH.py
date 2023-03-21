import base64
import glob
import io
import os
import math
import timeit
import warnings

from IPython.display import HTML
from IPython.display import display

import gym
import wandb
import random
import mujoco_py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from random import randint
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# starting a fake screen in the background
#  in order to render videos
# Take in account that in Windows OS this doesn't work because there is no xvfb.

from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1024,768),color_depth=24)
virtual_display.start()


# utility to get video file from directory
def get_video_filename(dir="video"):
  glob_mp4 = os.path.join(dir, "*.mp4") 
  mp4list = glob.glob(glob_mp4)
  assert len(mp4list) > 0, "couldnt find video files"
  return mp4list[-1]

PROJECT = "AIDL-PPO-HALFCHEETAH"
wandb.login()

class Agent(nn.Module):
    def __init__(self, obs_len, act_len, action_std_init):
        super(Agent, self).__init__()
        
        self.obs_len = obs_len
        self.act_len = act_len

        self.mlp = nn.Sequential(
            nn.Linear(obs_len, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
        )

        self.actor = nn.Linear(128, act_len)
        self.critic = nn.Linear(128, 1)
        

    def forward(self, state):
        out = self.mlp(state)
        action_scores = self.actor(out)
        state_value = self.critic(out)
        return torch.tanh(action_scores), state_value

    def compute_action(self, state, action_std):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self(state)

        action_var = torch.full((self.act_len,), action_std * action_std)
        cov_mat = torch.diag(action_var).unsqueeze(dim=0)
      
        m = torch.distributions.multivariate_normal.MultivariateNormal(probs, cov_mat)
        
        action = m.sample()
        
        action_clamped = action.clamp(-1.0, 1.0)
      
        return action_clamped.detach().numpy(), m.log_prob(action_clamped).detach().numpy(), state_value.detach()
    
transition = np.dtype([('s', np.float64, (17,)), ('a', np.float64, (6,)), ('a_logp', np.float64, (6,)),
                    ('r', np.float64), ('s_', np.float64, (17,))])

class ReplayMemory():
    def __init__(self, capacity):
        self.buffer_capacity = capacity
        self.buffer = np.empty(capacity, dtype=transition)
        self.counter = 0

    # Stores a transition and returns True or False depending on whether the buffer is full or not
    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False
        
def train(policy, optimizer, memory, hparams, action_std):

    gamma = hparams['gamma']
    ppo_epoch = hparams['ppo_epoch']
    batch_size = hparams['batch_size']
    clip_param = hparams['clip_param']
    c1 = hparams['c1']
    c2 = hparams['c2']


    s = torch.tensor(memory.buffer['s'], dtype=torch.float)
    a = torch.tensor(memory.buffer['a'], dtype=torch.float)
    r = torch.tensor(memory.buffer['r'], dtype=torch.float).view(-1, 1)
    s_ = torch.tensor(memory.buffer['s_'], dtype=torch.float)

    old_a_logp = torch.tensor(memory.buffer['a_logp'], dtype=torch.float).view(-1, 1)
    action_var = torch.full((6,), action_std*action_std)
    cov_mat = torch.diag(action_var).unsqueeze(dim=0)

    with torch.no_grad():
        target_v = r + gamma * policy(s_)[1]
        adv = target_v - policy(s)[1]

    for _ in range(ppo_epoch):
        for index in BatchSampler(SubsetRandomSampler(range(memory.buffer_capacity)), batch_size, False):
            probs, _ = policy(s[index])
            dist = MultivariateNormal(probs, cov_mat)
            entropy = dist.entropy()
            
            a_logp = dist.log_prob(a[index]).unsqueeze(dim=1)


            ratio = torch.exp(a_logp-old_a_logp[index])

            surr1 = ratio * adv[index]

            surr2 = torch.clamp(ratio,1-clip_param,1+clip_param) * adv[index]

            policy_loss = torch.min(surr1, surr2).mean()
            value_loss = F.smooth_l1_loss(policy(s[index])[1], target_v[index])
            entropy = entropy.mean()

            loss = -policy_loss+c1*value_loss-c2*entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return -policy_loss.item(), value_loss.item(), entropy.item(), ratio.mean().item()

#The test function does a episode on the enviroment of the model and renders and saves a video in Wandb

def test( action_std, env, policy, render=False):
    state, ep_reward, done = env.reset(), 0, False
    while not done:
        action, _, _ = policy.compute_action(state, action_std)
        state, reward, done, _ = env.step(action)
        ep_reward += reward
    env.close()
    mp4 = get_video_filename()
    wandb.log({"Video eval": wandb.Video(mp4, fps=4, format="mp4")})
    return ep_reward

hparams = {
    'gamma' : 0.99,
    'log_interval' : 10,
    'num_episodes': 50000,
    'lr' : 1e-4,
    'clip_param': 0.1,
    'ppo_epoch': 45,
    'replay_size': 600,
    'batch_size': 128,
    'c1': 3.,
    'c2': 0.01,
    'std_init': 1.0,
    'video_interval': 200
}

# Create environment
env_name = "HalfCheetah-v3"
env = gym.make(env_name)

#Delete the comment of the wrapper clause so you can save videos of the enviroment.
#Take in account that rendering videos makes the training longer.

#env = gym.wrappers.RecordVideo(env, "./video" )

# Get number of actions from gym action space
n_inputs = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

def Halfcheetah():
    # Fix random seed (for reproducibility)
    seed=0
    env.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Initialize wandb run
    wandb.finish() # execute to avoid overlapping runnings (advice: later remove duplicates in wandb)
    wandb.init(project=PROJECT, config=hparams)
   

    #Initialize the action_std, decay and init for the covariance matrix of the distribution
    action_std_decay = ((hparams['log_interval']/hparams['num_episodes'])*hparams['std_init'])
    action_std_init = hparams['std_init'] + (action_std_decay) #We add a decay before hand because on the episode 0 it will decay with the log of that episode.
    action_std = action_std_init

    # Create policy and optimizer
    policy = Agent(n_inputs, n_actions, action_std_init)
    optimizer = torch.optim.Adam(policy.parameters(), lr=hparams['lr'])

    memory = ReplayMemory(hparams['replay_size'])

    # Training loop
    print("Target reward: {}".format(env.spec.reward_threshold))
    running_reward = -100
    ep_rew_history_reinforce = []
    for i_episode in range(hparams['num_episodes']):
        # Collect experience
   
        state = env.reset() 
        ep_reward, done  = 0, False

        while not done:  # Don't infinite loop while learning
            action, a_logp, state_value = policy.compute_action(state, action_std)
            next_state, reward, done, _ = env.step(action)

            if memory.store((state, action, a_logp, reward, next_state)):
                policy_loss, value_loss, avg_entropy, ratio = train(policy, optimizer, memory, hparams, action_std)
                wandb.log(
                    {
                    'policy_loss': policy_loss,
                    'value_loss': value_loss,
                    'running_reward': running_reward,
                    'mean_entropy': avg_entropy,
                    'ratio': ratio
                    })


            state = next_state

            ep_reward += reward

            if done:
                break

        # Update running reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    
    
        ep_rew_history_reinforce.append((i_episode, ep_reward))
        if i_episode % hparams['log_interval'] == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')
            action_std = action_std - action_std_decay
            action_std = round(action_std, 5)
            #Empirical limit we found of the action_std where below 0.22 it will drop the entropy to a level where the agent stops working
            if action_std < 0.250:
                action_std = 0.250
            print(f'Action_std {action_std}')

        if running_reward > 3000.0:
            if running_reward > saving_reward:
                saving_reward = running_reward
                torch.save(policy, f'./{wandb.run.name}_{i_episode}_Reward-{running_reward}_policy.pt')
                torch.save(optimizer, f'./{wandb.run.name}_{i_episode}_Reward-{running_reward}_optimizer.pt')
                print(f'Policy and Optimizer have been saved')

    #Delete the comment of the if clause so you can save videos with the hyperparameter video_interval. 
    #Take in account that rendering videos makes the training longer.

        #if i_episode % hparams['video_interval'] == 0:
            #ep_reward = test(action_std,test_env, policy)  

        if running_reward > env.spec.reward_threshold:
            print("Solved!")
            break

    print(f"Finished training! Running reward is now {running_reward}")
    ep_reward = test(action_std,env, policy)  

    wandb.finish()

def train_sweep():
    hparams = {
    'gamma' : 0.99,
    'log_interval' : 20,
    'num_episodes': 2000,
    'lr' : 1e-4,
    'clip_param': 0.1,
    'ppo_epoch': 4,
    'replay_size': 500,
    'batch_size': 128,
    'c1': 3.,
    'c2': 0.01,
    'std_init': 1.0
    }

    run = wandb.init(PROJECT)
    hparams.update(wandb.config)
    
    # Create environment
    env_name = "HalfCheetah-v3"
    env = gym.make(env_name)

    # Fix random seed (for reproducibility)
    seed=0
    env.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Get number of actions from gym action space
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    #Initialize the action_std, decay and init for the covariance matrix of the distribution    
    action_std_decay = (hparams['log_interval']/hparams['num_episodes'])*hparams['std_init']
    action_std_init = hparams['std_init']
    action_std = action_std_init

    # Create policy and optimizer
    policy = Agent(n_inputs, n_actions, 1.0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=hparams['lr'])
    eps = np.finfo(np.float32).eps.item()
    memory = ReplayMemory(hparams['replay_size'])

    # Training loop
    print("Target reward: {}".format(env.spec.reward_threshold))
    running_reward = -100
    ep_rew_history_reinforce = []
    for i_episode in range(hparams['num_episodes']):
        # Collect experience
        state = env.reset()
        ep_reward, done =  0, False
        while not done:  # Don't infinite loop while learning
            action, a_logp, state_value = policy.compute_action(state, action_std)
            next_state, reward, done, _ = env.step(action)

            if memory.store((state, action, a_logp, reward, next_state)):
                policy_loss, value_loss, avg_entropy, ratio = train(policy, optimizer, memory, hparams,action_std)
                wandb.log(
                    {
                    'policy_loss': policy_loss,
                    'value_loss': value_loss,
                    'avg_reward': running_reward,
                    'avg_entropy': avg_entropy,
                    'ratio': ratio
                    })


            state = next_state

            ep_reward += reward

            if done:
                break

        # Update running reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        
        
        ep_rew_history_reinforce.append((i_episode, ep_reward))

        #Every log interval we also decay the action std to reduce the entropy and eventually converge on a solution.

        if i_episode % hparams['log_interval'] == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}') 
            action_std = action_std - action_std_decay
            action_std = round(action_std, 5)
            print(f'Action_std {action_std}')

    print(f"Finished training! Running reward is now {running_reward}")

sweep_config = {
    "name": f"ppo_sweep_0",
    "method": 'bayes',
    "metric": {
        "name": "avg_reward",
        "goal": "maximize"
    },
    "parameters": {
        "lr": {
          "distribution": "uniform",
          "max": 0.001,
          "min": 0.000001
        },
        "ppo_epoch": {
          "distribution": "int_uniform",
          "max": 100,
          "min": 2
        },
        "c1": {
          "distribution": "int_uniform",
          "max": 3.,
          "min": 1.
        },
        "c2": {
          "distribution": "uniform",
          "max": 0.1,
          "min": 0.005
        },
        "replay_size": {
          "distribution": "int_uniform",
          "max": 1000.,
          "min": 256.
        }
  }
}        