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

import numpy as np
from random import randint
from collections import namedtuple

import mujoco_py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# if gpu is to be used

# Starting a fake screen in the background
# in order to render videos

from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1024,768),color_depth=24)
virtual_display.start()

# Utility to get video file from directory
def get_video_filename(dir="video"):
  glob_mp4 = os.path.join(dir, "*.mp4") 
  mp4list = glob.glob(glob_mp4)
  assert len(mp4list) > 0, "couldnt find video files"
  return mp4list[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

PROJECT = "AIDL-PPO-HALFCHEETAH-VIDEO"

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
    
def test( action_std, env, policy):
    state, ep_reward, done = env.reset(), 0, False
    while not done:
        action, _, _ = policy.compute_action(state, action_std)
        state, reward, done, _ = env.step(action)
        ep_reward += reward
    env.close()
    mp4 = get_video_filename()
    wandb.log({"Video eval": wandb.Video(mp4, fps=4, format="mp4")})
    return ep_reward

# Create environment
env_name = "HalfCheetah-v3"
env = gym.make(env_name)
env = gym.wrappers.RecordVideo(env, "./video" )

# Get number of actions from gym action space
n_inputs = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

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
    'std_init': 1.0, #This value should be the Action_std that was used on that episode in particular taken from the log from the train run
    'video_interval': 200
}

#We should use the same seed as in the training but it shouldnt affect too much with the pretrained model.
seed=0
env.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#The action_std will be directly 
action_std =  hparams['std_init']
wandb.finish()

wandb.init(project=PROJECT, config=hparams)
wandb.run.name = 'ppo_halfcheetah_test_video'

#Retrieving the model collected from the training colab, important that the model is saved as model.pt so the path is correct.
policy = torch.load("/content/model.pt")

#Apply the test function to try the model with a single episode and save the video on wandb
ep_reward = test(action_std, env, policy)

wandb.finish()