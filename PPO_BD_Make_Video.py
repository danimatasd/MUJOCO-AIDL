import numpy as np
import mujoco
import os
import wandb

from typing import Callable, Optional, Union, List
import scipy.linalg
import mediapy as media
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT = "AIDL-PPO-BD-SWEEP_BNN"
MUJOCO_STEPS = 5

wandb.login()

class Agent(nn.Module):
    def __init__(self, obs_len, act_len):
        super(Agent, self).__init__()
        
        self.obs_len = obs_len
        self.act_len = act_len

        self.mlp = nn.Sequential(
            nn.Linear(obs_len, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        self.actor = nn.Sequential(
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,act_len))
        self.critic = nn.Sequential(
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,1))

    def forward(self, state):
        out = self.mlp(state)
        action_scores = self.actor(out)
        state_value = self.critic(out)
        return action_scores, state_value

    def compute_action(self, state, action_std):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self(state)
        probs = torch.tanh(probs)
        #probs, action_std, state_value = self(state)

        action_var = torch.full((self.act_len,), action_std * action_std)
        cov_mat = torch.diag(action_var).unsqueeze(dim=0)
      
        m = torch.distributions.multivariate_normal.MultivariateNormal(probs, cov_mat)
        
        action = m.sample()

        action_clamped = torch.tanh(action)

        action_clamped[0][0] = action_clamped[0][0]*0.6-0.1
        action_clamped[0][3] = action_clamped[0][3]*0.6+0.1
        action_clamped[0][6] = action_clamped[0][6]*0.6-0.1
        action_clamped[0][9] = action_clamped[0][9]*0.6+0.1
      
        return action_clamped.detach().numpy(), m.log_prob(action_clamped).detach().numpy(), state_value.detach()
    
class Env():
    def __init__(self):
      self.model = mujoco.MjModel.from_xml_path("./anybotics_anymal_c/scene.xml")
      self.data = mujoco.MjData(self.model)
      self.renderer=mujoco.Renderer(self.model)
      mujoco.mj_kinematics(self.model, self.data)
      mujoco.mj_forward(self.model, self.data)
      self.FRAMERATE = 60 #Hz
      self.DURATION = 8 #Seconds
      self.TIMESTEP = 0.002 # 0.002 By Default
      self.done = False
      self.model.opt.timestep = self.TIMESTEP
      # Make a new camera, move it to a closer distance.
      self.camera = mujoco.MjvCamera()
      mujoco.mjv_defaultFreeCamera(self.model, self.camera)
      self.camera.distance = 5
      self.frames=[]

    def reset(self):
      # Simulate and save data
      mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
      self.data.ctrl=np.zeros(12)
      state= np.array(self.data.qpos.copy())
      state= np.append(state, self.data.qvel.copy())
      self.frames.clear()
      return state

    def step(self, action, render=False):
      self.done=False
      reward = 0
      self.data.ctrl=action
      for i in range(MUJOCO_STEPS):
        mujoco.mj_step(self.model, self.data)
        reward = reward + self.data.qvel[0] + (self.data.qpos[2]-0.5)
        if render and (len(self.frames) < self.data.time * self.FRAMERATE):
          self.camera.lookat = self.data.body('LH_SHANK').subtree_com
          self.renderer.update_scene(self.data, self.camera)
          pixels = self.renderer.render()
          self.frames.append(pixels.copy())

      state= np.array(self.data.qpos.copy())
      state= np.append(state, self.data.qvel.copy())
      if self.data.time > self.DURATION:
        self.done = True
      if self.data.qpos[2] < 0.3:
        self.done = True
        reward = reward - 100
      return state, reward, self.done

    def close(self,episode):
      path=f'./video_{episode+1}.mp4'
      media.write_video(path, self.frames, fps=self.FRAMERATE)

def test(action_std ,env, policy, num_video, render=False):
    state, ep_reward, done = env.reset(), 0, False
    counter=0
    reward_list=[]
    cumulative_reward_list=[]
    time_list=[]
    while not done:
        action, _, _ = policy.compute_action(state, action_std)
        state, reward, done = env.step(action, render=True)
        reward_list.append(reward)
        time_list.append(counter*0.002*MUJOCO_STEPS)
        ep_reward += reward
        cumulative_reward_list.append(ep_reward)
        counter = counter + 1

    print(f'Closing Video episode after {counter} steps')
    env.close(num_video+1)
    wandb.log({"Video eval": wandb.Video(f'./video_{num_video+1}.mp4', fps=4, format="mp4")})

    #Ploting Episode Reward
    plt.plot(time_list, reward_list)
    plt.plot(time_list, cumulative_reward_list)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Reward')
    plt.title('Episode instant and cumulative Reward')

    wandb.log({"Reward eval": plt})

    return ep_reward

def make_video():
  
  hparams = {
      'num_videos': 10,
      'std_init': 0.000000001,
      }
  
  wandb.init(project=PROJECT)
  
  env = Env()
  
  # Fix random seed (for reproducibility)
  seed=4
  #random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  # Get number of actions from gym action space
  n_inputs = 37
  n_actions = 12

  # Create policy and optimizer
  policy = Agent(n_inputs, n_actions)
    
  policy = torch.load('./skilled-sweep-1_9015_Reward-3180.37_policy.pt')
  optimizer = torch.load('./skilled-sweep-1_9015_Reward-3180.37_optimizer.pt')

  action_std = hparams['std_init']
  
  for i_video in range(hparams['num_videos']): 
    ep_reward = test(action_std, env, policy,i_video)
    print(f'Video #{i_video+1} reward: {ep_reward}')

make_video()