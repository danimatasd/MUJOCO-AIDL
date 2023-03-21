import numpy as np
import mujoco

from typing import List
import mediapy as media
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

MUJOCO_STEPS = 5

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

    def close(self,episode,reward):
      path=f'./video_{episode}_reward_{reward}.mp4'
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

    env.close(num_video+1,ep_reward)
    #wandb.log({"Video eval": wandb.Video(f'./video_{num_video+1}_reward_{ep_reward}.mp4', fps=4, format="mp4")})

    #Ploting Episode Reward
    plt.plot(time_list, reward_list)
    plt.plot(time_list, cumulative_reward_list)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Reward')
    plt.title('Episode instant and cumulative Reward')

    return ep_reward

def make_video(num_videos=10, std_init=0.85820):
  
  env = Env()
  
  # Fix random seed (for reproducibility)
  seed=0
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  # Set number of actions from gym action space
  n_inputs = 37
  n_actions = 12

  # Create policy and optimizer
  policy = Agent(n_inputs, n_actions)
    
  policy = torch.load('./Policy_Reward_6036.56.pt')

  action_std = std_init
  
  for i_video in range(num_videos): 
    ep_reward = test(action_std, env, policy,i_video)
    print(f'Video #{i_video+1} reward: {ep_reward}')

make_video(5)