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

PROJECT = "AIDL-PPO-BD-SWEEP"
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
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,act_len))
        self.critic = nn.Sequential(
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,1))

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
      path=f'./video_{episode}.mp4'
      media.write_video(path, self.frames, fps=self.FRAMERATE)
      
transition = np.dtype([('s', np.float64, (37,)), ('a', np.float64, (12,)), ('a_logp', np.float64, (12,)),
                       ('r', np.float64), ('s_', np.float64, (37,))])


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
    action_var = torch.full((12,), action_std * action_std)
    cov_mat = torch.diag(action_var).unsqueeze(dim=0)

    with torch.no_grad():
        target_v = r + gamma * policy(s_)[1]
        adv = target_v - policy(s)[1]

    for _ in range(ppo_epoch):
        for index in BatchSampler(SubsetRandomSampler(range(memory.buffer_capacity)), batch_size, False):
            probs, _ = policy(s[index])
            dist = MultivariateNormal(probs,cov_mat)
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

def test(action_std ,env, policy, episode, render=False):
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

    print(f'Closing Test episode after {counter} steps')
    env.close(episode)
    wandb.log({"Video eval": wandb.Video(f'./video_{episode}.mp4', fps=4, format="mp4")})

    #Ploting Episode Reward
    plt.plot(time_list, reward_list)
    plt.plot(time_list, cumulative_reward_list)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Reward')
    plt.title('Episode instant and cumulative Reward')

    wandb.log({"Reward eval": plt})

    return ep_reward

sweep_configuration = {
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
          "min": 0.00001
        },
        "ppo_epoch": {
          "distribution": "int_uniform",
          "max": 30,
          "min": 5
        },
        "c1": {
          "distribution": "int_uniform",
          "max": 3.,
          "min": 1.
        },
        "c2": {
          "distribution": "uniform",
          "max": 0.05,
          "min": 0.005
        },
        "replay_size": {
          "distribution": "int_uniform",
          "max": 8000.,
          "min": 3200.
        },
        "std_init": {
          "distribution": "uniform",
          "max": 2.,
          "min": 0.5
        },
        "std_min": {
          "distribution": "uniform",
          "max": 0.8,
          "min": 0.1
        }
    }
  }

def train_sweep(is_sweep=True):
  
    hparams = {
      'gamma' : 0.99,
      'log_interval' : 1000,
      'num_episodes': 10000,
      'lr' : 1e-4,
      'clip_param': 0.1,
      'ppo_epoch': 15,
      'replay_size': 6400,
      'batch_size': 128,
      'c1': 2.,
      'c2': 0.015,
      'std_init': 1.5,
      'std_min': 0.95,
      }
    
    wandb.init(project=PROJECT)        

    if is_sweep:
      hparams.update(wandb.config)
      print('Params updated')
      
    # Create environment
    env = Env()
    #print(wandb.run.name)

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
    optimizer = torch.optim.Adam(policy.parameters(), lr=hparams['lr'])
    eps = np.finfo(np.float32).eps.item()
    memory = ReplayMemory(hparams['replay_size'])
    
    #policy = torch.load('./magic-sponge-79_20436_Reward-3049.75_policy.pt')
    #optimizer = torch.load('./magic-sponge-79_20436_Reward-3049.75_optimizer.pt')

    action_std_decay = -(hparams['std_min']-hparams['std_init'])*hparams['log_interval']/hparams['num_episodes']
    action_std_init = hparams['std_init']
    action_std = action_std_init

    # Training loop
    print("Target reward: 6000")
    running_reward = -100
    saving_reward = 0
    for i_episode in range(hparams['num_episodes']):
        # Collect experience
        state, ep_reward, done = env.reset(), 0, False
        counter=0

        while not done:  # Don't infinite loop while learning
            action, a_logp, state_value = policy.compute_action(state, action_std)
            next_state, reward, done = env.step(action, render=False)

            if memory.store((state, action, a_logp, reward, next_state)):
                policy_loss, value_loss, avg_entropy, ratio = train(policy, optimizer, memory, hparams, action_std)
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

            counter = counter +1

            if done:
                break

        # Update running reward
        running_reward = round(0.05 * ep_reward + (1 - 0.05) * running_reward,2)
        
        if i_episode % hparams['log_interval'] == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}\tAction standard deviation: {action_std:.5f}')
            print(f'We have trained for {counter} steps')
            action_std = action_std - action_std_decay
            action_std = round(action_std, 5)
            ep_reward = test(action_std, env, policy,i_episode)
            print(f'Video reward: {ep_reward}')
        
        if running_reward > 2500:
            if running_reward > saving_reward:
                saving_reward = running_reward
                torch.save(policy, f'./{wandb.run.name}_{i_episode}_Reward-{running_reward}_policy.pt')
                torch.save(optimizer, f'./{wandb.run.name}_{i_episode}_Reward-{running_reward}_optimizer.pt')
                wandb.save(f'./{wandb.run.name}_{i_episode}_Reward-{running_reward}_policy.pt')
                wandb.save(f'./{wandb.run.name}_{i_episode}_Reward-{running_reward}_optimizer.pt')
                print(f'Policy and Optimizer have been saved')

        if running_reward > 6000:
            print("Solved!")
            torch.save(policy, f'./{wandb.run.name}_{i_episode}_Reward-{running_reward}_policy.pt')
            torch.save(optimizer, f'./{wandb.run.name}_{i_episode}_Reward-{running_reward}_optimizer.pt')
            wandb.save(f'./{wandb.run.name}_{i_episode}_Reward-{running_reward}_policy.pt')
            wandb.save(f'./{wandb.run.name}_{i_episode}_Reward-{running_reward}_optimizer.pt')
            break

    print(f"Finished training! Running reward is now {running_reward}")

#train_sweep(False)

sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT)
wandb.agent(sweep_id, function=train_sweep, count=50, project=PROJECT)