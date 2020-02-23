import gym
import torch
from collections import namedtuple

version = "1.0.0"

class Agent:
    def __init__(self, env, actor, noise = None, rend_wait = -1, rend_interval = -1 \
                 , frame = None, max_step = None, device = None):
        self.env = env
        if max_step is not None:
            self.env._max_episode_steps = max_step
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            
        self.actor = actor
        self.noise = noise
        self.device = device
        
        self.wait = rend_wait
        self.interval = rend_interval
        self.frame = frame
        
    def reset(self):
        self.env.close()
        self.env.reset()
        
    def render(self, epoch):
        if self.wait >= 0 and epoch < self.wait:
            return
        if self.interval >= 0 and epoch % self.interval == 0:
            rend = self.env.render("rgb_array")
            if self.frame is not None:
                self.frame.append(self.env.render("rgb_array"))
        
    def episode(self, epoch):
        self.obs = self.env.reset()
        self.render(epoch)
            
        while True:
            with torch.no_grad():
                act_v = self.actor(torch.FloatTensor(self.obs).to(self.device)).cpu().numpy()
                if self.noise is not None:
                    act_v += self.noise.get_noise()
                if self.env.action_space.shape:
                    act_v = act_v.clip(self.env.action_space.low, self.env.action_space.high)
                act = self.actor.get_action(act_v)

            next_obs, rew, done, etc = self.env.step(act)
            self.render(epoch)

            obs = self.obs
            self.obs = next_obs

            yield obs, act_v, act, next_obs, rew, done, etc
            if done:
                break
                
import numpy as np
import math

class NoiseMaker():
    def __init__(self, action_size, n_type = None, param = None, decay = True):
        self.action_size = action_size
        self.state = np.zeros(action_size, dtype=np.float32)
        self.count = 0
        self.decay = decay
        if n_type is None:
            n_type = "normal"
        self.type = n_type
        
        if param is None:
            self.param = {
                "start": 0.9,
                "end":0.02,
                "decay": 2000
            }
            if n_type =="ou":
                self.param["ou_mu"] = 0.0
                self.param["ou_th"] = 0.15
                self.param["ou_sig"] = 0.2
        else:
            self.param = param
            
    def get_noise(self):
        eps = self.param["end"] + (self.param["start"] - self.param["end"]) \
                * math.exp(-1*self.count/ self.param["decay"])
        
        noise = np.random.normal(size=self.action_size)
        if self.type == "ou":
            self.state += self.param["ou_th"] * (self.param["ou_mu"] - self.state) \
                        + self.param["ou_sig"] * noise
            noise = self.state
        if not self.decay:
            eps = 1
        self.count += 1
            
        return noise * eps
    
import collections
import random

class Replay:
    def __init__(self, size):
        self.memory = collections.deque(maxlen = size)
        
    def push(self, data):
        self.memory.append(data)
        
    def prepare(self, env):
        pass
        
    def sample(self, size):
        if len(self.memory) >= size:
            return random.sample(self.memory, size)
    
    def __len__(self):
        return len(self.memory)