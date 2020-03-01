import gym
import torch
from collections import namedtuple

version = "1.0.0"

StepInfo = namedtuple("StepInfo", ("obs", "act_v", "act", "last_obs", "rew", "done", "etc", "n"))

class Agent:
    def __init__(self, env, actor, noise = None, rend_wait = -1, rend_interval = -1                  , frame = None, max_step = None, device = None):
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
        
        self.step_set = False
        self.n_step = 1
        self.gamma = 0.99
        
    def set_n_step(self, n, gamma):
        self.step_set = True
        self.n_step = n
        self.gamma= gamma
        
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
        assert self.step_set
        buffer = []
        self.obs = self.env.reset()
        self.render(epoch)

        total_rew = 0
        while True:
            with torch.no_grad():
                act_v = self.actor(torch.FloatTensor([self.obs])                                   .to(self.device)).cpu().squeeze().numpy()
                if self.noise is not None:
                    act_v += self.noise.get_noise()
                if self.env.action_space.shape:
                    act_v = act_v.clip(self.env.action_space.low, self.env.action_space.high)
                act = self.actor.get_action(act_v)

            next_obs, rew, done, etc = self.env.step(act)
            total_rew += rew
            self.render(epoch)

            obs = self.obs
            self.obs = next_obs

            buffer.append(StepInfo(obs, act_v, act, next_obs, rew, done, etc, -1))
            if done:
                break
                
            if len(buffer) < self.n_step:
                continue
                
            yield self.unroll_step(buffer)
            buffer.pop(0)
            
        while len(buffer):
            yield self.unroll_step(buffer)
            buffer.pop(0)
        print(epoch, "%.5f"%total_rew, end=' ')
        return

    def unroll_step(self, buffer):
        assert len(buffer)
        
        rews = list(map(lambda b:b.rew, buffer))
        rews.reverse()
        rew_sum = 0

        for r in rews:
            rew_sum*=self.gamma
            rew_sum+=r
            
        done = buffer[-1].done if len(buffer) == self.n_step else True
        return StepInfo(buffer[0].obs, buffer[0].act_v, buffer[0].act, buffer[-1].last_obs, rew_sum, done, buffer[0].etc, len(buffer))

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
        eps = self.param["end"] + (self.param["start"] - self.param["end"]) * math.exp(-1*self.count/ self.param["decay"])
        
        noise = np.random.normal(size=self.action_size)
        if self.type == "ou":
            self.state += self.param["ou_th"] * (self.param["ou_mu"] - self.state) + self.param["ou_sig"] * noise
            noise = self.state
        if not self.decay:
            eps = 1
        self.count += 1
            
        return noise * eps

import collections
import random
import numpy as np
import math

class Replay:
    def __init__(self, size, prio = False, alph = 0.6, beta = 0.4):
        self.memory = collections.deque(maxlen = size)
        self.size = size
        self.priorities = collections.deque(maxlen = size)
        self.prio = prio
        self.alph = alph if prio else 0
        self.beta = beta if prio else 0
        self.count = 0
        
    def push(self, data):
        self.memory.append(data)
        self.count += 1
        if self.prio:
            max_prio = np.array(self.priorities).max() if len(self.priorities) else 1.
            self.priorities.append(max_prio)
        
    def prepare(self, env):
        pass
        
    def sample(self, size):
        if self.prio:
            probs = np.array(self.priorities, dtype=np.float32)
            min_prio = np.array(self.priorities).min()
            if min_prio < 1:
                probs /= min_prio
            probs = probs ** self.alph
            probs /= probs.sum()
        else:
            probs = np.ones(len(self),) / len(self)
        
        indices = np.random.choice(len(self), size, p=probs)
        sample = [self.memory[idx] for idx in indices]
        
        beta = 1. + (self.beta - 1.) * math.exp(-1 * self.count / self.size)
        weights = (len(self) * probs[indices]) ** (-beta)
        weights /= weights.sum()
        
        return sample, indices, weights
        
    def update_priorities(self, indices, prios):
        if not self.prio:
            return
        prios += 1e-6
        for idx, prio in zip(indices,prios):
            self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.memory)