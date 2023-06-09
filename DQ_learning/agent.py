import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from abc import ABC, abstractmethod
from typing import Tuple
from epidemic_env.env import Env


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent(ABC):
    """Implements acting and learning. (Abstract class, for implementations see DQNAgent and NaiveAgent).

    Args:
        ABC (_type_): _description_

    Returns:
        _type_: _description_
    """
    @abstractmethod
    def __init__(self,  env, *args, **kwargs):
        """
        Args:
            env (_type_): the simulation environment
        """
        
    @abstractmethod
    def load_model(self, savepath:str):
        """Loads weights from a file.

        Args:
            savepath (str): path at which weights are saved.
        """
        
    @abstractmethod
    def save_model(self, savepath:str):
        """Saves weights to a specified path

        Args:
            savepath (str): the path
        """
        
    @abstractmethod
    def optimize_model(self)->float:
        """Perform one optimization step.

        Returns:
            float: the loss
        """
    
    @abstractmethod
    def reset():
        """Resets the agent's inner state
        """
        
    @abstractmethod 
    def act(self, obs:torch.Tensor)->Tuple[int, float]:
        """Selects an action based on an observation.

        Args:
            obs (torch.Tensor): an observation

        Returns:
            Tuple[int, float]: the selected action (as an int) and associated Q/V-value as a float
        """


class DQNAgent(Agent):
    def __init__(self,  env:Env,
                 model:torch.nn,
                 criterion=nn.SmoothL1Loss(),
                 lr:float=5e-3,
                 epsilon:float=0.7,
                 gamma:float=0.9,
                 buffer_size:int=20000,
                 batch_size:int=2048,
                 n_episodes:int=500, 
                 epsilon_min:float=0.3)->None:

        self.env = env

        model_params = {
            'n_observations': len(env.observation_space.sample().flatten()),
            'n_actions': env.action_space.n,
        }
        
        self.policy_net = model(**model_params)
        self.target_net = model(**model_params)
        # self.target_net.load_state_dict(self.policy_net.state_dict())

        self.criterion = criterion
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.memory = ReplayMemory(buffer_size)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.lr = lr
        self.T_max = n_episodes

    def load_model(self, savepath:str): 
        self.policy_net.load_state_dict(torch.load(savepath))

    def save_model(self, savepath:str):
        torch.save(self.policy_net.state_dict(), savepath)

    def optimize_model(self, update = True):
        if len(self.memory) < self.batch_size:
                return np.double(0)
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if update: 
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        else: 
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss

    def reset(self):
        pass

    def act(self, state, time, decreasing = False):
        epsilon = self.epsilon
        if decreasing:
            epsilon = max(self.epsilon * (self.T_max - time) / self.T_max, self.epsilon_min)
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        
        
        
class DQNAgent_Factorize(Agent):
    def __init__(self,  env:Env,
                 model:torch.nn,
                 criterion=nn.SmoothL1Loss(),
                 lr:float=5e-3,
                 epsilon:float=0.7,
                 gamma:float=0.9,
                 buffer_size:int=20000,
                 batch_size:int=2048,
                 n_episodes:int=500, 
                 epsilon_min:float=0.3)->None:

        self.env = env
        
        model_params = {
            'n_observations': len(env.observation_space.sample().flatten()),
            'n_actions': 4*2,
        }
        
        self.policy_net = model(**model_params)
        self.target_net = model(**model_params)
        # self.target_net.load_state_dict(self.policy_net.state_dict())

        self.criterion = criterion
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.memory = ReplayMemory(buffer_size)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.lr = lr
        self.T_max = n_episodes

    def load_model(self, savepath:str): 
        self.policy_net.load_state_dict(torch.load(savepath))

    def save_model(self, savepath:str):
        torch.save(self.policy_net.state_dict(), savepath)

    def optimize_model(self, update = True):
        if len(self.memory) < self.batch_size:
                return np.double(0)
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        action_batch = torch.cat([e for e in batch.action], axis=0)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = torch.sum(self.policy_net(state_batch).gather(2, action_batch.unsqueeze(2)), axis=1)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if update: 
            next_state_values[non_final_mask] = torch.sum(self.target_net(non_final_next_states).max(2)[0], axis=1)
        else: 
            with torch.no_grad():
                next_state_values[non_final_mask] = torch.sum(self.target_net(non_final_next_states).max(2)[0], axis=1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss

    def reset(self):
        pass

    def act(self, state, time, decreasing = False):
        epsilon = self.epsilon
        if decreasing:
            epsilon = max(self.epsilon * (self.T_max - time) / self.T_max, self.epsilon_min)
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                output = torch.argmax(self.policy_net(state)[0], axis = 1)
                return output.to(dtype=torch.int64)
        else:
            action = self.env.action_space.sample()
            return torch.tensor(action, device=self.device).to(dtype=torch.int64)