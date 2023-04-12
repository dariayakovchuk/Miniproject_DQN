import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from abc import ABC, abstractmethod
from typing import Tuple
from epidemic_env.env import Env


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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
