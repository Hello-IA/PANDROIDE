try:
    from easypip import easyimport
except ModuleNotFoundError:
    from subprocess import run

    assert (
        run(["pip", "install", "easypip"]).returncode == 0
    ), "Could not install easypip"
    from easypip import easyimport

easyimport("swig")
easyimport("bbrl_utils>=0.5").setup()


import optuna
import copy
import os

import torch
import torch.nn as nn
from bbrl.workspace import Workspace
from bbrl.agents import Agent, Agents, TemporalAgent, KWAgentWrapper
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import build_mlp, setup_optimizer, soft_update_params
from bbrl_utils.notebook import setup_tensorboard
from omegaconf import OmegaConf
from torch.distributions import (
    Normal,
    TransformedDistribution,
)
import bbrl_gymnasium  # noqa: F401
from torch.distributions import Categorical
import matplotlib.pyplot as plt



class DiscreteQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        q_values = self.model(obs)
        self.set((f"{self.prefix}q_value", t), q_values)
        
        
class DiscretePolicy(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        """Creates a new Squashed Gaussian actor

        :param state_dim: The dimension of the state space
        :param hidden_layers: Hidden layer sizes
        :param action_dim: The dimension of the action space
        :param min_std: The minimum standard deviation, defaults to 1e-4
        """
        super().__init__()
        backbone_dim = [state_dim] + list(hidden_layers)
        self.layers = build_mlp(backbone_dim, activation=nn.ReLU())
        self.model = nn.Sequential(*self.layers)
        self.last_layer = nn.Linear(hidden_layers[-1], action_dim)
        
    def get_distribution(self, obs):
        scores = self.last_layer(self.model(obs))
        probs = torch.softmax(scores, dim=-1)
        
        return torch.distributions.Categorical(probs), scores, probs

    def forward(self, t, stochastic = False):
        """Computes the action a_t from a categorical distribution

        :param stochastic: True to use sampling, False to use the max probability action
        """
        obs = self.get(("env/env_obs", t))
        
        action_dist, _, action_probs  = self.get_distribution(obs)

        if stochastic:
            action = action_dist.sample()
        else:
            action = action_probs.argmax(1)

        log_prob = torch.log(action_probs)

        self.set(("action", t), action)
        self.set(("action_logprobs", t), log_prob)
        self.set(("action_probs", t), action_probs)
        