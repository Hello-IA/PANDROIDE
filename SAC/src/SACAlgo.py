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

from Agent import *

# Create the SAC algorithm environment
class SACAlgo(EpochBasedAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        # We need an actor
        self.actor = DiscretePolicy(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )

        # Builds the critics
        self.critic_1 = DiscreteQAgent(
            obs_size,
            cfg.algorithm.architecture.critic_hidden_size,
            act_size,
        ).with_prefix("critic-1/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix(
            "target-critic-1/"
        )

        self.critic_2 = DiscreteQAgent(
            obs_size,
            cfg.algorithm.architecture.critic_hidden_size,
            act_size,
        ).with_prefix("critic-2/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix(
            "target-critic-2/"
        )

        # Train and evaluation policies
        self.train_policy = self.actor
        #self.eval_policy = KWAgentWrapper(self.actor, stochastic=False)
        self.eval_policy = Agents( #Permet de pouvoir interroger l'actor et le critic
            KWAgentWrapper(self.actor, stochastic=False),
            self.critic_1
        )

    def evaluate(self, force=False):
        """Evaluate the current policy `self.eval_policy`

        Evaluation is conducted every `cfg.algorithm.eval_interval` steps, and
        we keep a copy of the best agent so far in `self.best_policy`

        Returns True if the current policy is the best so far
        """
        #print("IN EVALUATE")
        global all_taux_accord, steps_evaluation

        if force or ((self.nb_steps - self.last_eval_step) > self.cfg.algorithm.eval_interval):
            #print(f"{self.nb_steps=}")
            self.last_eval_step = self.nb_steps
            eval_workspace = Workspace()
            
            #print("Avant eval")
            self.eval_agent(eval_workspace, t=0, stop_variable="env/done")
            #print("Apres eval")
            
            #for key in eval_workspace.variables.keys():
                #print(key,"-> shape:",eval_workspace[key].shape)
            
            action_probs = eval_workspace["action_probs"]
            q_value = eval_workspace["critic-1/q_value"]

            # size-> (n_step, n_env)
            actor_action = action_probs.argmax(dim=-1) #on recupere l'action ayant la plus grande proba
            critic_action = q_value.argmax(dim=-1) #on recupere l'action ayant la plus grande Qvalue

            accord = actor_action == critic_action #matrice qui vaut True si actor_action[i,j] == critic_action[i,j], False sinon
            #print(f"{accord=}")

            total = accord.shape[0] * accord.shape[1]
            taux_accord = accord.sum().item()/total
            #self.logger.add_log("taux_accord", torch.tensor(taux_accord), self.nb_steps)
            #all_taux_accord[nbrun].append(taux_accord)
            #steps_evaluation[nbrun].append(self.nb_steps)
            #print(f"{taux_accord=}")

            rewards = eval_workspace["env/cumulated_reward"][-1]
    
            return (self.register_evaluation(rewards), torch.mean(rewards).item())