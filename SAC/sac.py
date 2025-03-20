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
    Independent,
    TransformedDistribution,
    TanhTransform,
)
import bbrl_gymnasium  # noqa: F401



class SquashedGaussianActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, min_std=1e-4):
        """Creates a new Squashed Gaussian actor

        :param state_dim: The dimension of the state space
        :param hidden_layers: Hidden layer sizes
        :param action_dim: The dimension of the action space
        :param min_std: The minimum standard deviation, defaults to 1e-4
        """
        super().__init__()
        self.min_std = min_std
        backbone_dim = [state_dim] + list(hidden_layers)
        self.layers = build_mlp(backbone_dim, activation=nn.ReLU())
        self.backbone = nn.Sequential(*self.layers)
        self.last_mean_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.last_std_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.softplus = nn.Softplus()
        
        # cache_size avoids numerical infinites or NaNs when
        # computing log probabilities
        self.tanh_transform = TanhTransform(cache_size=1)

    def normal_dist(self, obs: torch.Tensor):
        """Compute normal distribution given observation(s)"""
        
        backbone_output = self.backbone(obs)
        mean = self.last_mean_layer(backbone_output)
        std_out = self.last_std_layer(backbone_output)
        std = self.softplus(std_out) + self.min_std
        # Independent ensures that we have a multivariate
        # Gaussian with a diagonal covariance matrix (given as
        # a vector `std`)
        return Independent(Normal(mean, std), 1)

    def forward(self, t, stochastic=True):
        """Computes the action a_t and its log-probability p(a_t| s_t)

        :param stochastic: True when sampling
        """
        normal_dist = self.normal_dist(self.get(("env/env_obs", t)))
        action_dist = TransformedDistribution(normal_dist, [self.tanh_transform])
        if stochastic:
            # Uses the re-parametrization trick
            action = action_dist.rsample()
        else:
            # Directly uses the mode of the distribution
            action = self.tanh_transform(normal_dist.mode)

        log_prob = action_dist.log_prob(action)
        # This line allows to deepcopy the actor...
        self.tanh_transform._cached_x_y = [None, None]
        self.set(("action", t), action)
        self.set(("action_logprobs", t), log_prob)
        
        
class ContinuousQAgent(Agent):
    def __init__(self, state_dim: int, hidden_layers: list[int], action_dim: int):
        """Creates a new critic agent $Q(s, a)$

        :param state_dim: The number of dimensions for the observations
        :param hidden_layers: The list of hidden layers for the NN
        :param action_dim: The numer of dimensions for actions
        """
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        obs_act = torch.cat((obs, action), dim=1)
        q_value = self.model(obs_act).squeeze(-1)
        self.set((f"{self.prefix}q_value", t), q_value)
        
# Create the SAC algorithm environment
class SACAlgo(EpochBasedAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()
        assert (
            self.train_env.is_continuous_action()
        ), "SAC code dedicated to continuous actions"

        # We need an actor
        self.actor = SquashedGaussianActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )

        # Builds the critics
        self.critic_1 = ContinuousQAgent(
            obs_size,
            cfg.algorithm.architecture.critic_hidden_size,
            act_size,
        ).with_prefix("critic-1/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix(
            "target-critic-1/"
        )

        self.critic_2 = ContinuousQAgent(
            obs_size,
            cfg.algorithm.architecture.critic_hidden_size,
            act_size,
        ).with_prefix("critic-2/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix(
            "target-critic-2/"
        )

        # Train and evaluation policies
        self.train_policy = self.actor
        self.eval_policy = KWAgentWrapper(self.actor, stochastic=False)
        
def setup_entropy_optimizers(cfg):
    if cfg.algorithm.entropy_mode == "auto":
        # Note: we optimize the log of the entropy coef which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        # Comment and code taken from the SB3 version of SAC
        log_entropy_coef = nn.Parameter(
            torch.log(torch.ones(1) * cfg.algorithm.init_entropy_coef)
        )
        entropy_coef_optimizer = setup_optimizer(
            cfg.entropy_coef_optimizer, log_entropy_coef
        )
        return entropy_coef_optimizer, log_entropy_coef
    else:
        return None, None


def compute_critic_loss(
    cfg,
    reward: torch.Tensor,
    must_bootstrap: torch.Tensor,
    t_actor: TemporalAgent,
    t_q_agents: TemporalAgent,
    t_target_q_agents: TemporalAgent,
    rb_workspace: Workspace,
    ent_coef: torch.Tensor,
):
    r"""Computes the critic loss for a set of $S$ transition samples

    Args:
        cfg: The experimental configuration
        reward: Tensor (2xS) of rewards
        must_bootstrap: Tensor (2xS) of indicators
        t_actor: The actor agent
        t_q_agents: The critics
        t_target_q_agents: The target of the critics
        rb_workspace: The transition workspace
        ent_coef: The entropy coefficient $\alpha$

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The two critic losses (scalars)
    """

    # Replay the actor so we get the necessary statistics

    t_q_agents(rb_workspace, t=0, n_steps=1)
    with torch.no_grad():
        t_actor(rb_workspace, t=1, n_steps=1)
        
        action_logprobs = rb_workspace["action_logprobs"]
        t_target_q_agents(rb_workspace, t=1, n_steps=1)
        
    q_value_1, q_value_2, q_values_next_1, q_values_next_2 = rb_workspace["critic-1/q_value", "critic-2/q_value", "target-critic-1/q_value", "target-critic-2/q_value"]

    # Compute temporal difference

    q_values_next = torch.minimum(q_values_next_1[1], q_values_next_2[1])
    esperance_interne = q_values_next - ent_coef*action_logprobs[1]
    
    target = reward[1] + cfg.algorithm.discount_factor*esperance_interne*must_bootstrap.int()
    
    mse = nn.MSELoss()
    
    critic_loss_1 = mse(q_value_1[0], target)
    critic_loss_2 = mse(q_value_2[0], target)

    return critic_loss_1, critic_loss_2


def compute_actor_loss(
    ent_coef, t_actor: TemporalAgent, t_q_agents: TemporalAgent, rb_workspace: Workspace
):
    r"""
    Actor loss computation
    :param ent_coef: The entropy coefficient $\alpha$
    :param t_actor: The actor agent (temporal agent)
    :param t_q_agents: The critics (as temporal agent)
    :param rb_workspace: The replay buffer (2 time steps, $t$ and $t+1$)
    """

    # Recompute the action with the current actor (at $a_t$)

    t_actor(rb_workspace, t=0, n_steps=1)
    action_logprobs = rb_workspace["action_logprobs"]
    
    t_q_agents(rb_workspace, t=0, n_steps=1)
    q_values_1, q_values_2 = rb_workspace["critic-1/q_value", "critic-2/q_value"]


    # Compute Q-values

    current_q_values = torch.minimum(q_values_1, q_values_2)

    # Compute the actor loss
    
    actor_loss = action_logprobs[0]*ent_coef - current_q_values[0]



    return actor_loss.mean()


import numpy as np


def run_sac(sac: SACAlgo):
    cfg = sac.cfg
    logger = sac.logger


    # init_entropy_coef is the initial value of the entropy coef alpha.
    ent_coef = cfg.algorithm.init_entropy_coef
    tau = cfg.algorithm.tau_target

    # Creates the temporal actors
    t_actor = TemporalAgent(sac.train_policy)
    t_q_agents = TemporalAgent(Agents(sac.critic_1, sac.critic_2))
    t_target_q_agents = TemporalAgent(Agents(sac.target_critic_1, sac.target_critic_2))

    # Configure the optimizer
    actor_optimizer = setup_optimizer(cfg.actor_optimizer, sac.actor)
    critic_optimizer = setup_optimizer(cfg.critic_optimizer, sac.critic_1, sac.critic_2)
    entropy_coef_optimizer, log_entropy_coef = setup_entropy_optimizers(cfg)


    # If entropy_mode is not auto, the entropy coefficient ent_coef remains
    # fixed. Otherwise, computes the target entropy
    if cfg.algorithm.entropy_mode == "auto":
        # target_entropy is \mathcal{H}_0 in the SAC and aplications paper.
        target_entropy = -np.prod(sac.train_env.action_space.shape).astype(np.float32)

    # Loops over successive replay buffers
    for rb in sac.iter_replay_buffers():
        rb_workspace = rb.get_shuffled(sac.cfg.algorithm.batch_size)
        # Implement the SAC algorithm
        terminated, reward = rb_workspace["env/terminated", "env/reward"]
        # Critic update part #############################
        
        critic_optimizer.zero_grad()
        critic_loss_1, critic_loss_2 = compute_critic_loss(cfg, reward, ~terminated, t_actor, t_q_agents, t_target_q_agents, rb_workspace, ent_coef)
        
        logger.add_log("critic_loss_1", critic_loss_1, sac.nb_steps)
        logger.add_log("critic_loss_2", critic_loss_2, sac.nb_steps)
        critic_loss = critic_loss_1 + critic_loss_2
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            sac.critic_1.parameters(), cfg.algorithm.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            sac.critic_2.parameters(), cfg.algorithm.max_grad_norm
        )
        critic_optimizer.step()
        
        # Actor update part #############################

        actor_optimizer.zero_grad()
        actor_loss = compute_actor_loss(ent_coef, t_actor, t_q_agents, rb_workspace)
        sac.logger.add_log("actor_loss", actor_loss, sac.nb_steps)

        
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            sac.train_policy.parameters(), sac.cfg.algorithm.max_grad_norm
        )
        actor_optimizer.step()

        # Entropy optimizer part
        if entropy_coef_optimizer is not None:
            # See Eq. (17) of the SAC and Applications paper. The log
            # probabilities *must* have been computed when computing the actor
            # loss.
            action_logprobs_rb = rb_workspace["action_logprobs"].detach()
            entropy_coef_loss = -(
                log_entropy_coef.exp() * (action_logprobs_rb + target_entropy)
            ).mean()
            entropy_coef_optimizer.zero_grad()
            entropy_coef_loss.backward()
            entropy_coef_optimizer.step()
            logger.add_log("entropy_coef_loss", entropy_coef_loss, sac.nb_steps)
            logger.add_log("entropy_coef", torch.tensor(ent_coef), sac.nb_steps)

        ####################################################

        # Soft update of target q function
        soft_update_params(sac.critic_1, sac.target_critic_1, tau)
        soft_update_params(sac.critic_2, sac.target_critic_2, tau)

        sac.evaluate()
        
params = {
    "save_best": True,
    "base_dir": "${gym_env.env_name}/sac-S${algorithm.seed}_${current_time:}",
    "algorithm": {
        "seed": 1,
        "n_envs": 8,
        "n_steps": 32,
        "buffer_size": 1e6,
        "batch_size": 256,
        "max_grad_norm": 0.5,
        "nb_evals": 16,
        "eval_interval": 2_000,
        "learning_starts": 10_000,
        "max_epochs": 2_000,
        "discount_factor": 0.98,
        "entropy_mode": "auto",  # "auto" or "fixed"
        "init_entropy_coef": 2e-7,
        "tau_target": 0.05,
        "architecture": {
            "actor_hidden_size": [64, 64],
            "critic_hidden_size": [256, 256],
        },
    },
    "gym_env": {"env_name": "CartPoleContinuous-v1"},
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,
    },
    "entropy_coef_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,
    },
}

agents = SACAlgo(OmegaConf.create(params))
run_sac(agents)

# Visualize the best policy
agents.visualize_best()