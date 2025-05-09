# Prepare the environment
try:
    from easypip import easyimport
except ModuleNotFoundError:
    from subprocess import run

    assert (
        run(["pip", "install", "easypip"]).returncode == 0
    ), "Could not install easypip"
    from easypip import easyimport

easyimport("swig")
easyimport("bbrl_utils").setup(maze_mdp=True)

import os
import copy
import math
import bbrl_gymnasium  # noqa: F401
import torch
import torch.nn as nn
from bbrl import get_arguments, get_class

from bbrl.agents import Agent, Agents, TemporalAgent, PrintAgent, KWAgentWrapper
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import build_mlp, setup_optimizer, soft_update_params
from bbrl_utils.notebook import setup_tensorboard
from bbrl.visu.plot_policies import plot_policy
from omegaconf import OmegaConf


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
            #print("Stochastic")
            action = action_dist.sample()
        else:

            #print("Pas stochastic")
            action = action_probs.argmax(1)

        log_prob = torch.log(action_probs)

        self.set(("action", t), action)
        self.set(("action_logprobs", t), log_prob)
        self.set(("action_probs", t), action_probs)
        
        
from torch.distributions import Normal

class AddGaussianNoise(Agent):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        dist = Normal(act, self.sigma)
        action = dist.sample()
        self.set(("action", t), action)
        
        
class AddOUNoise(Agent):
    """
    Ornstein-Uhlenbeck process noise for actions as suggested by DDPG paper
    """

    def __init__(self, std_dev, theta=0.15, dt=1e-2):
        self.theta = theta
        self.std_dev = std_dev
        self.dt = dt
        self.x_prev = 0

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        x = (
            self.x_prev
            + self.theta * (act - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape)
        )
        self.x_prev = x
        self.set(("action", t), x)
        
        
# Defines the (Torch) mse loss function
# `mse(x, y)` computes $\|x-y\|^2$


def compute_critic_loss(cfg, reward: torch.Tensor, must_bootstrap: torch.Tensor, q_values: torch.Tensor, target_q_values: torch.Tensor, actions: torch.Tensor):
    """Compute the DDPG critic loss from a sample of transitions

    :param cfg: The configuration
    :param reward: The reward (shape 2xB)
    :param must_bootstrap: Must bootstrap flag (shape 2xB)
    :param q_values: The computed Q-values (shape 2xB)
    :param target_q_values: The Q-values computed by the target critic (shape 2xB)
    :return: the loss (a scalar)
    """
    #print(reward[:-1].squeeze().shape)
    
    next_q_max = target_q_values[1].mean(dim=1) 
    #print(next_q_max)
    target = reward[1].squeeze() + cfg.algorithm.discount_factor * next_q_max * must_bootstrap[1].int()  


    

    mse = nn.MSELoss()
    #print("q_values[0].squeeze(-1)",q_values[0].squeeze(-1).shape)
    #print("target", target.shape)
    #print("q_values[0].squeeze(-1)", q_values[0].squeeze(-1).shape)
    soft_q_values = torch.gather(q_values[0], dim=1, index=actions[0, :].unsqueeze(-1)).squeeze(-1)
    
    critic_loss = mse(soft_q_values, target)


    return critic_loss


def compute_actor_loss(q_values):
    """Returns the actor loss

    :param q_values: The q-values (shape 2xB)
    :return: A scalar (the loss)
    """
    return -q_values[0].mean()


params = {
    "save_best": True,
    "base_dir": "${gym_env.env_name}/sac-S${algorithm.seed}_${current_time:}",
    "algorithm": {
        "seed": 1,
        "n_envs": 1,
        "n_steps": 32,
        "buffer_size": 1e6,
        "batch_size": 256,
        "max_grad_norm": 0.5,
        "nb_evals": 10,
        "eval_interval": 2_000,
        "learning_starts": 10_000,
        "max_epochs": 2_000,
        "discount_factor": 0.98,
        "entropy_mode": "auto",  # "auto" or "fixed"
        "init_entropy_coef": 2e-7,
        "tau_target": 0.05,
        "architecture": {
            "actor_hidden_size": [22, 22],
            "critic_hidden_size": [128, 128],
        },
    },
    "gym_env": {"env_name": "CartPole-v1"},
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 0.00022551554188448688, 
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 0.0002567848671307378, 
    },
    "entropy_coef_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 0.00028917836049013965,  
    },
}




class TD3(EpochBasedAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Define the agents and optimizers for TD3

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()
        
        self.critic_1 = DiscreteQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix("target-critic_1/")

        self.critic_2 = DiscreteQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix("target-critic_2/")
        self.actor = DiscretePolicy(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )


        self.train_policy = self.train_policy = self.actor
        self.eval_policy = Agents( #Permet de pouvoir interroger l'actor et le critic
            KWAgentWrapper(self.actor, stochastic=False),
            self.critic_1
        )

        
        self.t_actor = TemporalAgent(self.actor)
        self.t_critic_1 = TemporalAgent(self.critic_1)
        self.t_critic_2 = TemporalAgent(self.critic_2)
        self.t_target_critic_1 = TemporalAgent(self.target_critic_1)
        self.t_target_critic_2 = TemporalAgent(self.target_critic_2)
        
        #Dois etre completer
        
        # Configure the optimizer
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        critic_optimizer_args = get_arguments(cfg.critic_optimizer)
        parameters = nn.Sequential(self.critic_1, self.critic_2).parameters()
        self.critic_optimizer = get_class(cfg.critic_optimizer)(
            parameters, **critic_optimizer_args
        )
       




def run_td3(td3: TD3):
    for rb in td3.iter_replay_buffers():
        rb_workspace = rb.get_shuffled(td3.cfg.algorithm.batch_size)

        done, terminated, reward = rb_workspace["env/done", "env/truncated", "env/reward"]
        must_bootstrap = torch.logical_or(~done[1], terminated[1])
        td3.t_critic_1(rb_workspace, t = 0, n_steps=1, detach_actions=True)
        q_values_1 = rb_workspace["critic/q_value"]
        td3.t_critic_2(rb_workspace, t = 0, n_steps=1, detach_actions=True)
        q_values_2 = rb_workspace["critic/q_value"]

        
        with torch.no_grad():
            td3.t_actor(rb_workspace, t=1, n_steps=1)
            actions = rb_workspace["action"]
            
            td3.t_target_critic_1(rb_workspace, t=1, n_steps=1, detach_actions=True)
            target_q_values_1 = rb_workspace["target-critic_1/q_value"]
            
            td3.t_target_critic_2(rb_workspace, t=1, n_steps=1, detach_actions=True)
            target_q_values_2 = rb_workspace["target-critic_2/q_value"]

        t_target_critic = torch.min(target_q_values_1, target_q_values_2)
        # Compute the critic loss

        # Critic update
        
        # Compute critic loss
        critic_loss_1 = compute_critic_loss(td3.cfg, reward, must_bootstrap, q_values_1, t_target_critic, actions)
        critic_loss_2 = compute_critic_loss(td3.cfg, reward, must_bootstrap, q_values_2, t_target_critic, actions)
        critic_loss = critic_loss_1 + critic_loss_2
        # Gradient step (critic)
        td3.logger.add_log("critic_loss", critic_loss, td3.nb_steps)
        td3.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.critic_1.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            td3.critic_1.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        
        td3.critic_optimizer.step()
        

        td3.t_actor(rb_workspace, t=0, n_steps=1)
        
        td3.t_critic_1(rb_workspace, t=0, n_steps=1)
        q_values_1 = rb_workspace["critic/q_value"]

        
        actor_loss = compute_actor_loss(q_values_1)


        # Gradient step (actor)
        td3.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.actor.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.actor_optimizer.step()

        # Soft update of target q function
        soft_update_params(
            td3.critic_1, td3.target_critic_1, td3.cfg.algorithm.tau_target
        )
        soft_update_params(
            td3.critic_2, td3.target_critic_2, td3.cfg.algorithm.tau_target
        )
        
        # Evaluate the actor if needed
        td3.evaluate()


td3 = TD3(OmegaConf.create(params))
run_td3(td3)
#td3.visualize_best()