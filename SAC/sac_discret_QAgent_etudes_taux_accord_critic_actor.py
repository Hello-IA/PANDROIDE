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
        scores = scores - scores.max(dim=-1, keepdim=True)[0]
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
    
            return (self.register_evaluation(rewards), rewards)

        
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
        
        actions, action_probs, action_logprobs = rb_workspace["action", "action_probs", "action_logprobs"]
        t_target_q_agents(rb_workspace, t=1, n_steps=1)
        
        q_values_next_1, q_values_next_2 =rb_workspace["target-critic-1/q_value", "target-critic-2/q_value"]
        q_values_next = torch.minimum(q_values_next_1[1], q_values_next_2[1])

        
        esperance_interne = (action_probs[1] * (
                q_values_next - ent_coef * action_logprobs[1]
        )).mean(dim=1)
        target = reward[1] + cfg.algorithm.discount_factor*esperance_interne*must_bootstrap[1].int()
        
    
    q_value_1, q_value_2  = rb_workspace["critic-1/q_value", "critic-2/q_value"]

    q_value_1 = q_value_1[0]
    q_value_2 = q_value_2[0]

    actions = actions[0, :]

    soft_q_values = torch.gather(q_value_1, dim=1, index=actions.unsqueeze(-1)).squeeze(-1)
    soft_q_values2 = torch.gather(q_value_2, dim=1, index=actions.unsqueeze(-1)).squeeze(-1)
    critic_loss_1 = torch.nn.MSELoss(reduction="none")(soft_q_values, target)
    critic_loss_2 = torch.nn.MSELoss(reduction="none")(soft_q_values2, target)

    mean_critic_loss_1 = critic_loss_1.mean()
    mean_critic_loss_2 = critic_loss_2.mean()
    

    return mean_critic_loss_1, mean_critic_loss_2



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


    t_actor(rb_workspace, t=0, n_steps=1)
    action_probs, action_logprobs = rb_workspace["action_probs", "action_logprobs"]
    
    t_q_agents(rb_workspace, t=0, n_steps=1)
    q_value_1, q_value_2 = rb_workspace["critic-1/q_value", "critic-2/q_value"]
    
    q_value_1 = q_value_1[0]
    q_value_2 = q_value_2[0]

    action_logprobs = action_logprobs[0]

    current_q_values = torch.minimum(q_value_1, q_value_2)

    inside_term = ent_coef * action_logprobs - current_q_values

    actor_loss = (action_probs * inside_term).sum(dim=1).mean()

    return actor_loss


import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def plot_learning_curve(logger_critic_loss_1, logger_actor_loss, logger_reward, logger_nb_steps, save_path=None):
    """
    Plot learning curves from the logger data
    
    Args:
        logger: The logger object from the algorithm
        save_path: Optional path to save the figure
    """

    # Create a figure with multiple subplots
    

    plt.plot(logger_nb_steps, logger_actor_loss)
    plt.title('Actor Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.show()

    
    plt.plot(logger_nb_steps, logger_critic_loss_1, label='Critic 1')
    plt.title('Critic Losses')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

<<<<<<< HEAD
    plt.plot(range(len(logger_reward)), logger_reward, label='Reward')
    plt.title('Reward')
    plt.xlabel('Steps')
    plt.ylabel('Mean reward')
    plt.legend()
    plt.show()
    
=======
>>>>>>> 0a07302a9ba98b9e1b432ad82acb4044da9346b1

all_taux_accord = [] #recupere le taux d'accord pour chaque evaluation de la politique
steps_evaluation = [] #recupere le taux d'accord pour chaque evaluation de la politique

def run_sac(sac: SACAlgo):


    cfg = sac.cfg
    logger = sac.logger

    #logger_critic_loss, logger_actor_loss, logger_reward, logger_nb_steps = [], [], [], []
    logger_reward = []
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

        
        #logger_nb_steps.append(sac.nb_steps)
        
        logger.add_log("critic_loss_1", critic_loss_1, sac.nb_steps)
        logger.add_log("critic_loss_2", critic_loss_2, sac.nb_steps)
        critic_loss = critic_loss_1 + critic_loss_2
        #logger_critic_loss.append(critic_loss.detach().numpy())
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
        
        #logger_actor_loss.append(actor_loss.detach().numpy())
        
        logger.add_log("actor_loss", actor_loss, sac.nb_steps)

        
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
            eval_reward = sac.evaluate()
            if eval_reward  != None:
                logger_reward.append(torch.max(eval_reward[1]).item())
            #print(logger_reward)
    return logger_reward[-1]
        ####################################################
    """
        # Soft update of target q function
        soft_update_params(sac.critic_1, sac.target_critic_1, tau)
        soft_update_params(sac.critic_2, sac.target_critic_2, tau)
        eval_reward = sac.evaluate(nbrun)
        if eval_reward  != None:
            logger_reward.append(torch.max(eval_reward[1]).item())
            #print(logger_reward)
    plot_learning_curve(logger_critic_loss, logger_actor_loss, logger_reward, logger_nb_steps, "sac_learning_curve.png")
<<<<<<< HEAD
    """
def objective(trial):
    # Sample values of alpha_critic and alpha_actor
    params = {
=======
        
all_taux_accord_n = []
steps_evaluation_n = []
import copy

def n_run_sac(sac: SACAlgo, n):
    #lancement de la fonction run n fois
    global all_taux_accord, steps_evaluation
    
    for i in range (n):
        all_taux_accord = []
        steps_evaluation = []
        run_sac(sac)
        all_taux_accord_n.append(copy.deepcopy(all_taux_accord))
        steps_evaluation_n.append(copy.deepcopy(steps_evaluation))
    
    
        

params = {
>>>>>>> 0a07302a9ba98b9e1b432ad82acb4044da9346b1
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
        "max_epochs": 100,
        "discount_factor": 0.98,
        "entropy_mode": "auto",  # "auto" or "fixed"
        "init_entropy_coef": 2e-7,
        "tau_target": 0.05,
        "architecture": {
            "actor_hidden_size": [20, 20],
            "critic_hidden_size": [256, 256],
        },
    },
    "gym_env": {"env_name": "CartPole-v1"},
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-3,
    },
    "entropy_coef_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,
    },
    }
    params["critic_optimizer"]["lr"] = trial.suggest_float('alpha_critic', 0.003, 0.01)
    params["actor_optimizer"]["lr"] = trial.suggest_float('alpha_actor', 0.003, 0.01)
    params["entropy_coef_optimizer"]["lr"] = trial.suggest_float('entropy_coef_optimizer',  0.003, 0.01)
    params["algorithm"]["architecture"]["actor_hidden_size"] = trial.suggest_categorical('actor_hidden_size', ((15, 15),(20, 20), (25, 25)))
    params["algorithm"]["architecture"]["critic_hidden_size"] = trial.suggest_categorical('critic_hidden_size', ((32, 32), (64, 64), (128, 128)))
    agents = SACAlgo(OmegaConf.create(params))
    final_reward = run_sac(agents)


    # We want to maximize the norm of the final value function

    return final_reward

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
        "max_epochs": 100,
        "discount_factor": 0.98,
        "entropy_mode": "auto",  # "auto" or "fixed"
        "init_entropy_coef": 2e-7,
        "tau_target": 0.05,
        "architecture": {
            "actor_hidden_size": [15, 15],
            "critic_hidden_size": [64, 64],
        },
    },
    "gym_env": {"env_name": "CartPole-v1"},
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 0.006899231168956194,
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 0.007271329043761232,
    },
    "entropy_coef_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 3e-4,
    },
}
"""
study_Bayes = optuna.create_study(direction='maximize')
study_Bayes.optimize(objective, n_trials=400)

# Data frame contains the params and the corresponding norm of the final value function
study_Bayes_analyse = study_Bayes.trials_dataframe(attrs=('params', 'value')) 

best_Bayes = study_Bayes.best_params
print ('The best parameters founded using Bayesian optimization are: ', best_Bayes, '\n\n')


"""

agents = SACAlgo(OmegaConf.create(params))
run_sac(agents)
    
import json

data = {
    'all_taux_accord': all_taux_accord,
    'steps_evaluation': steps_evaluation
}

with open('data.json', 'w') as f:
    json.dump(data, f)

# Visualize the best policy
#agents.visualize_best()

"""
plt.plot(steps_evaluation, all_taux_accord)
plt.xlabel("steps")
plt.ylabel("taux d'accord")
plt.title("Taux d'accord entre l'actor et le critic lors de chaque évaluation")
plt.show()
"""