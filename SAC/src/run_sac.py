
import torch
import torch.nn as nn

from bbrl.agents import  Agents, TemporalAgent
from bbrl_utils.nn import setup_optimizer, soft_update_params




from SACAlgo import *

import numpy as np

from plot import *

from compute import *
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



def run_sac(sac: SACAlgo):


    cfg = sac.cfg
    logger = sac.logger

    logger_critic_loss, logger_actor_loss, logger_reward, logger_nb_steps, logger_nb_steps_evale = [], [], [], [], []
    logger_reward = []
    # init_entropy_coef is the initial value of the entropy coef alpha.
    ent_coef = cfg.algorithm.init_entropy_coef
    #Sent_coef = 0
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

        
        logger_nb_steps.append(sac.nb_steps)
        
        logger.add_log("critic_loss_1", critic_loss_1, sac.nb_steps)
        logger.add_log("critic_loss_2", critic_loss_2, sac.nb_steps)
        critic_loss = critic_loss_1 + critic_loss_2
        logger_critic_loss.append(critic_loss.detach().numpy())
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
        
        logger_actor_loss.append(actor_loss.detach().numpy())
        
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
        
        # Soft update of target q function
        soft_update_params(sac.critic_1, sac.target_critic_1, tau)
        soft_update_params(sac.critic_2, sac.target_critic_2, tau)
        eval_reward = sac.evaluate()
        if eval_reward  != None:
            logger_reward.append(eval_reward[1])
            logger_nb_steps_evale.append(eval_reward[2])
    #plot_learning_curve(logger_critic_loss, logger_actor_loss, logger_reward, logger_nb_steps, "sac_learning_curve.png")

    return logger_reward, logger_nb_steps_evale