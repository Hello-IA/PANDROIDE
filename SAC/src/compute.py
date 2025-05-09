
import torch
from bbrl.workspace import Workspace
from bbrl.agents import TemporalAgent

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

        #print("q_values_next",q_values_next.shape)
        #print("reward[1]",reward[1].shape)
        esperance_interne = (action_probs[1] * (
                q_values_next - ent_coef * action_logprobs[1]
        )).mean(dim=1)
        
        target = reward[1] + cfg.algorithm.discount_factor*esperance_interne*must_bootstrap[1].int()
        actions = actions[0, :]
        
    
    q_value_1, q_value_2  = rb_workspace["critic-1/q_value", "critic-2/q_value"]

    q_value_1 = q_value_1[0]
    q_value_2 = q_value_2[0]

    

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

