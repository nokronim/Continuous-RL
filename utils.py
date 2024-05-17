import torch
import torch.nn as nn


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for t in range(n_steps):
        # select action using policy with exploration
        a = agent.get_action(states=s)

        ns, r, terminated, truncated, _ = env.step(a)

        exp_replay.add(s, a, r, ns, terminated)

        s = env.reset()[0] if terminated or truncated else ns

        sum_rewards += r

    return sum_rewards, s


def optimize(env, name, model, optimizer, loss, max_grad_norm, n_iterations):
    """
    Makes one step of SGD optimization, clips norm with max_grad_norm and
    logs everything into tensorboard
    """
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    # logging
    env.writer.add_scalar(name, loss.item(), n_iterations)
    env.writer.add_scalar(name + "_grad_norm", grad_norm.item(), n_iterations)


def compute_critic_target(
    target_actor, target_critic1, target_critic2, rewards, next_states, is_done
):
    """
    Important: use target networks for this method! Do not use "fresh" models except fresh policy in SAC!
    input:
        rewards - PyTorch tensor, (batch_size)
        next_states - PyTorch tensor, (batch_size x features)
        is_done - PyTorch tensor, (batch_size)
    output:
        critic target - PyTorch tensor, (batch_size)
    """
    gamma = 0.997
    with torch.no_grad():
        critic_target = rewards + gamma * (1 - is_done) * torch.min(
            target_critic1.get_qvalues(
                next_states,
                target_actor.get_target_action(states=next_states),
            ),
            target_critic2.get_qvalues(
                next_states,
                target_actor.get_target_action(states=next_states),
            ),
        )

    assert not critic_target.requires_grad, "target must not require grad."
    assert len(critic_target.shape) == 1, "dangerous extra dimension in target?"

    return critic_target


def compute_actor_loss(actor, target_critic1, states):
    """
    Returns actor loss on batch of states
    input:
        states - PyTorch tensor, (batch_size x features)
    output:
        actor loss - PyTorch tensor, (batch_size)
    """
    # make sure you have gradients w.r.t. actor parameters
    actions = actor.get_best_action(states)

    assert (
        actions.requires_grad
    ), "actions must be differentiable with respect to policy parameters"

    # compute actor loss
    # TD3
    actor_loss = -target_critic1.get_qvalues(states, actions)
    return actor_loss


def update_target_networks(model, target_model, tau):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
