import torch
import torch.nn as nn


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
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
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    # logging
    env.writer.add_scalar(name, loss.item(), n_iterations)
    env.writer.add_scalar(name + "_grad_norm", grad_norm.item(), n_iterations)


def compute_critic_target(
    target_actor, target_critic1, target_critic2, rewards, next_states, gamma, is_done
):
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
    actions = actor.get_best_action(states)

    assert (
        actions.requires_grad
    ), "actions must be differentiable with respect to policy parameters"

    actor_loss = -target_critic1.get_qvalues(states, actions)
    return actor_loss


def update_target_networks(model, target_model, tau):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
