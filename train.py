import os
import time

import gymnasium as gym
import hydra
import numpy as np
import torch
from tqdm import trange

from actor import RandomActor, TD3_Actor
from critic import Critic
from logger import TensorboardSummaries as Summaries
from replay_buffer import ReplayBuffer
from utils import (
    compute_actor_loss,
    compute_critic_target,
    optimize,
    play_and_record,
    update_target_networks,
)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):
        env_name = "HalfCheetah-v4"

        env = gym.make(env_name, render_mode="rgb_array")

        # we want to look inside
        env.reset()

        # examples of states and actions
        print(
            "observation space: ",
            env.observation_space,
            "\nobservations:",
            env.reset()[0],
        )
        print("action space: ", env.action_space)

        env = gym.make(env_name, render_mode="rgb_array")
        env = Summaries(env, env_name)

        state_dim = env.observation_space.shape[
            0
        ]  # dimension of state space (27 numbers)
        action_dim = env.action_space.shape[0]  # dimension of action space (8 numbers)

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # iterations passed
        n_iterations = 1000

        exp_replay = ReplayBuffer(self.cfg.max_buffer_size)

        # models to train
        actor = TD3_Actor(state_dim, action_dim).to(DEVICE)
        critic1 = Critic(state_dim, action_dim).to(DEVICE)
        critic2 = Critic(state_dim, action_dim).to(DEVICE)

        # target networks: slow-updated copies of actor and two critics
        target_critic1 = Critic(state_dim, action_dim).to(DEVICE)
        target_critic2 = Critic(state_dim, action_dim).to(DEVICE)
        target_actor = TD3_Actor(state_dim, action_dim).to(
            DEVICE
        )  # comment this line if you chose SAC

        # initialize them as copies of original models
        target_critic1.load_state_dict(critic1.state_dict())
        target_critic2.load_state_dict(critic2.state_dict())
        target_actor.load_state_dict(
            actor.state_dict()
        )  # comment this line if you chose SAC

        # optimizers: for every model we have
        opt_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
        opt_critic1 = torch.optim.Adam(critic1.parameters(), lr=3e-4)
        opt_critic2 = torch.optim.Adam(critic2.parameters(), lr=3e-4)

        np.random.seed(self.cfg.seed)
        # env.unwrapped.seed(seed)
        torch.manual_seed(self.cfg.seed)

        interaction_state, _ = env.reset(seed=self.cfg.seed)
        random_actor = RandomActor(env)

        for n_iterations in trange(
            0, self.cfg.iterations, self.cfg.timesteps_per_epoch
        ):
            # if experience replay is small yet, no training happens
            # we also collect data using random policy to collect more diverse starting data
            if len(exp_replay) < self.cfg.start_timesteps:
                _, interaction_state = play_and_record(
                    interaction_state,
                    random_actor,
                    env,
                    exp_replay,
                    DEVICE,
                    self.cfg.timesteps_per_epoch,
                )
                continue

            # perform a step in environment and store it in experience replay
            _, interaction_state = play_and_record(
                interaction_state,
                actor,
                env,
                exp_replay,
                DEVICE,
                self.cfg.timesteps_per_epoch,
            )

            # sample a batch from experience replay
            states, actions, rewards, next_states, is_done = exp_replay.sample(
                self.cfg.batch_size
            )

            # move everything to PyTorch tensors
            states = torch.tensor(states, device=DEVICE, dtype=torch.float)
            actions = torch.tensor(actions, device=DEVICE, dtype=torch.float)
            rewards = torch.tensor(rewards, device=DEVICE, dtype=torch.float)
            next_states = torch.tensor(next_states, device=DEVICE, dtype=torch.float)
            is_done = torch.tensor(
                is_done.astype("float32"), device=DEVICE, dtype=torch.float
            )

            # losses
            critic1_loss = (
                critic1.get_qvalues(states, actions)
                - compute_critic_target(
                    target_actor,
                    target_critic1,
                    target_critic2,
                    rewards,
                    next_states,
                    is_done,
                    DEVICE,
                )
            ) ** 2
            optimize(
                env,
                "critic1",
                critic1,
                opt_critic1,
                critic1_loss,
                self.cfg.max_grad_norm,
                n_iterations,
            )

            critic2_loss = (
                critic2.get_qvalues(states, actions)
                - compute_critic_target(
                    target_actor,
                    target_critic1,
                    target_critic2,
                    rewards,
                    next_states,
                    is_done,
                    DEVICE,
                )
            ) ** 2
            optimize(
                env,
                "critic2",
                critic2,
                opt_critic2,
                critic2_loss,
                self.cfg.max_grad_norm,
                n_iterations,
            )

            # actor update is less frequent in TD3
            if n_iterations % self.cfg.policy_update_freq == 0:
                actor_loss = compute_actor_loss(actor, target_critic1, states)
                optimize(
                    env,
                    "actor",
                    actor,
                    opt_actor,
                    actor_loss,
                    self.cfg.max_grad_norm,
                    n_iterations,
                )

                # update target networks
                update_target_networks(critic1, target_critic1, self.cfg.tau)
                update_target_networks(critic2, target_critic2, self.cfg.tau)
                update_target_networks(
                    actor, target_actor, self.cfg.tau
                )  # comment this line if you chose SAC

        if not os.path.exists(self.cfg.models_path):
            os.makedirs(self.cfg.models_path)
        PATH = (
            self.cfg.models_path
            + "/actor_"
            + str(self.cfg.iterations)
            + "_"
            + str(self.cfg.batch_size)
            + "_"
            + str(time.time())
        )
        torch.save(actor, PATH)


@hydra.main(config_path="configs", config_name="cheetah_config", version_base="1.3.2")
def main(cfg):
    trainer = Trainer(cfg.train)
    trainer.train()


if __name__ == "__main__":
    main()
