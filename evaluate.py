import time

import gymnasium as gym
import hydra
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = gym.make(self.cfg.env_name, render_mode="rgb_array")

        if self.cfg.model_load_path is not None:
            self.actor = torch.load(self.cfg.model_load_path)
        else:
            raise ValueError("Please, set model_load_path for evaluation")

    def evaluate(self, n_games=1, t_max=1000):
        """
        Plays n_games and returns rewards and rendered games
        """
        rewards = []

        for _ in range(n_games):
            s, _ = self.env.reset()

            total_reward = 0
            for _ in range(t_max):
                # select action for final evaluation of your policy
                action = self.actor.get_best_action(s)

                assert (action.max() <= 1).all() and (action.min() >= -1).all()

                s, r, terminated, truncated, _ = self.env.step(
                    action.cpu().detach().numpy()
                )
                total_reward += r

                if terminated or truncated:
                    break

            rewards.append(total_reward)

        self.env.close()
        return np.array(rewards)

    def visualise(self):
        with gym.make(self.cfg.env_name, render_mode="rgb_array") as env, RecordVideo(
            env=env,
            video_folder="./videos",
            name_prefix=self.cfg.env_name + "_" + str(time.time()),
        ) as env_monitor:
            self.env = env_monitor
            # note that t_max is 300, so collected reward will be smaller than 1000
            self.evaluate(n_games=1, t_max=300)


@hydra.main(config_path="configs", config_name="cheetah_config", version_base="1.3.2")
def main(cfg):
    evaluator = Evaluator(cfg)
    sessions = evaluator.evaluate()
    score = sessions.mean()
    print(f"Your score: {score}")
    evaluator.visualise()


if __name__ == "__main__":
    main()
