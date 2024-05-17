from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import hydra
from gymnasium.wrappers import RecordVideo
from IPython.display import HTML


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = gym.make(self.cfg.env_name, render_mode="rgb_array")
        self.actor = torch.load(self.cfg.model_load_path)


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

                s, r, terminated, truncated, _ = self.env.step(action.cpu().detach().numpy())
                total_reward += r

                if terminated or truncated:
                    break

            rewards.append(total_reward)

        self.env.close()
        return np.array(rewards)


    def visualise(self):
        with gym.make(self.cfg.env_name, render_mode="rgb_array") as env, RecordVideo(
            env=env, video_folder="./videos"
        ) as env_monitor:
            self.env = env_monitor
            # note that t_max is 300, so collected reward will be smaller than 1000
            self.evaluate(n_games=1, t_max=300)

        video_paths = sorted([s for s in Path("videos").iterdir() if s.suffix == ".mp4"])
        video_path = video_paths[0]  # You can also try other indices


        data_url = str(video_path) + "_" + self.cfg.env_name

        HTML(f"""
            <video width="480" height="480" controls>
            <source src="{data_url}" type="video/mp4">
            </video>
            """)

@hydra.main(config_path="configs", config_name="cheetah_config", version_base="1.3.2")
def main(cfg):
    evaluator = Evaluator(cfg)
    sessions = evaluator.evaluate()
    score = sessions.mean()
    print(f"Your score: {score}")
    evaluator.visualise()


if __name__ == "__main__":
    main()
