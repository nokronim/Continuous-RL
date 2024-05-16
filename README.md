# Continius-RL
RL applied for MuJoCO environments with continuous action space

![https://github.com/nokronim/Continius-RL/blob/main/rl-video-episode-0.gif](https://github.com/nokronim/Continius-RL/blob/main/assets/rl-video-episode-0.gif)

# Poetry setup
```
poetry shell

poetry install

# set <your_config> in configs directory
poetry run python3 main.py --config_name "<your_config>"
```

# Docker usage
```
docker build -t mujoco_td3 .

docker run -it mujoco_td3
```
