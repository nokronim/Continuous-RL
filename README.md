# Continius-RL
RL applied for MuJoCO environments with continuous action space 

![]([name-of-giphy.gif](https://github.com/nokronim/Continius-RL/blob/main/rl-video-episode-0.gif))

# Poetry setup
```
poetry shell

poetry install

poetry run python3 main.py
```

# Docker usage
```
docker build -t mujoco_td3 . 

docker run -it mujoco_td3
```
