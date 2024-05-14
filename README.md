# Continius-RL
RL applied for MuJoCO environments with continuous action space 

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
