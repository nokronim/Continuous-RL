# Continuous-RL

RL applied for MuJoCO environments with continuous action space

![https://github.com/nokronim/Continius-RL/blob/main/rl-video-episode-0.gif](https://github.com/nokronim/Continius-RL/blob/main/assets/rl-video-episode-0.gif)

# Poetry setup

```bash
poetry shell
```
```bash
poetry install
```

```bash
poetry run python3 main.py --config-name "<your_config>"
```

# Docker usage
```bash
docker build -t continuous_rl .
```
```bash
docker run -it continuous_rl
```
## Acknowledgement
Made with the assistance of https://github.com/daniyalaliev
