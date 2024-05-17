# Continuous-RL

RL applied for MuJoCO environments with continuous action space

![ant](https://github.com/nokronim/Continius-RL/blob/main/assets/ant.gif)
![half-cheetah](https://github.com/nokronim/Continius-RL/blob/main/assets/half-cheetah.gif)

# Poetry setup

```bash
poetry shell
```
```bash
poetry install
```

```bash
poetry run python3 train.py --config-name "<your_config>"
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
