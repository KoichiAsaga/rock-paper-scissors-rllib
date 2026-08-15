# Rock-Paper-Scissors RLlib

A minimal implementation of a Rock-Paper-Scissors (RPS) environment and reinforcement
learning agent trained with [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html).
This project is an educational example of building a custom Gymnasium environment and
training an agent (PPO) on a discrete action space.

> 日本語の説明は [README_jp.md](README_jp.md) を参照してください。

---

## 🚀 Features

* ✅ Custom Rock-Paper-Scissors `gymnasium.Env`
* ✅ PPO training with Ray RLlib
* ✅ Stochastic opponent with configurable hand probabilities
* ✅ Ready-to-run development environment via Dev Container
* ✅ Step-by-step educational notebook

---

## 📂 Repository Structure

```
rock-paper-scissors-rllib/
├── .devcontainer/       # Dev Container settings (Docker-based dev environment)
├── train.py             # RPS environment + PPO training script
├── train.ipynb          # Educational notebook explaining the RL workflow
├── README.md            # This file (English)
├── README_jp.md         # Japanese README
└── .gitignore
```

---

## 🔧 Setup

This project is designed to run inside a Dev Container.

1. Install [VS Code](https://code.visualstudio.com/), the
   [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers),
   and [Docker](https://www.docker.com/).
2. Clone the repository:

   ```bash
   git clone https://github.com/KoichiAsaga/rock-paper-scissors-rllib.git
   cd rock-paper-scissors-rllib
   ```

3. Open the folder in VS Code, press `Ctrl+Shift+P`, and select
   **Dev Containers: Reopen in Container**.

---

## ▶️ Usage

### Run the training script

```bash
python train.py
```

Training logs are written to the `logs/` directory.

### Or follow the notebook

Open `train.ipynb` and run the cells top to bottom with a Python 3.10 kernel to
walk through the reinforcement learning workflow step by step.

---

## 🎮 Environment

* **Observation space**: `MultiDiscrete` — the history of hands over the rounds.
* **Action space**: `Discrete(3)` — Rock (0), Paper (1), Scissors (2).
* **Reward**: `+1` win, `-1` lose, `0` draw.
* **Opponent**: plays stochastically (default weights `[0.6, 0.2, 0.2]`), so the
  agent learns to exploit the biased distribution.

---

## 🧑‍💻 Algorithm

* PPO (Proximal Policy Optimization) via `ray.rllib`

---

## 📜 License

This project is licensed under the MIT License.

---

## 🔗 References

* [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html)
* [Gymnasium Documentation](https://gymnasium.farama.org/)
