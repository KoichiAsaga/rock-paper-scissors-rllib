# Rock-Paper-Scissors RLlib

A simple implementation of a Rock-Paper-Scissors (RPS) environment and training agents using [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html).
This project demonstrates how to build a minimal multi-agent environment and train reinforcement learning agents for discrete action spaces.

---

## 🚀 Features

* ✅ Custom Rock-Paper-Scissors Gym environment
* ✅ Compatible with Ray RLlib multi-agent API
* ✅ Simple and educational example of competitive RL

---

## 📂 Repository Structure

```
rock-paper-scissors-rllib/
├── .devcontainer            # devcontainer settings
├── train.py                 # Training script using RLlib
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

---

## 🔧 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/KoichiAsaga/rock-paper-scissors-rllib.git
   cd rock-paper-scissors-rllib
   ```

2. Install dependencies:
* build devcontainer
  
---

## ▶️ Usage

### Train an agent:

```bash
python train.py
tensorboard --logdir .
```
---

## 🧑‍💻 Example Algorithms

* PPO

---

## 📜 License

This project is licensed under the MIT License.

---

## 🔗 References

* [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html)
* [OpenAI Gym](https://gym.openai.com/)
