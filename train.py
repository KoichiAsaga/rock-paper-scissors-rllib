import numpy as np
from enum import IntEnum
from typing import Dict, Any
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from ray.rllib.env import EnvContext
from ray.rllib.utils.annotations import override as ray_override
import gymnasium as gym
import os
import shutil


class RockPaperScissorsEnum(IntEnum):
    """Enumeration representing Rock-Paper-Scissors hands"""

    ROCK = 0
    PAPER = 1
    SCISSORS = 2


class FightResultEnum(IntEnum):
    """Enumeration representing the result of Rock-Paper-Scissors"""

    WIN = 0
    LOSE = 1
    DRAW = 2


def get_rock_paper_scissors_result(
    my_hand: RockPaperScissorsEnum, enemy_hand: RockPaperScissorsEnum
) -> FightResultEnum:
    """Determine the result of a Rock-Paper-Scissors match.

    Args:
        my_hand (RockPaperScissorsEnum): Your hand
        enemy_hand (RockPaperScissorsEnum): Opponent's hand

    Returns:
        FightResultEnum: Result of the match
    """
    if my_hand == enemy_hand:
        # Draw
        return FightResultEnum.DRAW
    elif (
        (
            my_hand == RockPaperScissorsEnum.ROCK
            and enemy_hand == RockPaperScissorsEnum.SCISSORS
        )
        or (
            my_hand == RockPaperScissorsEnum.PAPER
            and enemy_hand == RockPaperScissorsEnum.ROCK
        )
        or (
            my_hand == RockPaperScissorsEnum.SCISSORS
            and enemy_hand == RockPaperScissorsEnum.PAPER
        )
    ):
        # Win
        return FightResultEnum.WIN
    else:
        # Lose
        return FightResultEnum.LOSE


class StochasticRockPaperScissorsHandMaker:
    """Class to probabilistically generate Rock-Paper-Scissors hands."""

    def __init__(self, choices: list[RockPaperScissorsEnum], weights: list[float]):
        self.choices = choices
        self.weights = weights

    def __call__(self) -> RockPaperScissorsEnum:
        choice = np.random.choice(self.choices, p=self.weights)
        return RockPaperScissorsEnum(choice)


class RockPaperScissorsEnv(gym.Env):
    """Rock-Paper-Scissors environment.

    The AI will play multiple rounds, and the opponent's hand is determined probabilistically.
    """

    def __init__(self, config: EnvContext):
        super().__init__()
        self.config = config

        # Number of Rock-Paper-Scissors rounds
        self.num_play = config.get("num_play_rock_paper_scissors", 10)

        # Initialize the non-AI opponent
        choices = config.get(
            "enemy_hand_choices",
            [
                RockPaperScissorsEnum.ROCK,
                RockPaperScissorsEnum.PAPER,
                RockPaperScissorsEnum.SCISSORS,
            ],
        )
        weights = config.get("enemy_hand_weights", [0.6, 0.2, 0.2])
        self.enemy_hand_maker = StochasticRockPaperScissorsHandMaker(choices, weights)

        # NOTE: MultiDiscrete observation space works only with old-api-stack
        self.observation_space = gym.spaces.MultiDiscrete(
            [3 for _ in range(self.num_play + 1)]
        )
        self.action_space = gym.spaces.Discrete(3)

    @ray_override(gym.Env)
    def reset(self, **kargs) -> tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Returns:
            tuple: (observation, additional info)
        """
        self.step_num = 0
        obs = np.zeros((self.num_play + 1,), dtype=int)
        obs[self.step_num] = RockPaperScissorsEnum.ROCK  # Start with ROCK
        self.obs = obs
        return obs, {}

    @ray_override(gym.Env)
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one step.

        Args:
            action (int): AI's move (0: Rock, 1: Paper, 2: Scissors)

        Returns:
            tuple: (observation, reward, done, truncated, additional info)
        """
        # Play Rock-Paper-Scissors
        self.step_num += 1
        my_hand = RockPaperScissorsEnum(action)
        enemy_hand = self.enemy_hand_maker()
        fight_result = get_rock_paper_scissors_result(my_hand, enemy_hand)

        # Set reward based on the result
        if fight_result == FightResultEnum.WIN:
            reward = 1.0
        elif fight_result == FightResultEnum.LOSE:
            reward = -1.0
        else:
            reward = 0.0

        # End after a fixed number of rounds
        done = self.step_num == self.num_play

        # Observe opponent's hand
        self.obs[self.step_num] = enemy_hand.value
        return self.obs, reward, done, False, {}


register_env("rock-paper-scissors", lambda config: RockPaperScissorsEnv(config=config))

log_dir = "logs"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir, exist_ok=True)

config = (
    PPOConfig()
    .environment(
        env="rock-paper-scissors",
        env_config={
            "num_play_rock_paper_scissors": 10,
            "enemy_hand_choices": [
                RockPaperScissorsEnum.ROCK,
                RockPaperScissorsEnum.PAPER,
                RockPaperScissorsEnum.SCISSORS,
            ],
            "enemy_hand_weights": [0.6, 0.2, 0.2],
        },
    )
    .api_stack(
        enable_env_runner_and_connector_v2=False,
        enable_rl_module_and_learner=False,
    )
    .framework("torch")
    .training(
        model={
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        train_batch_size=1000,
    )
    .resources(num_gpus=0)
    .env_runners(batch_mode="complete_episodes")
    .debugging(
        log_level="DEBUG",
        logger_creator=lambda config: UnifiedLogger(config, log_dir, loggers=None),  # type: ignore[call-arg]
    )
)

trainable = config.build()

trainable.get_policy()
num_epochs = 20

for i in range(num_epochs):
    print(f"Epoch -> {i + 1}/{num_epochs}")
    trainable.train()
