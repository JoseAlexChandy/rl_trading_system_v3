"""
trainer.py
==========
Training orchestrator for the RL trading agent.

Handles:
  - Offline training on historical data
  - Online learning from live market data
  - Multi-asset training with curriculum
  - Performance tracking and model checkpointing
"""

import os, json, time, logging
from datetime import datetime
from typing import Optional, Callable

import numpy as np
import pandas as pd

from data_pipeline import load_all_assets, FEATURE_COLS, normalise_features
from trading_env import LeveragedTradingEnv
from ppo_lstm_agent import PPOLSTMAgent

logger = logging.getLogger("trainer")

# ─── Training configuration ──────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "n_episodes": 100,
    "rollout_steps": 2048,
    "window_size": 30,
    "initial_cash": 10.0,
    "max_leverage": 30.0,
    "hidden_size": 256,
    "lstm_layers": 2,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "entropy_coef": 0.02,
    "ppo_epochs": 4,
    "mini_batch_size": 64,
    "mode": "swing",
    "data_start": "2017-01-01",
}


class Trainer:
    """
    Orchestrates training of the PPO-LSTM agent across multiple assets.
    """

    def __init__(self, config: Optional[dict] = None,
                 on_episode: Optional[Callable] = None,
                 on_update: Optional[Callable] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.on_episode = on_episode  # callback(episode, metrics)
        self.on_update = on_update    # callback(step, loss_stats)
        self.agent = None
        self.envs = {}
        self.results = []

    def _build_env(self, asset_data: dict, mode: str = "swing") -> LeveragedTradingEnv:
        """Create environment from asset data."""
        df = asset_data["df"]
        features = asset_data["features"]
        prices = df["Close"].values

        return LeveragedTradingEnv(
            prices=prices,
            features=features,
            window_size=self.config["window_size"],
            initial_cash=self.config["initial_cash"],
            max_leverage=self.config["max_leverage"],
            mode=mode,
        )

    def load_data(self) -> dict:
        """Load all asset data."""
        logger.info("Loading multi-asset data ...")
        data = load_all_assets(start=self.config["data_start"])
        logger.info(f"Loaded {len(data)} assets")
        return data

    def train(self, data: Optional[dict] = None) -> dict:
        """
        Run full offline training.

        Returns dict with training metrics.
        """
        if data is None:
            data = self.load_data()

        if not data:
            raise ValueError("No asset data available for training")

        # Build environments for each asset
        for name, asset_data in data.items():
            self.envs[name] = self._build_env(asset_data, self.config["mode"])
            logger.info(f"  Env {name}: {len(asset_data['df'])} steps")

        # Get obs dimension from first env
        first_env = list(self.envs.values())[0]
        obs_dim = first_env.observation_space.shape[0]

        # Create agent
        self.agent = PPOLSTMAgent(
            obs_dim=obs_dim,
            n_actions=first_env.action_space.n,
            hidden_size=self.config["hidden_size"],
            lstm_layers=self.config["lstm_layers"],
            lr=self.config["lr"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_eps=self.config["clip_eps"],
            entropy_coef=self.config["entropy_coef"],
            ppo_epochs=self.config["ppo_epochs"],
            mini_batch_size=self.config["mini_batch_size"],
        )

        asset_names = list(self.envs.keys())
        n_episodes = self.config["n_episodes"]
        rollout_steps = self.config["rollout_steps"]

        all_metrics = []

        for ep in range(1, n_episodes + 1):
            # Rotate through assets (curriculum-style)
            asset_name = asset_names[(ep - 1) % len(asset_names)]
            env = self.envs[asset_name]

            obs, _ = env.reset()
            self.agent.reset_hidden()

            ep_reward = 0.0
            ep_steps = 0

            for step in range(rollout_steps):
                action, log_prob, value = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.agent.buffer.add(obs, action, log_prob, reward, value, float(done))

                obs = next_obs
                ep_reward += reward
                ep_steps += 1

                if done:
                    break

            # Get last value for GAE
            with __import__("torch").no_grad():
                _, last_val, _ = self.agent.network(
                    __import__("torch").tensor(obs, dtype=__import__("torch").float32,
                                               device=self.agent.device).unsqueeze(0),
                    self.agent.hidden,
                )
                last_value = last_val.item()

            # PPO update
            loss_stats = self.agent.update(last_value)

            equity = env._equity()
            roi = (equity - self.config["initial_cash"]) / self.config["initial_cash"] * 100

            metrics = {
                "episode": ep,
                "asset": asset_name,
                "equity": round(equity, 4),
                "roi_pct": round(roi, 2),
                "reward": round(ep_reward, 4),
                "steps": ep_steps,
                "trades": env.total_trades,
                "win_rate": round(env.winning_trades / max(env.total_trades, 1) * 100, 1),
                **{k: round(v, 6) for k, v in loss_stats.items()},
            }
            all_metrics.append(metrics)

            if self.on_episode:
                self.on_episode(ep, metrics)

            if ep % 10 == 0 or ep == 1:
                logger.info(
                    f"Ep {ep:3d}/{n_episodes} | {asset_name:8s} | "
                    f"Equity ${equity:.4f} | ROI {roi:+.1f}% | "
                    f"Trades {env.total_trades} | WR {metrics['win_rate']:.0f}%"
                )

        self.results = all_metrics
        return {
            "episodes": n_episodes,
            "final_metrics": all_metrics[-1] if all_metrics else {},
            "all_metrics": all_metrics,
        }

    def evaluate(self, data: dict, asset_name: str = "BTCUSDT") -> dict:
        """Evaluate trained agent on test data."""
        if self.agent is None:
            raise ValueError("Agent not trained yet")

        if asset_name not in data:
            raise ValueError(f"Asset {asset_name} not in data")

        env = self._build_env(data[asset_name], self.config["mode"])
        obs, _ = env.reset()
        self.agent.reset_hidden()

        while True:
            action, _, _ = self.agent.select_action(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        equity = env._equity()
        roi = (equity - self.config["initial_cash"]) / self.config["initial_cash"] * 100

        return {
            "asset": asset_name,
            "initial_cash": self.config["initial_cash"],
            "final_equity": round(equity, 4),
            "roi_pct": round(roi, 2),
            "total_trades": env.total_trades,
            "win_rate": round(env.winning_trades / max(env.total_trades, 1) * 100, 1),
            "history": env.history,
        }

    def save_model(self, path: str):
        if self.agent:
            self.agent.save(path)

    def load_model(self, path: str, obs_dim: int):
        if self.agent is None:
            self.agent = PPOLSTMAgent(obs_dim=obs_dim)
        self.agent.load(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    trainer = Trainer(config={
        "n_episodes": 30,
        "rollout_steps": 1024,
        "data_start": "2020-01-01",
        "mode": "swing",
    })

    result = trainer.train()
    print(json.dumps(result["final_metrics"], indent=2))
