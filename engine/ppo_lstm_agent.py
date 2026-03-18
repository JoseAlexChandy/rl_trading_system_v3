"""
ppo_lstm_agent.py  (v7 — Simpler + Higher Entropy)
====================================================
Key fixes:
  1. Simpler MLP (no LSTM) — LSTM was overfitting to temporal patterns
  2. Higher entropy coefficient (0.05) to force exploration
  3. Epsilon-greedy during evaluation to prevent hold-collapse
  4. EWC for online learning stability
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional
import logging, io, json, copy

logger = logging.getLogger("ppo_agent")


class ActorCritic(nn.Module):
    """Simple MLP Actor-Critic — generalizes better than LSTM for this task."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]  # Take last timestep if sequence
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """PPO agent with EWC support for online learning."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 3,
        hidden: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 128,
        eval_epsilon: float = 0.15,  # Epsilon-greedy during eval
        ewc_lambda: float = 1000.0,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.eval_epsilon = eval_epsilon
        self.ewc_lambda = ewc_lambda

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(obs_dim, n_actions, hidden).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.training_stats = []

        # EWC
        self._ewc_params = {}
        self._fisher = {}

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.network(s)

            if deterministic:
                # Epsilon-greedy even in eval to prevent hold-collapse
                if np.random.random() < self.eval_epsilon:
                    action = np.random.randint(0, logits.shape[-1])
                else:
                    action = logits.argmax(dim=-1).item()
                log_prob = 0.0
            else:
                dist = Categorical(logits=logits)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        return action, log_prob, value.item()

    def reset_hidden(self):
        pass  # No LSTM state

    def _compute_gae(self, rewards, values, dones, last_value):
        advantages = []
        gae = 0.0
        values = values + [last_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values[:-1], dtype=np.float32)
        return advantages, returns

    def update(self, last_value: float = 0.0) -> dict:
        if len(self.buffer) == 0:
            return {}

        advantages, returns = self._compute_gae(
            self.buffer.rewards, self.buffer.values,
            self.buffer.dones, last_value,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_logp = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        total_loss_sum, n_updates = 0.0, 0

        for _ in range(self.ppo_epochs):
            indices = np.arange(len(self.buffer))
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.mini_batch_size):
                mb = indices[start:start + self.mini_batch_size]
                logits, values = self.network(states[mb])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                ratio = (new_logp - old_logp[mb]).exp()
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, ret_t[mb])

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # EWC penalty
                if self._fisher:
                    ewc_loss = 0.0
                    for name, param in self.network.named_parameters():
                        if name in self._fisher:
                            ewc_loss += (self._fisher[name] * (param - self._ewc_params[name]).pow(2)).sum()
                    loss += self.ewc_lambda * ewc_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss_sum += loss.item()
                n_updates += 1

        self.buffer.clear()
        stats = {"total_loss": total_loss_sum / max(n_updates, 1), "entropy": entropy.item()}
        self.training_stats.append(stats)
        return stats

    def consolidate_ewc(self):
        """Compute Fisher information for EWC after a training phase."""
        self._ewc_params = {n: p.clone().detach() for n, p in self.network.named_parameters()}
        # Approximate Fisher with gradient magnitudes from recent training
        self._fisher = {}
        for n, p in self.network.named_parameters():
            if p.grad is not None:
                self._fisher[n] = p.grad.data.clone().pow(2)
            else:
                self._fisher[n] = torch.zeros_like(p)
        logger.info("EWC consolidated")

    def online_update(self, states, actions, rewards, dones, last_value=0.0):
        """Conservative online update with EWC protection."""
        if len(states) < 10:
            return {}
        # Temporarily fill buffer
        values = []
        with torch.no_grad():
            for s in states:
                t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                _, v = self.network(t)
                values.append(v.item())

        for s, a, lp, r, v, d in zip(states, actions, [0.0]*len(states), rewards, values, dones):
            self.buffer.add(s, a, lp, r, v, d)

        result = self.update(last_value)
        return result

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "ewc_params": {k: v.cpu() for k, v in self._ewc_params.items()} if self._ewc_params else {},
            "fisher": {k: v.cpu() for k, v in self._fisher.items()} if self._fisher else {},
        }, path)
        logger.info(f"Model saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.training_stats = ckpt.get("training_stats", [])
        if ckpt.get("ewc_params"):
            self._ewc_params = {k: v.to(self.device) for k, v in ckpt["ewc_params"].items()}
        if ckpt.get("fisher"):
            self._fisher = {k: v.to(self.device) for k, v in ckpt["fisher"].items()}
        logger.info(f"Model loaded ← {path}")

    def get_state_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save({"network": self.network.state_dict(), "optimizer": self.optimizer.state_dict(),
                     "training_stats": self.training_stats}, buf)
        return buf.getvalue()

    def load_state_bytes(self, data: bytes):
        buf = io.BytesIO(data)
        ckpt = torch.load(buf, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.training_stats = ckpt.get("training_stats", [])


# Backward compat
PPOLSTMAgent = PPOAgent
