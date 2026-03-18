"""
trading_env.py  (v9 — Position-Based with Natural Trade Limiting)
=================================================================
The agent steps through every bar and decides: FLAT / LONG / SHORT.
Position changes incur fees. SL/TP are automatic.
The key: reward is based on POSITION P&L, not trade P&L.
This naturally limits trading because switching positions costs fees.

The observation includes binary confluence signals so the agent
can learn when WaveTrend + StochRSI + MACD align.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

TAKER_FEE = 0.0004  # 0.04% per side


class TradingEnv(gym.Env):
    """
    Position-based trading environment with automatic SL/TP.
    
    Actions: 0=FLAT, 1=LONG, 2=SHORT
    
    The agent learns WHEN to enter (confluence) and WHEN to exit.
    Fee cost naturally prevents overtrading.
    """

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        feature_names: list = None,
        initial_cash: float = 10.0,
        max_leverage: float = 10.0,
        sl_pct: float = 0.02,
        tp_pct: float = 0.06,
        fee: float = TAKER_FEE,
        max_steps: int = None,
        noise_std: float = 0.0,  # Training noise
    ):
        super().__init__()
        self.prices = prices.astype(np.float64)
        self.features = features.astype(np.float32)
        self.feature_names = feature_names or []
        self.initial_cash = initial_cash
        self.max_leverage = max_leverage
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.fee = fee
        self.max_steps = max_steps or len(prices)
        self.noise_std = noise_std

        n_feat = features.shape[1]
        # obs = features + [position, unrealized_pnl, equity_ratio, bars_in_pos,
        #                    drawdown, win_rate, trade_count, time_pct, momentum_5, momentum_20]
        self.obs_dim = n_feat + 10
        self.observation_space = spaces.Box(-5.0, 5.0, (self.obs_dim,), np.float32)
        self.action_space = spaces.Discrete(3)  # FLAT, LONG, SHORT

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_cash
        self.position = 0      # -1, 0, 1
        self.entry_price = 0.0
        self.entry_bar = 0
        self.step_idx = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.peak_equity = self.initial_cash
        self.history = []
        self._steps_in_position = 0
        self._total_fee_paid = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        idx = min(self.step_idx, len(self.features) - 1)
        feat = self.features[idx].copy()

        # Add noise during training for robustness
        if self.noise_std > 0:
            feat += np.random.normal(0, self.noise_std, feat.shape).astype(np.float32)

        price = self.prices[idx]
        unrealized = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized = (price - self.entry_price) / self.entry_price * self.position

        equity = self._equity()
        portfolio = np.array([
            float(self.position),
            unrealized * self.max_leverage,  # Leveraged unrealized P&L
            np.log(equity / self.initial_cash + 1e-8),
            float(self._steps_in_position) / 50.0,
            (self.peak_equity - equity) / (self.peak_equity + 1e-8),  # Drawdown
            float(self.winning_trades) / max(self.total_trades, 1),   # Win rate
            float(self.total_trades) / 50.0,
            float(self.step_idx) / len(self.prices),
            self.prices[idx] / self.prices[max(0, idx-5)] - 1.0 if idx >= 5 else 0.0,
            self.prices[idx] / self.prices[max(0, idx-20)] - 1.0 if idx >= 20 else 0.0,
        ], dtype=np.float32)

        obs = np.concatenate([feat, portfolio])
        return np.clip(obs, -5.0, 5.0)

    def _equity(self):
        if self.position == 0:
            return self.cash
        price = self.prices[min(self.step_idx, len(self.prices) - 1)]
        pnl_pct = (price - self.entry_price) / self.entry_price * self.position
        margin = self.cash
        unrealized = margin * self.max_leverage * pnl_pct
        return max(self.cash + unrealized, 0.0)

    def _close_position(self, reason="manual"):
        """Close current position and return P&L."""
        if self.position == 0:
            return 0.0

        price = self.prices[min(self.step_idx, len(self.prices) - 1)]
        pnl_pct = (price - self.entry_price) / self.entry_price * self.position
        margin = self.cash
        notional = margin * self.max_leverage
        gross_pnl = notional * pnl_pct
        close_fee = notional * self.fee
        net_pnl = gross_pnl - close_fee
        self._total_fee_paid += close_fee

        self.cash += net_pnl
        self.cash = max(self.cash, 0.001)
        self.total_trades += 1

        pnl_pct_actual = net_pnl / margin * 100

        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        self.history.append({
            "entry_bar": self.entry_bar,
            "exit_bar": self.step_idx,
            "side": "long" if self.position == 1 else "short",
            "entry": self.entry_price,
            "exit": price,
            "leverage": self.max_leverage,
            "pnl": net_pnl,
            "pnl_pct": pnl_pct_actual,
            "balance": self.cash,
            "bars_held": self.step_idx - self.entry_bar,
            "reason": reason,
        })

        self.position = 0
        self.entry_price = 0.0
        self._steps_in_position = 0
        return net_pnl

    def _open_position(self, direction):
        """Open a new position."""
        price = self.prices[min(self.step_idx, len(self.prices) - 1)]
        notional = self.cash * self.max_leverage
        open_fee = notional * self.fee
        self._total_fee_paid += open_fee
        # Fee is deducted from potential profit, not from cash upfront

        self.position = direction
        self.entry_price = price
        self.entry_bar = self.step_idx
        self._steps_in_position = 0

    def step(self, action):
        if self.step_idx >= len(self.prices) - 1:
            if self.position != 0:
                self._close_position("end_of_data")
            return self._get_obs(), 0.0, True, False, self._info()

        price = self.prices[self.step_idx]
        reward = 0.0
        prev_equity = self._equity()

        # Check SL/TP if in position
        if self.position != 0:
            pnl_pct = (price - self.entry_price) / self.entry_price * self.position
            lev_pnl = pnl_pct * self.max_leverage

            if lev_pnl <= -self.sl_pct * self.max_leverage:
                # Stop loss hit
                net = self._close_position("stop_loss")
                reward = -1.0  # Fixed penalty for SL
            elif lev_pnl >= self.tp_pct * self.max_leverage:
                # Take profit hit
                net = self._close_position("take_profit")
                reward = 3.0  # Big reward for TP (3:1 R:R)

        # Process action (only if not already closed by SL/TP)
        desired = action - 1  # 0→-1(short), 1→0(flat), 2→1(long) ... wait
        # Remap: 0=FLAT, 1=LONG, 2=SHORT
        if action == 0:
            desired = 0
        elif action == 1:
            desired = 1
        elif action == 2:
            desired = -1

        if desired != self.position:
            # Close existing position first
            if self.position != 0:
                net = self._close_position("signal_exit")
                if net > 0:
                    reward += 1.0
                else:
                    reward -= 0.5

            # Open new position
            if desired != 0 and self.cash > 0.01:
                self._open_position(desired)

        # Position holding reward/penalty
        if self.position != 0:
            self._steps_in_position += 1
            # Small reward for holding a winning position
            curr_pnl = (price - self.entry_price) / self.entry_price * self.position
            if curr_pnl > 0:
                reward += 0.01  # Tiny reward for being in profit
            elif self._steps_in_position > 30:
                reward -= 0.005  # Tiny penalty for holding too long without profit

        # Equity change reward (small)
        new_equity = self._equity()
        eq_change = (new_equity - prev_equity) / (prev_equity + 1e-8)
        reward += eq_change * 2.0  # Scale equity changes

        # Advance
        self.step_idx += 1
        self.peak_equity = max(self.peak_equity, self._equity())

        # Liquidation check
        if self._equity() < self.initial_cash * 0.02:
            if self.position != 0:
                self._close_position("liquidation")
            return self._get_obs(), -5.0, True, False, self._info()

        terminated = self.step_idx >= len(self.prices) - 1
        if terminated and self.position != 0:
            self._close_position("end_of_data")

        return self._get_obs(), reward, terminated, False, self._info()

    def _info(self):
        return {
            "equity": self._equity(),
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1) * 100,
            "fees_paid": self._total_fee_paid,
        }


# Backward compat
HybridTradingEnv = TradingEnv
LeveragedTradingEnv = TradingEnv
SignalTradingEnv = TradingEnv
N_ACTIONS = 3
ACTIONS = {0: "flat", 1: "long", 2: "short"}
