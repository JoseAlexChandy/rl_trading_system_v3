"""
binance_trader.py — Live Trading Engine using V9 RL Model + Binance Futures
============================================================================
Connects to Binance Futures API, streams 1h candles, runs the RL agent,
and places orders when the agent signals a position change.

Sends trade signals to Telegram group.
"""

import os
import sys
import json
import time
import asyncio
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# Binance
from binance.client import Client as BinanceClient
from binance.enums import *

# Telegram
import telegram

# Local engine
sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import load_btc_multitf, build_features as compute_indicators_for_df
from trading_env import TradingEnv, N_ACTIONS, ACTIONS
from ppo_lstm_agent import PPOAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "live_trading.log")),
    ],
)
logger = logging.getLogger("binance_trader")

# ─── Configuration ──────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "results", "trained_model.pt")
OBS_DIM = 84
HIDDEN = 256
SYMBOL = "BTCUSDT"
DEFAULT_LEVERAGE = 7
TRADE_INTERVAL_SECONDS = 3600  # 1 hour (matches 1h candles)
MAX_POSITION_PCT = 0.1  # Use 10% of balance per trade


class TelegramSignalBot:
    """Sends trade signals to a Telegram group/channel."""

    def __init__(self, token: str, chat_id: str):
        self.bot = telegram.Bot(token=token) if token else None
        # Support multiple chat IDs separated by commas
        self.chat_ids = [c.strip() for c in chat_id.split(",")] if chat_id else []
        self.chat_id = self.chat_ids[0] if self.chat_ids else ""
        self.enabled = bool(token and chat_id)
        if self.enabled:
            logger.info(f"Telegram bot initialized for chat {chat_id}")
        else:
            logger.warning("Telegram bot disabled (missing token or chat_id)")

    async def send_signal(self, signal: Dict[str, Any]):
        """Send a formatted trade signal to the Telegram group."""
        if not self.enabled:
            logger.info(f"[Telegram DISABLED] Signal: {signal}")
            return

        try:
            side = signal.get("side", "unknown").upper()
            emoji = "🟢" if side == "LONG" else "🔴" if side == "SHORT" else "⚪"
            action = signal.get("action", "unknown").upper()

            if action == "OPEN":
                msg = (
                    f"{emoji} **{action} {side}** — {signal.get('symbol', SYMBOL)}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 Entry Price: `${signal.get('entry_price', 0):,.2f}`\n"
                    f"⚡ Leverage: `{signal.get('leverage', DEFAULT_LEVERAGE)}x`\n"
                    f"💰 Position Size: `${signal.get('notional', 0):,.2f}`\n"
                    f"🎯 Take Profit: `${signal.get('tp_price', 0):,.2f}` ({signal.get('tp_pct', 0):+.1f}%)\n"
                    f"🛑 Stop Loss: `${signal.get('sl_price', 0):,.2f}` ({signal.get('sl_pct', 0):+.1f}%)\n"
                    f"📈 Model Confidence: `{signal.get('confidence', 0):.1f}%`\n"
                    f"💵 Balance: `${signal.get('balance', 0):,.2f}`\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🤖 RL Agent V9 | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                )
            elif action == "CLOSE":
                pnl = signal.get("pnl", 0)
                pnl_emoji = "✅" if pnl > 0 else "❌"
                msg = (
                    f"{pnl_emoji} **{action} {side}** — {signal.get('symbol', SYMBOL)}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 Exit Price: `${signal.get('exit_price', 0):,.2f}`\n"
                    f"💰 P&L: `${pnl:+,.2f}` ({signal.get('pnl_pct', 0):+.1f}%)\n"
                    f"⏱ Duration: `{signal.get('duration', 'N/A')}`\n"
                    f"📋 Reason: `{signal.get('reason', 'signal')}`\n"
                    f"💵 Balance: `${signal.get('balance', 0):,.2f}`\n"
                    f"📊 Win Rate: `{signal.get('win_rate', 0):.1f}%`\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🤖 RL Agent V9 | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                )
            else:
                msg = f"ℹ️ {json.dumps(signal, indent=2)}"

            for cid in self.chat_ids:
              await self.bot.send_message(
                chat_id=cid,
                text=msg,
                parse_mode="Markdown",
            )
            logger.info(f"Telegram signal sent: {action} {side}")
        except Exception as e:
            logger.error(f"[ORDER ERROR] {type(e).__name__}: {e}", exc_info=True)
            logger.error(f"Telegram send failed: {e}")

    async def send_status(self, text: str):
        """Send a status update to the Telegram group."""
        if not self.enabled:
            return
        try:
            for cid in self.chat_ids:
                await self.bot.send_message(chat_id=cid, text=text)
        except Exception as e:
            logger.error(f"Telegram status send failed: {e}")


class BinanceLiveTrader:
    """
    Live trading engine that:
    1. Fetches latest 1h candles from Binance
    2. Computes multi-TF indicators
    3. Runs the RL agent to get action
    4. Places orders on Binance Futures
    5. Sends signals to Telegram
    """

    def __init__(
        self,
        binance_key: str,
        binance_secret: str,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        leverage: int = DEFAULT_LEVERAGE,
        testnet: bool = True,
        dry_run: bool = False,
    ):
        self.leverage = leverage
        self.testnet = testnet
        self.dry_run = dry_run
        self.symbol = SYMBOL
        self.running = False

        # State tracking
        self.current_position = 0  # -1, 0, 1
        self.entry_price = 0.0
        self.entry_time = None
        self.total_trades = 0
        self.winning_trades = 0
        self.trade_history = []
        self.initial_balance = -1.0  # -1 means not yet started
        self._last_features = None
        self._last_prices = None
        self._expected_features = 74  # model trained on 74 features

        # Binance client
        if testnet:
            self.client = BinanceClient(
                binance_key, binance_secret, testnet=True
            )
            logger.info("Connected to Binance TESTNET")
        else:
            self.client = BinanceClient(binance_key, binance_secret)
            logger.info("Connected to Binance LIVE")

        # Telegram bot
        self.telegram = TelegramSignalBot(telegram_token, telegram_chat_id)

        # Load RL agent
        self.agent = PPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=HIDDEN)
        self.agent.load(MODEL_PATH)
        self.agent.eval_epsilon = 0.0  # Pure greedy in live
        logger.info(f"RL Agent V9 loaded ({sum(p.numel() for p in self.agent.network.parameters()):,} params)")

        # Feature pipeline state
        self._last_features = None
        self._last_prices = None

    def _setup_leverage(self):
        """Set leverage on Binance Futures."""
        try:
            self.client.futures_change_leverage(
                symbol=self.symbol, leverage=self.leverage
            )
            logger.info(f"Leverage set to {self.leverage}x for {self.symbol}")
        except Exception as e:
            logger.warning(f"Failed to set leverage: {e}")

    def _get_balance(self) -> float:
        """Get USDT futures balance."""
        try:
            account = self.client.futures_account()
            for asset in account["assets"]:
                if asset["asset"] == "USDT":
                    return float(asset["walletBalance"])
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
        return 0.0

    def _get_current_price(self) -> float:
        """Get current BTC price."""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return 0.0

    def _get_position(self) -> Dict:
        """Get current Binance position info."""
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            for pos in positions:
                if pos["symbol"] == self.symbol:
                    amt = float(pos["positionAmt"])
                    return {
                        "amount": amt,
                        "side": 1 if amt > 0 else -1 if amt < 0 else 0,
                        "entry_price": float(pos["entryPrice"]),
                        "unrealized_pnl": float(pos["unRealizedProfit"]),
                        "leverage": int(pos["leverage"]),
                    }
        except Exception as e:
            logger.error(f"Failed to get position: {e}")
        return {"amount": 0, "side": 0, "entry_price": 0, "unrealized_pnl": 0, "leverage": self.leverage}

    def _fetch_latest_features(self) -> tuple:
        """Fetch latest candle data and compute features."""
        try:
            features, prices, feat_names = load_btc_multitf()
            if features.shape[1] != self._expected_features:
                logger.warning(f"Feature shape mismatch: got {features.shape[1]}, expected {self._expected_features}. Using cached data.")
                if self._last_features is not None:
                    return self._last_features, self._last_prices, []
            self._last_features = features
            self._last_prices = prices
            logger.info(f"Features updated: {features.shape}, latest price ${prices[-1]:,.2f}")
            return features, prices, feat_names
        except Exception as e:
            logger.error(f"Failed to fetch features: {e}")
            if self._last_features is not None:
                logger.warning("Using cached features from last successful fetch")
                return self._last_features, self._last_prices, []
            return None, None, []

    def _get_observation(self, features, prices, step_idx) -> np.ndarray:
        """Build the observation vector matching the v9 env format."""
        idx = min(step_idx, len(features) - 1)
        feat = features[idx].copy()
        price = prices[idx]

        unrealized = 0.0
        if self.current_position != 0 and self.entry_price > 0:
            unrealized = (price - self.entry_price) / self.entry_price * self.current_position

        balance = self._get_balance() if not self.dry_run else 10.0
        equity = balance
        if self.current_position != 0:
            equity += balance * self.leverage * unrealized

        steps_in_pos = 0
        if self.entry_time:
            steps_in_pos = int((datetime.now(timezone.utc) - self.entry_time).total_seconds() / 3600)

        peak_equity = max(equity, self.initial_balance) if self.initial_balance > 0 else equity

        portfolio = np.array([
            float(self.current_position),
            unrealized * self.leverage,
            np.log(equity / max(self.initial_balance, 0.01) + 1e-8),
            float(steps_in_pos) / 50.0,
            (peak_equity - equity) / (peak_equity + 1e-8),
            float(self.winning_trades) / max(self.total_trades, 1),
            float(self.total_trades) / 50.0,
            0.5,  # time_pct (middle of data)
            prices[idx] / prices[max(0, idx - 5)] - 1.0 if idx >= 5 else 0.0,
            prices[idx] / prices[max(0, idx - 20)] - 1.0 if idx >= 20 else 0.0,
        ], dtype=np.float32)

        obs = np.concatenate([feat, portfolio])
        return np.clip(obs, -5.0, 5.0)

    async def _place_order(self, side: str, close: bool = False) -> Optional[Dict]:
        """Place a market order on Binance Futures."""
        if self.dry_run:
            price = self._get_current_price()
            logger.info(f"[DRY RUN] Would {side} at ${price:,.2f}")
            return {"price": price, "status": "DRY_RUN"}

        try:
            balance = self._get_balance()
            price = self._get_current_price()

            if close:
                # Close existing position
                pos = self._get_position()
                if abs(pos["amount"]) > 0:
                    close_side = SIDE_SELL if pos["amount"] > 0 else SIDE_BUY
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side=close_side,
                        type=ORDER_TYPE_MARKET,
                        quantity=abs(pos["amount"]),
                        reduceOnly=True,
                    )
                    order_id = order.get("orderId")
                    import asyncio
                    filled = await self._verify_order_filled(order_id)
                    if not filled:
                        logger.error(f"Close order {order_id} did not fill! Not updating internal state.")
                        await self.telegram.send_status(f"⚠️ Close order failed to fill! Order ID: {order_id}")
                        return None
                    logger.info(f"Position closed and verified: {order}")
                    return order
            else:
                # Open new position
                notional = balance * self.leverage * MAX_POSITION_PCT
                quantity = round(notional / price, 3)
                # Ensure minimum order size ($100 notional minimum on Binance)
                min_qty = round(100.0 / price + 0.001, 3)
                if quantity < min_qty:
                    quantity = min_qty
                    logger.info(f"Adjusted to minimum order size: {quantity} BTC (${quantity*price:.2f} notional)")
                if False:
                    logger.warning(f"Order too small: {quantity} BTC (${notional:.2f} notional)")
                    return None

                order_side = SIDE_BUY if side == "LONG" else SIDE_SELL
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=order_side,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity,
                )
                order_id = str(order.get("orderId", ""))
                import time; time.sleep(1)
                try:
                    filled_order = self.client.futures_get_order(symbol=self.symbol, orderId=order_id)
                    if filled_order.get("status") != "FILLED":
                        logger.error(f"Open order {order_id} not filled! Status: {filled_order.get('status')}")
                        return None
                except Exception as ve:
                    logger.warning(f"Could not verify open order: {ve}")
                logger.info(f"Order placed and verified: {side} {quantity} BTC at ~${price:,.2f}")
                return order

        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None

    async def _handle_action(self, action: int, price: float, confidence: float):
        """Process the RL agent's action and place orders."""
        desired = {0: 0, 1: 1, 2: -1}.get(action, 0)

        if desired == self.current_position:
            return  # No change

        balance = self._get_balance() if not self.dry_run else 10.0

        # Close existing position
        if self.current_position != 0:
            pnl = 0.0
            pnl_pct = 0.0
            if self.entry_price > 0:
                pnl_pct = (price - self.entry_price) / self.entry_price * self.current_position * self.leverage * 100
                pnl = balance * pnl_pct / 100

            close_result = await self._place_order("", close=True)
            if close_result is None:
                logger.error("Close order failed! Skipping state update — will retry next cycle.")
                await self.telegram.send_status(
                    f"⚠️ Failed to close position!\n"
                    f"Will retry next cycle."
                )
                return

            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1

            duration = "N/A"
            if self.entry_time:
                delta = datetime.now(timezone.utc) - self.entry_time
                hours = delta.total_seconds() / 3600
                duration = f"{hours:.1f}h"

            trade_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "side": "long" if self.current_position == 1 else "short",
                "entry": self.entry_price,
                "exit": price,
                "pnl": round(pnl, 4),
                "pnl_pct": round(pnl_pct, 2),
                "duration": duration,
                "balance": round(balance + pnl, 4),
            }
            self.trade_history.append(trade_record)

            # Send close signal to Telegram
            await self.telegram.send_signal({
                "action": "CLOSE",
                "side": "LONG" if self.current_position == 1 else "SHORT",
                "symbol": self.symbol,
                "exit_price": price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "duration": duration,
                "reason": "RL signal",
                "balance": balance + pnl,
                "win_rate": self.winning_trades / max(self.total_trades, 1) * 100,
            })

            self.current_position = 0
            self.entry_price = 0.0
            self.entry_time = None

        # Open new position
        if desired != 0:
            side_str = "LONG" if desired == 1 else "SHORT"
            open_result = await self._place_order(side_str, close=False)
            if open_result is None:
                logger.error("Open order failed! Not updating state.")
                await self.telegram.send_status(
                    f"⚠️ Failed to open {side_str} position!\n"
                    f"Will retry next cycle."
                )
                return

            self.current_position = desired
            self.entry_price = price
            self.entry_time = datetime.now(timezone.utc)

            # Calculate SL/TP
            sl_pct = 2.0  # 2% SL
            tp_pct = 6.0  # 6% TP (3:1 R:R)
            if desired == 1:
                sl_price = price * (1 - sl_pct / 100)
                tp_price = price * (1 + tp_pct / 100)
            else:
                sl_price = price * (1 + sl_pct / 100)
                tp_price = price * (1 - tp_pct / 100)

            notional = balance * self.leverage

            # Send open signal to Telegram
            await self.telegram.send_signal({
                "action": "OPEN",
                "side": side_str,
                "symbol": self.symbol,
                "entry_price": price,
                "leverage": self.leverage,
                "notional": notional,
                "tp_price": tp_price,
                "tp_pct": tp_pct * (1 if desired == 1 else -1),
                "sl_price": sl_price,
                "sl_pct": -sl_pct * (1 if desired == 1 else -1),
                "confidence": confidence,
                "balance": balance,
            })

    def _sync_position_with_binance(self):
        """Sync internal position state with actual Binance position."""
        if self.dry_run:
            return
        try:
            pos = self._get_position()
            actual_side = pos["side"]
            actual_entry = pos["entry_price"]

            if actual_side != self.current_position:
                logger.warning(
                    f"⚠️ Position mismatch! Engine thinks: {self.current_position}, "
                    f"Binance actual: {actual_side}. Syncing..."
                )
                self.current_position = actual_side
                self.entry_price = actual_entry
                # Send Telegram alert
                import asyncio
                asyncio.get_event_loop().run_until_complete(
                    self.telegram.send_status(
                        f"⚠️ Position Sync Alert\n"
                        f"Engine was out of sync with Binance!\n"
                        f"Corrected to: {'LONG' if actual_side == 1 else 'SHORT' if actual_side == -1 else 'FLAT'}\n"
                        f"Entry: ${actual_entry:,.2f}"
                    )
                )
            else:
                logger.info(f"Position sync OK: {actual_side} @ ${actual_entry:,.2f}")
        except Exception as e:
            logger.error(f"Position sync failed: {e}")

    async def _verify_order_filled(self, order_id: str) -> bool:
        """Verify an order actually filled on Binance."""
        try:
            import time
            time.sleep(1)  # Give Binance a moment
            order = self.client.futures_get_order(
                symbol=self.symbol,
                orderId=order_id
            )
            status = order.get("status", "")
            if status == "FILLED":
                logger.info(f"Order {order_id} confirmed FILLED")
                return True
            else:
                logger.warning(f"Order {order_id} status: {status} — not filled!")
                return False
        except Exception as e:
            logger.error(f"Order verification failed: {e}")
            return False

    async def run_once(self):
        """Run a single trading cycle: fetch data → get action → execute."""
        logger.info("─── Trading Cycle ───")

        # 0. Sync position with Binance before doing anything
        self._sync_position_with_binance()

        # 1. Fetch latest features
        features, prices, feat_names = self._fetch_latest_features()
        if features is None:
            logger.error("No features available, skipping cycle")
            return

        # 2. Build observation
        obs = self._get_observation(features, prices, len(prices) - 1)

        # 3. Get agent action
        self.agent.reset_hidden()
        action, log_prob, value = self.agent.select_action(obs, deterministic=True)
        action_name = ACTIONS[action]

        # Confidence from value estimate
        confidence = min(abs(value) * 20, 99.0)

        price = prices[-1]
        logger.info(
            f"Price: ${price:,.2f} | Action: {action_name} | "
            f"Confidence: {confidence:.1f}% | Position: {self.current_position}"
        )

        # 4. Execute action
        await self._handle_action(action, price, confidence)

        # 5. Log status
        balance = self._get_balance() if not self.dry_run else 10.0
        wr = self.winning_trades / max(self.total_trades, 1) * 100
        logger.info(
            f"Balance: ${balance:,.2f} | Trades: {self.total_trades} | "
            f"Win Rate: {wr:.1f}% | Position: {self.current_position}"
        )

    async def start(self):
        """Start the live trading loop."""
        logger.info("=" * 60)
        logger.info("STARTING LIVE TRADING — RL Agent V9")
        logger.info(f"Symbol: {self.symbol} | Leverage: {self.leverage}x")
        logger.info(f"Testnet: {self.testnet} | Dry Run: {self.dry_run}")
        logger.info("=" * 60)

        self._setup_leverage()
        self.initial_balance = self._get_balance() if not self.dry_run else 10.0
        logger.info(f"Starting balance: ${self.initial_balance:,.2f}")

        # Sync with current Binance position
        if not self.dry_run:
            pos = self._get_position()
            self.current_position = pos["side"]
            self.entry_price = pos["entry_price"]
            if self.current_position != 0:
                logger.info(f"Existing position: {'LONG' if self.current_position == 1 else 'SHORT'} at ${self.entry_price:,.2f}")

        await self.telegram.send_status(
            f"🤖 **RL Trading Bot Started**\n"
            f"Symbol: {self.symbol}\n"
            f"Leverage: {self.leverage}x\n"
            f"Balance: ${self.initial_balance:,.2f}\n"
            f"Mode: {'TESTNET' if self.testnet else 'LIVE'}"
        )

        # Pre-seed feature cache with startup data
        logger.info("Pre-seeding feature cache...")
        self._fetch_latest_features()
        self.running = True
        while self.running:
            try:
                await self.run_once()
            except Exception as e:
                logger.error(f"Trading cycle error: {e}", exc_info=True)
                await self.telegram.send_status(f"⚠️ Error: {str(e)[:200]}")

            # Wait for next candle
            logger.info(f"Sleeping {TRADE_INTERVAL_SECONDS}s until next candle...")
            await asyncio.sleep(TRADE_INTERVAL_SECONDS)

    def stop(self):
        """Stop the trading loop."""
        self.running = False
        logger.info("Trading stopped.")

    def get_status(self) -> Dict:
        """Get current trading status for the dashboard."""
        balance = self._get_balance() if not self.dry_run else 10.0
        return {
            "running": self.running,
            "symbol": self.symbol,
            "leverage": self.leverage,
            "testnet": self.testnet,
            "dry_run": self.dry_run,
            "balance": balance,
            "initial_balance": self.initial_balance,
            "roi_pct": (balance - self.initial_balance) / self.initial_balance * 100 if self.initial_balance > 0 else 0.0,
            "current_position": self.current_position,
            "entry_price": self.entry_price,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1) * 100,
            "trade_history": self.trade_history[-20:],  # Last 20 trades
        }


# ─── CLI Entry Point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RL Live Trader")
    parser.add_argument("--key", default=os.environ.get("BINANCE_API_KEY", ""))
    parser.add_argument("--secret", default=os.environ.get("BINANCE_API_SECRET", ""))
    parser.add_argument("--tg-token", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    parser.add_argument("--tg-chat", default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    parser.add_argument("--leverage", type=int, default=DEFAULT_LEVERAGE)
    parser.add_argument("--testnet", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trader = BinanceLiveTrader(
        binance_key=args.key,
        binance_secret=args.secret,
        telegram_token=args.tg_token,
        telegram_chat_id=args.tg_chat,
        leverage=args.leverage,
        testnet=not args.live,
        dry_run=args.dry_run,
    )

    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        trader.stop()
        print("\nTrading stopped by user.")
