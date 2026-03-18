"""
api_server.py  (v9 — Live Trading + Telegram Signals)
======================================================
FastAPI server that bridges the Python RL engine to the Node.js frontend.
Runs on port 8100 and provides REST endpoints for:
  - Training management
  - Live trading control (Binance Futures)
  - Telegram signal bot
  - Data fetching & model management
  - Online learning
"""

import os, sys, json, logging, asyncio, time, traceback, base64, threading
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import load_btc_multitf
from trading_env import TradingEnv, N_ACTIONS, ACTIONS
from ppo_lstm_agent import PPOAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("api_server")

# ─── Global state ─────────────────────────────────────────────────────────────

OBS_DIM = 84
HIDDEN = 256
MODEL_PATH = os.path.join(os.path.dirname(__file__), "results", "trained_model.pt")

engine_state = {
    "agent": None,
    "data": None,
    "features": None,
    "prices": None,
    "training_active": False,
    "metrics_history": [],
    "current_run": None,
    # Live trading
    "live_trader": None,
    "live_thread": None,
    # Telegram
    "telegram_bot": None,
}

# ─── Pydantic models ─────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    n_episodes: int = 50
    mode: str = "swing"
    initial_cash: float = 10.0
    max_leverage: float = 10.0
    data_start: str = "2018-01-01"
    lr: float = 3e-4

class LiveStartRequest(BaseModel):
    leverage: int = 7
    testnet: bool = True
    dry_run: bool = False

class TelegramTestRequest(BaseModel):
    message: str = "Test signal from RL Trading Bot"

class PredictRequest(BaseModel):
    state: list

# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RL Engine API starting ...")
    # Auto-load v9 model
    try:
        agent = PPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=HIDDEN)
        agent.load(MODEL_PATH)
        engine_state["agent"] = agent
        logger.info(f"V9 model loaded: {sum(p.numel() for p in agent.network.parameters()):,} params")
    except Exception as e:
        logger.warning(f"Could not auto-load model: {e}")

    # Auto-load data
    try:
        features, prices, feat_names = load_btc_multitf()
        engine_state["features"] = features
        engine_state["prices"] = prices
        engine_state["data"] = {"features": features, "prices": prices, "feat_names": feat_names}
        logger.info(f"Data loaded: {features.shape[0]} bars, {features.shape[1]} features")
    except Exception as e:
        logger.warning(f"Could not auto-load data: {e}")

    yield
    # Cleanup
    if engine_state.get("live_trader"):
        engine_state["live_trader"].stop()
    logger.info("RL Engine API shutting down")


app = FastAPI(title="RL Trading Engine API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    live_trader = engine_state.get("live_trader")
    return {
        "status": "ok",
        "training_active": engine_state["training_active"],
        "live_trading_active": live_trader.running if live_trader else False,
        "agent_loaded": engine_state["agent"] is not None,
        "data_loaded": engine_state["data"] is not None,
    }

# ─── Data ────────────────────────────────────────────────────────────────────

@app.post("/data/load")
async def load_data(start: str = "2018-01-01"):
    try:
        features, prices, feat_names = load_btc_multitf()
        engine_state["features"] = features
        engine_state["prices"] = prices
        engine_state["data"] = {"features": features, "prices": prices, "feat_names": feat_names}
        return {"status": "ok", "bars": len(prices), "features": features.shape[1]}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/data/summary")
async def data_summary():
    if engine_state["data"] is None:
        return {"status": "no_data"}
    f = engine_state["features"]
    p = engine_state["prices"]
    return {
        "status": "ok",
        "bars": len(p),
        "features": f.shape[1],
        "price_range": f"${p.min():,.0f} - ${p.max():,.0f}",
        "latest_price": f"${p[-1]:,.2f}",
    }

# ─── Training ────────────────────────────────────────────────────────────────

def _run_training(config: dict):
    engine_state["training_active"] = True
    engine_state["metrics_history"] = []
    try:
        features = engine_state["features"]
        prices = engine_state["prices"]
        if features is None:
            features, prices, _ = load_btc_multitf()

        n = len(prices)
        split = int(n * 0.8)
        train_f, train_p = features[:split], prices[:split]

        agent = PPOAgent(obs_dim=features.shape[1] + 10, n_actions=N_ACTIONS, hidden=HIDDEN)

        for ep in range(config.get("n_episodes", 50)):
            # Random window
            window = min(len(train_p), 5000)
            start = np.random.randint(0, max(1, len(train_p) - window))
            end = start + window

            env = TradingEnv(
                prices=train_p[start:end],
                features=train_f[start:end],
                initial_cash=config.get("initial_cash", 10),
                max_leverage=config.get("max_leverage", 10),
                noise_std=0.02,
            )

            obs, _ = env.reset()
            agent.reset_hidden()
            ep_reward = 0
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

            for step in range(len(train_p[start:end]) - 1):
                action, lp, val = agent.select_action(obs)
                next_obs, reward, done, trunc, info = env.step(action)

                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(lp)
                values.append(val)
                dones.append(done)
                ep_reward += reward
                obs = next_obs

                if done:
                    break

            # PPO update
            if len(states) > 64:
                agent.update(states, actions, rewards, log_probs, values, dones)

            metrics = {
                "episode": ep + 1,
                "equity": info.get("equity", 0),
                "trades": info.get("total_trades", 0),
                "win_rate": info.get("win_rate", 0),
                "reward": ep_reward,
            }
            engine_state["metrics_history"].append(metrics)
            engine_state["current_run"] = metrics

            if (ep + 1) % 10 == 0:
                logger.info(f"Ep {ep+1}: equity=${metrics['equity']:.2f}, trades={metrics['trades']}, WR={metrics['win_rate']:.1f}%")

        engine_state["agent"] = agent
        agent.save(MODEL_PATH)
        engine_state["current_run"] = {**metrics, "status": "completed"}

    except Exception as e:
        logger.error(f"Training error: {traceback.format_exc()}")
        engine_state["current_run"] = {"status": "failed", "error": str(e)}
    finally:
        engine_state["training_active"] = False

@app.post("/train/start")
async def start_training(req: TrainRequest, bg: BackgroundTasks):
    if engine_state["training_active"]:
        raise HTTPException(409, "Training already in progress")
    config = req.model_dump()
    bg.add_task(_run_training, config)
    return {"status": "training_started", "config": config}

@app.get("/train/status")
async def training_status():
    return {
        "active": engine_state["training_active"],
        "current": engine_state["current_run"],
        "history_count": len(engine_state["metrics_history"]),
    }

@app.get("/train/metrics")
async def training_metrics(last_n: int = 100):
    return {
        "total": len(engine_state["metrics_history"]),
        "metrics": engine_state["metrics_history"][-last_n:],
    }

# ─── Prediction ──────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(req: PredictRequest):
    if engine_state["agent"] is None:
        raise HTTPException(400, "No agent loaded")
    state = np.array(req.state, dtype=np.float32)
    action, log_prob, value = engine_state["agent"].select_action(state, deterministic=True)
    return {
        "action": action,
        "action_name": ACTIONS[action],
        "confidence": float(np.exp(log_prob)) if log_prob != 0 else 1.0,
        "value_estimate": value,
    }

# ─── Live Trading ────────────────────────────────────────────────────────────

@app.get("/live/status")
async def live_status():
    trader = engine_state.get("live_trader")
    if trader:
        return trader.get_status()
    return {
        "running": False, "symbol": "BTCUSDT", "leverage": 7, "testnet": True,
        "dry_run": False, "balance": 0, "initial_balance": 0, "roi_pct": 0,
        "current_position": 0, "entry_price": 0, "total_trades": 0,
        "winning_trades": 0, "win_rate": 0, "trade_history": [],
    }

@app.post("/live/start")
async def live_start(req: LiveStartRequest):
    from binance_trader import BinanceLiveTrader

    binance_key = os.environ.get("BINANCE_API_KEY", "")
    binance_secret = os.environ.get("BINANCE_API_SECRET", "")
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not binance_key or not binance_secret:
        raise HTTPException(400, "Binance API keys not configured. Set BINANCE_API_KEY and BINANCE_API_SECRET.")

    # Stop existing trader
    if engine_state.get("live_trader"):
        engine_state["live_trader"].stop()

    trader = BinanceLiveTrader(
        binance_key=binance_key,
        binance_secret=binance_secret,
        telegram_token=tg_token,
        telegram_chat_id=tg_chat,
        leverage=req.leverage,
        testnet=req.testnet,
        dry_run=req.dry_run,
    )
    engine_state["live_trader"] = trader

    # Run in background thread
    def run_trader():
        asyncio.run(trader.start())

    thread = threading.Thread(target=run_trader, daemon=True)
    thread.start()
    engine_state["live_thread"] = thread

    return {"status": "started", "testnet": req.testnet, "leverage": req.leverage}

@app.post("/live/stop")
async def live_stop():
    trader = engine_state.get("live_trader")
    if trader:
        trader.stop()
        return {"status": "stopped"}
    return {"status": "not_running"}

@app.get("/live/trades")
async def live_trades(limit: int = 50):
    trader = engine_state.get("live_trader")
    if trader:
        return {"trades": trader.trade_history[-limit:]}
    return {"trades": []}

# ─── Telegram ────────────────────────────────────────────────────────────────

@app.post("/telegram/test")
async def telegram_test(req: TelegramTestRequest):
    import telegram as tg
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        return {"success": False, "error": "Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."}

    try:
        bot = tg.Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=req.message)
        return {"success": True, "message": "Test message sent!"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/telegram/config")
async def telegram_config():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    return {
        "configured": bool(token and chat_id),
        "chat_id": chat_id if chat_id else "",
        "has_token": bool(token),
    }

# ─── Portfolio ───────────────────────────────────────────────────────────────

@app.get("/portfolio")
async def get_portfolio():
    trader = engine_state.get("live_trader")
    if trader and trader.running:
        balance = 10.0  # Will be real balance when Binance is connected
        return {
            "cash": balance,
            "equity": balance,
            "positions": [{"side": trader.current_position, "entry": trader.entry_price}] if trader.current_position != 0 else [],
            "peak": max(balance, trader.initial_balance),
            "drawdown": 0,
        }
    return {"cash": 10, "equity": 10, "positions": [], "peak": 10, "drawdown": 0}

# ─── Model ───────────────────────────────────────────────────────────────────

@app.post("/model/save")
async def save_model():
    if engine_state["agent"] is None:
        raise HTTPException(400, "No agent loaded")
    path = os.path.join(os.path.dirname(__file__), "results", f"model_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pt")
    engine_state["agent"].save(path)
    return {"status": "ok", "path": path}

@app.get("/model/bytes")
async def get_model_bytes():
    if engine_state["agent"] is None:
        raise HTTPException(400, "No agent loaded")
    import io, torch
    buf = io.BytesIO()
    torch.save(engine_state["agent"].network.state_dict(), buf)
    data = buf.getvalue()
    return {"status": "ok", "size": len(data), "data_b64": base64.b64encode(data).decode()}

@app.get("/evaluate/{asset}")
async def evaluate_asset(asset: str):
    if engine_state["agent"] is None:
        raise HTTPException(400, "No agent loaded")
    if engine_state["features"] is None:
        raise HTTPException(400, "No data loaded")

    features = engine_state["features"]
    prices = engine_state["prices"]
    n = len(prices)
    split = int(n * 0.8)
    test_f, test_p = features[split:], prices[split:]

    env = TradingEnv(prices=test_p, features=test_f, initial_cash=10, max_leverage=7)
    obs, _ = env.reset()
    engine_state["agent"].reset_hidden()

    while True:
        action, _, _ = engine_state["agent"].select_action(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        if done:
            break

    return {
        "asset": asset,
        "equity": info["equity"],
        "roi_pct": (info["equity"] - 10) / 10 * 100,
        "trades": info["total_trades"],
        "win_rate": info["win_rate"],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
