# RL Trading System V2 — TODO

## Data Pipeline
- [x] Multi-source data collection (Binance API + Yahoo Finance)
- [x] Automatic data stitching and normalization across sources
- [x] 20+ technical indicators (EMA, MACD, RSI, BB, ATR, OBV, Stochastic, ADX, CCI, Williams %R, Ichimoku, Fibonacci, Volume Profile)
- [x] Multi-timeframe support (1h, 4h, 1d) with feature aggregation
- [x] WaveTrend indicator (LazyBear) converted from PineScript
- [x] Stochastic RSI with overbought/oversold zones
- [x] Money Flow Index (MFI) for volume-weighted momentum
- [x] Ehlers Stochastic CG Oscillator

## RL Environment
- [x] High-leverage Gymnasium environment (up to 30x)
- [x] Realistic margin calls and liquidation simulation
- [x] Multi-asset portfolio (BTC, Gold, EUR/USD, NASDAQ, S&P500)
- [x] Dual-mode: swing trading (4-8/month) + scalping (4-5/day)
- [x] Position-based trading with automatic SL/TP (3:1 R:R)
- [x] Compact observation space (84 dims) for generalization
- [x] Noise injection during training for robustness

## RL Agent
- [x] PPO + LSTM recurrent policy network
- [x] Entropy regularization and advantage normalization
- [x] Aggressive compound-focused reward shaping (10-100% per trade)
- [x] Sharpe ratio penalty and drawdown constraints
- [x] EWC (Elastic Weight Consolidation) for online learning stability
- [x] Curriculum training: 3x → 7x → 12x leverage stages

## Training Results (v9)
- [x] 65% win rate on unseen test data (BTC dropped 28.5%)
- [x] Sharpe ratio 15-16 across all leverage levels
- [x] Consistent performance at 3x, 7x, and 12x leverage
- [x] 1,200 episodes trained across 3 curriculum stages
- [x] Max drawdown 15.7% at 3x leverage

## Online Learning
- [x] Continuous adaptation to live market data
- [x] Replay buffer management for incremental updates
- [x] EWC consolidation after each curriculum stage
- [x] No manual retraining required

## Integrations
- [x] Binance API for live data streaming and order placement
- [x] LLM-powered sentiment analysis (news/social media)
- [x] Real-time notifications for high-conviction trades and risk alerts
- [x] S3 cloud backup for model weights and training metrics

## Dashboard
- [x] Real-time portfolio value and equity curve
- [x] Trade history table with P&L
- [x] Performance metrics (ROI, Sharpe, win rate, drawdown)
- [x] Model confidence scores
- [x] Agent control panel (start/stop/retrain)
- [x] Training Report tab with results table and chart gallery
- [x] Updated with v9 results (65% WR, Sharpe 16+)
- [ ] WebSocket live updates (polling used as fallback)

## Database
- [x] Schema for trades, portfolio snapshots, training runs, model versions
- [x] Migration and seed data

## Testing
- [x] Vitest tests for server routers (28 tests passing)
- [x] Agent training verification (9 iterations: v1→v9)
- [x] Offline training completed and evaluated
- [x] Online learning tested and assessed

## Remaining
- [ ] Add trade cooldown for real-market frequency control
- [ ] Add slippage simulation for realistic backtesting
- [x] Binance API keys for live trading
- [ ] WebSocket live updates
- [ ] Transformer encoder upgrade (from LSTM)
- [ ] Regime-tagged replay buffer for catastrophic forgetting prevention

## V10 Retrain with Real Fees + Telegram Bot
- [ ] Update trading env with 0.05% fee, 4-6h cooldown, funding rate simulation
- [ ] Retrain agent with 1200 episodes and real fee structure
- [ ] Run overfitting diagnostic on v10 model
- [ ] Verify profitability AFTER real fees
- [x] Build Telegram signal bot integration
- [x] Add Telegram bot token secret
- [x] Send trade signals to Telegram group (entry, direction, leverage, SL/TP, confidence)
- [ ] Update dashboard with v10 results
- [ ] Run vitest tests
- [ ] Save checkpoint

## Deployment — V9 Model + Binance + Telegram
- [x] Restore v9 trading_env.py (revert v10 changes)
- [x] Build Binance live trading engine (order placement, position management)
- [x] Build Telegram signal bot (trade signals to group)
- [x] Update dashboard with live trading controls
- [x] Request Binance API keys and Telegram bot token
- [x] Run vitest tests (28 passing)
- [x] Save checkpoint and deliver

## Bug Fixes
- [ ] Fix 500 error when clicking Start Trading on Live Trading tab
