"""
Diagnose why the agent makes 0 trades on test data.
"""
import os, sys, logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import MultiTimeframePipeline
from trading_env import ScalpingTradingEnv
from ppo_lstm_agent import TransformerPPOAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("diagnose")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    logger.info("=== DIAGNOSIS: Why 0 trades on test? ===")

    # 1. Load data
    pipeline = MultiTimeframePipeline(symbol="BTCUSDT", base_tf="1h")
    features, prices, ts = pipeline.run(do_normalize=True)
    features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)
    features = np.clip(features, -5.0, 5.0)

    split = int(len(features) * 0.8)
    tr_f, tr_p = features[:split], prices[:split]
    te_f, te_p = features[split:], prices[split:]
    logger.info(f"Train: {tr_f.shape}, Test: {te_f.shape}")

    # 2. Feature distribution shift
    tr_mean, tr_std = np.mean(tr_f, axis=0), np.std(tr_f, axis=0)
    te_mean = np.mean(te_f, axis=0)
    shift = np.abs(tr_mean - te_mean) / (tr_std + 1e-8)
    logger.info(f"Features >2std shift: {np.sum(shift > 2.0)}/{features.shape[1]}")
    logger.info(f"Max shift: {np.max(shift):.2f} at idx {np.argmax(shift)}")

    # 3. Create envs
    env_cfg = dict(initial_cash=10.0, max_leverage=5.0, sl_pct=0.008,
                   tp_pct=0.02, max_trades_per_day=5, candles_per_day=24, cooldown_bars=2)
    train_env = ScalpingTradingEnv(prices=tr_p, features=tr_f, **env_cfg)
    test_env = ScalpingTradingEnv(prices=te_p, features=te_f, **env_cfg)

    train_obs, _ = train_env.reset()
    test_obs, _ = test_env.reset()
    n_feat = tr_f.shape[1]
    logger.info(f"Train obs range: [{train_obs.min():.3f}, {train_obs.max():.3f}]")
    logger.info(f"Test obs range:  [{test_obs.min():.3f}, {test_obs.max():.3f}]")
    logger.info(f"Position+Account (train): {train_obs[n_feat:]}")
    logger.info(f"Position+Account (test):  {test_obs[n_feat:]}")

    # 4. Load model
    obs_dim = n_feat + 14
    agent = TransformerPPOAgent(
        obs_dim=obs_dim, n_actions=5, seq_len=32,
        d_model=128, n_heads=4, n_layers=3,
        lr=3e-4, entropy_coef=0.03, ewc_lambda=5000.0,
    )
    mp = os.path.join(RESULTS_DIR, "best_model_v6.pt")
    if os.path.exists(mp):
        agent.load(mp)
        logger.info("Model loaded")

    # 5. Deterministic actions on train
    obs, _ = train_env.reset()
    agent.reset_sequence()
    tr_acts = []
    for _ in range(min(300, len(tr_p)-1)):
        a, _, _, _ = agent.select_action(obs, deterministic=True)
        tr_acts.append(a)
        obs, _, done, _, _ = train_env.step(a)
        if done: break
    c = np.bincount(tr_acts, minlength=5)
    logger.info(f"TRAIN det: Hold={c[0]} Long={c[1]} Short={c[2]} Close={c[3]} Tight={c[4]}")

    # 6. Deterministic actions on test
    obs, _ = test_env.reset()
    agent.reset_sequence()
    te_acts = []
    for _ in range(min(300, len(te_p)-1)):
        a, _, _, _ = agent.select_action(obs, deterministic=True)
        te_acts.append(a)
        obs, _, done, _, _ = test_env.step(a)
        if done: break
    c = np.bincount(te_acts, minlength=5)
    logger.info(f"TEST det:  Hold={c[0]} Long={c[1]} Short={c[2]} Close={c[3]} Tight={c[4]}")

    # 7. Stochastic actions on test
    obs, _ = test_env.reset()
    agent.reset_sequence()
    st_acts = []
    for _ in range(min(300, len(te_p)-1)):
        a, _, _, _ = agent.select_action(obs, deterministic=False)
        st_acts.append(a)
        obs, _, done, _, _ = test_env.step(a)
        if done: break
    c = np.bincount(st_acts, minlength=5)
    logger.info(f"TEST stoch: Hold={c[0]} Long={c[1]} Short={c[2]} Close={c[3]} Tight={c[4]}")

    # 8. Raw logits on first test obs
    obs, _ = test_env.reset()
    agent.reset_sequence()
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        try:
            logits, val = agent.policy(obs_t)
            probs = torch.softmax(logits, dim=-1)
            logger.info(f"Logits: {logits.squeeze().numpy()}")
            logger.info(f"Probs:  {probs.squeeze().numpy()}")
            logger.info(f"Value:  {val.item():.4f}")
        except Exception as e:
            logger.info(f"Raw output error: {e}")

    # 9. Full stochastic test run
    test_env2 = ScalpingTradingEnv(prices=te_p, features=te_f, **env_cfg)
    obs, _ = test_env2.reset()
    agent.reset_sequence()
    for _ in range(len(te_p)-1):
        a, _, _, _ = agent.select_action(obs, deterministic=False)
        obs, _, done, _, info = test_env2.step(a)
        if done: break
    logger.info(f"Full stoch test: bal=${info.get('balance',0):.2f}, trades={info.get('total_trades',0)}, WR={info.get('win_rate',0):.1f}%")

    # 10. Summary
    logger.info("\n=== DIAGNOSIS SUMMARY ===")
    te_c = np.bincount(te_acts, minlength=5)
    st_c = np.bincount(st_acts, minlength=5)
    if te_c[0] > 280:
        logger.info("ROOT CAUSE: Agent always HOLD on test (deterministic)")
        if st_c[1] + st_c[2] > 10:
            logger.info("  -> Stochastic mode trades! Policy learned but collapsed to HOLD")
            logger.info("  -> FIX: Use epsilon-greedy eval, or increase entropy, or add exploration bonus")
        else:
            logger.info("  -> Even stochastic doesn't trade. Policy fully collapsed.")
            logger.info("  -> FIX: Fundamental reward/architecture change needed")
        if np.sum(shift > 2.0) > 20:
            logger.info(f"  -> ALSO: {np.sum(shift > 2.0)} features have >2std distribution shift")
            logger.info("  -> FIX: Use online normalization or percentile-based features")


if __name__ == "__main__":
    main()
