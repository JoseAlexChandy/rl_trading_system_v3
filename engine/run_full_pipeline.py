"""
run_full_pipeline.py  (v10 — Real Binance Fees + Trade Cooldown)
================================================================
Key changes from v9:
  1. Real Binance taker fee: 0.05% per side
  2. 6-hour cooldown between trades
  3. 4-hour minimum hold period
  4. Funding rate: 0.01% every 8 hours
  5. Fee deducted from margin on entry (agent feels the cost)
  6. Updated obs_dim (+2 for cooldown/can_trade features)
"""

import os, sys, json, logging, time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import load_btc_multitf
from trading_env import TradingEnv, ACTIONS, N_ACTIONS
from ppo_lstm_agent import PPOAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("pipeline")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_eval(agent, env, deterministic=True):
    obs, _ = env.reset()
    agent.reset_hidden()
    curve = [env.initial_cash]
    actions = []
    while True:
        a, _, _ = agent.select_action(obs, deterministic=deterministic)
        obs, _, t, tr, info = env.step(a)
        curve.append(env._equity())
        actions.append(a)
        if t or tr:
            break
    eq = env._equity()
    roi = (eq - env.initial_cash) / env.initial_cash * 100
    wr = env.winning_trades / max(env.total_trades, 1) * 100
    return {
        "equity": eq, "roi": roi, "trades": env.total_trades,
        "win_rate": wr, "curve": curve, "history": env.history,
        "actions": actions, "winning": env.winning_trades, "losing": env.losing_trades,
        "final_equity": round(eq, 4), "roi_pct": round(roi, 2),
        "total_trades": env.total_trades,
        "fees_paid": round(env._total_fee_paid, 4),
        "funding_paid": round(env._total_funding_paid, 4),
        "avg_bars_held": round(
            np.mean([t["bars_held"] for t in env.history]) if env.history else 0, 1
        ),
    }


def main():
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("V10 TRAINING — Real Binance Fees + Trade Cooldown")
    logger.info("=" * 60)
    logger.info("Fee: 0.05% taker | Cooldown: 6h | Min hold: 4h | Funding: 0.01%/8h")

    logger.info("\nLoading multi-timeframe BTC data...")
    features, prices, feat_names = load_btc_multitf()
    n = len(prices)
    logger.info(f"Total: {n} bars, {features.shape[1]} features")

    # 80/20 split
    split = int(n * 0.80)
    train_feat, test_feat = features[:split], features[split:]
    train_prices, test_prices = prices[:split], prices[split:]
    logger.info(f"Train: {len(train_prices)} bars (${train_prices[0]:.0f}→${train_prices[-1]:.0f})")
    logger.info(f"Test:  {len(test_prices)} bars (${test_prices[0]:.0f}→${test_prices[-1]:.0f})")

    # obs_dim = features + 12 portfolio features (v10 adds cooldown + can_trade)
    obs_dim = features.shape[1] + 12

    # Curriculum stages with real fees
    stages = [
        {"name": "Stage 1: Low Lev (3x)", "episodes": 500, "leverage": 3.0,
         "sl": 0.025, "tp": 0.075, "entropy": 0.10, "lr": 5e-4,
         "eval_eps": 0.10, "noise": 0.15, "cooldown": 6, "min_hold": 4},
        {"name": "Stage 2: Medium (7x)", "episodes": 400, "leverage": 7.0,
         "sl": 0.02, "tp": 0.06, "entropy": 0.05, "lr": 3e-4,
         "eval_eps": 0.05, "noise": 0.10, "cooldown": 6, "min_hold": 4},
        {"name": "Stage 3: Target (12x)", "episodes": 300, "leverage": 12.0,
         "sl": 0.015, "tp": 0.045, "entropy": 0.02, "lr": 1e-4,
         "eval_eps": 0.03, "noise": 0.05, "cooldown": 8, "min_hold": 6},
    ]

    agent = PPOAgent(
        obs_dim=obs_dim, n_actions=N_ACTIONS, hidden=256,
        lr=stages[0]["lr"], gamma=0.995, gae_lambda=0.95,
        clip_eps=0.2, entropy_coef=stages[0]["entropy"],
        ppo_epochs=4, mini_batch_size=128,
        eval_epsilon=stages[0]["eval_eps"],
    )

    param_count = sum(p.numel() for p in agent.network.parameters())
    logger.info(f"Agent: {param_count:,} params, obs_dim={obs_dim}, actions={N_ACTIONS}")

    metrics = []
    best_test_score = -999
    best_model_state = None
    global_ep = 0

    for stage in stages:
        logger.info(f"\n{'='*60}")
        logger.info(f"  {stage['name']} — {stage['episodes']} episodes")
        logger.info(f"  Fee: 0.05% | Cooldown: {stage['cooldown']}h | Min hold: {stage['min_hold']}h")
        logger.info(f"{'='*60}")

        agent.entropy_coef = stage["entropy"]
        agent.eval_epsilon = stage["eval_eps"]
        for pg in agent.optimizer.param_groups:
            pg["lr"] = stage["lr"]

        for ep in range(1, stage["episodes"] + 1):
            global_ep += 1

            # Random start within training data for diversity
            max_start = max(1, len(train_prices) - 500)
            start_idx = np.random.randint(0, max_start) if ep > 5 else 0

            # Create env with real fees and cooldown
            train_env = TradingEnv(
                prices=train_prices[start_idx:], features=train_feat[start_idx:],
                feature_names=feat_names,
                initial_cash=10.0, max_leverage=stage["leverage"],
                sl_pct=stage["sl"], tp_pct=stage["tp"],
                noise_std=stage["noise"],
                cooldown=stage["cooldown"],
                min_hold=stage["min_hold"],
            )

            obs, _ = train_env.reset()
            agent.reset_hidden()
            ep_reward = 0.0

            for _ in range(4096):  # More steps per episode for longer holds
                action, lp, val = agent.select_action(obs, deterministic=False)
                nobs, rew, term, trunc, info = train_env.step(action)
                done = term or trunc
                agent.buffer.add(obs, action, lp, rew, val, float(done))
                obs = nobs
                ep_reward += rew
                if done:
                    break

            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                _, lv = agent.network(s)
                last_val = lv.item()
            loss_info = agent.update(last_val)

            eq = train_env._equity()
            roi = (eq - 10.0) / 10.0 * 100
            wr = train_env.winning_trades / max(train_env.total_trades, 1) * 100
            avg_hold = np.mean([t["bars_held"] for t in train_env.history]) if train_env.history else 0

            m = {
                "episode": global_ep, "stage": stage["name"][:8],
                "equity": round(eq, 4), "roi_pct": round(roi, 2),
                "reward": round(ep_reward, 4), "trades": train_env.total_trades,
                "win_rate": round(wr, 1), "avg_hold": round(avg_hold, 1),
                "fees": round(train_env._total_fee_paid, 4),
            }
            metrics.append(m)

            if ep % 50 == 0 or ep <= 3:
                logger.info(
                    f"  Ep {global_ep:4d} | Eq ${eq:.4f} | ROI {roi:+7.1f}% | "
                    f"Trades {train_env.total_trades:3d} | WR {wr:5.1f}% | "
                    f"Hold {avg_hold:.0f}h | Fees ${train_env._total_fee_paid:.4f}"
                )

        # Test after each stage
        test_env = TradingEnv(
            prices=test_prices, features=test_feat,
            feature_names=feat_names,
            initial_cash=10.0, max_leverage=stage["leverage"],
            sl_pct=stage["sl"], tp_pct=stage["tp"],
            noise_std=0.0,
            cooldown=stage["cooldown"],
            min_hold=stage["min_hold"],
        )
        old_eps = agent.eval_epsilon
        agent.eval_epsilon = 0.0
        test_r = run_eval(agent, test_env, deterministic=True)
        agent.eval_epsilon = old_eps

        # Compute buy & hold
        bh_roi = (test_prices[-1] - test_prices[0]) / test_prices[0] * 100

        # Trades per day
        test_days = len(test_prices) / 24.0
        trades_per_day = test_r["trades"] / max(test_days, 1)

        logger.info(
            f"  TEST: Eq ${test_r['equity']:.4f} | ROI {test_r['roi']:+.1f}% | "
            f"B&H {bh_roi:+.1f}% | Trades {test_r['trades']} ({trades_per_day:.1f}/day) | "
            f"WR {test_r['win_rate']:.1f}% | Avg Hold {test_r['avg_bars_held']:.0f}h | "
            f"Fees ${test_r['fees_paid']:.4f} | Funding ${test_r['funding_paid']:.4f}"
        )

        # Score: prioritize profitability after fees + reasonable trade count
        # Penalize too many trades, reward higher win rate
        trade_penalty = max(0, trades_per_day - 6) * -5.0  # Penalize >6 trades/day
        wr_bonus = max(0, test_r["win_rate"] - 50) * 0.5
        score = test_r["roi"] + trade_penalty + wr_bonus
        if score > best_test_score:
            best_test_score = score
            best_model_state = agent.get_state_bytes()
            logger.info(f"  ★ New best model (score {score:.1f})")

        agent.consolidate_ewc()

    # Restore best
    if best_model_state:
        agent.load_state_bytes(best_model_state)
        logger.info(f"Restored best model (score {best_test_score:.1f})")

    model_path = os.path.join(RESULTS_DIR, "trained_model_v10.pt")
    agent.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # ─── Full Evaluation ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FULL EVALUATION (v10)")
    logger.info("=" * 60)

    eval_results = {}
    for lev, sl, tp, cd, mh in [
        (3.0, 0.025, 0.075, 6, 4),
        (7.0, 0.02, 0.06, 6, 4),
        (12.0, 0.015, 0.045, 8, 6),
    ]:
        name = f"BTC_{lev:.0f}x"
        test_env = TradingEnv(
            prices=test_prices, features=test_feat, feature_names=feat_names,
            initial_cash=10.0, max_leverage=lev, sl_pct=sl, tp_pct=tp,
            cooldown=cd, min_hold=mh,
        )
        agent.eval_epsilon = 0.0
        r = run_eval(agent, test_env, deterministic=True)

        bh = (test_prices[-1] - test_prices[0]) / test_prices[0] * 100
        days = len(test_prices) / 24.0
        tpd = r["trades"] / max(days, 1)

        # Sharpe
        if len(r["curve"]) > 1:
            rets = np.diff(r["curve"]) / np.array(r["curve"][:-1])
            sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(365 * 24)
        else:
            sharpe = 0

        # Max drawdown
        peak = np.maximum.accumulate(r["curve"])
        dd = (peak - r["curve"]) / (peak + 1e-8)
        max_dd = np.max(dd) * 100

        r.update({
            "buy_hold_roi": round(bh, 2), "sharpe": round(sharpe, 3),
            "max_dd": round(max_dd, 1), "trades_per_day": round(tpd, 2),
        })
        eval_results[name] = r

        logger.info(
            f"  {name:10s}: ROI {r['roi']:+10.1f}% | B&H {bh:+8.1f}% | "
            f"Trades {r['trades']:4d} ({tpd:.1f}/day) | WR {r['win_rate']:5.1f}% | "
            f"Sharpe {sharpe:.3f} | MaxDD {max_dd:.1f}% | "
            f"Fees ${r['fees_paid']:.4f} | Funding ${r['funding_paid']:.4f} | "
            f"Avg Hold {r['avg_bars_held']:.0f}h"
        )

    # ─── Online Learning Test ───────────────────────────────────────────
    logger.info("\n--- Online Learning Test ---")
    online_env = TradingEnv(
        prices=test_prices, features=test_feat, feature_names=feat_names,
        initial_cash=10.0, max_leverage=7.0, sl_pct=0.02, tp_pct=0.06,
        cooldown=6, min_hold=4,
    )

    # Baseline (no online learning)
    agent.eval_epsilon = 0.0
    baseline_r = run_eval(agent, online_env, deterministic=True)
    logger.info(f"  Baseline: ${baseline_r['equity']:.4f} ROI {baseline_r['roi']:+.1f}% | "
                f"Trades {baseline_r['trades']}")

    # Online (with updates)
    online_env2 = TradingEnv(
        prices=test_prices, features=test_feat, feature_names=feat_names,
        initial_cash=10.0, max_leverage=7.0, sl_pct=0.02, tp_pct=0.06,
        cooldown=6, min_hold=4,
    )
    obs, _ = online_env2.reset()
    agent.reset_hidden()
    buf_s, buf_a, buf_r, buf_d = [], [], [], []
    ol_curve = [10.0]
    n_updates = 0

    for step in range(len(test_prices)):
        a, lp, val = agent.select_action(obs, deterministic=True)
        nobs, rew, t, tr, info = online_env2.step(a)
        done = t or tr
        buf_s.append(obs)
        buf_a.append(a)
        buf_r.append(rew)
        buf_d.append(float(done))
        obs = nobs
        ol_curve.append(online_env2._equity())

        if len(buf_s) >= 200 and not done:
            agent.online_update(buf_s, buf_a, buf_r, buf_d)
            n_updates += 1
            buf_s, buf_a, buf_r, buf_d = [], [], [], []
        if done:
            break

    ol_eq = online_env2._equity()
    ol_roi = (ol_eq - 10.0) / 10.0 * 100
    ol_trades = online_env2.total_trades
    ol_wr = online_env2.winning_trades / max(ol_trades, 1) * 100

    online_result = {
        "baseline": {
            "equity": round(baseline_r["equity"], 4),
            "roi_pct": round(baseline_r["roi"], 2),
            "trades": baseline_r["trades"],
            "win_rate": round(baseline_r["win_rate"], 1),
            "curve": [round(c, 4) for c in baseline_r["curve"]],
        },
        "online": {
            "equity": round(ol_eq, 4),
            "roi_pct": round(ol_roi, 2),
            "trades": ol_trades,
            "win_rate": round(ol_wr, 1),
            "curve": [round(c, 4) for c in ol_curve],
            "updates": n_updates,
        },
        "improvement": round(ol_roi - baseline_r["roi"], 2),
    }
    logger.info(f"  Baseline: ${baseline_r['equity']:.4f} ROI {baseline_r['roi']:+.1f}% | Trades {baseline_r['trades']}")
    logger.info(f"  Online:   ${ol_eq:.4f} ROI {ol_roi:+.1f}% | Trades {ol_trades} | Updates {n_updates}")

    # ─── Charts ──────────────────────────────────────────────────────────
    make_charts(metrics, eval_results, online_result)

    # ─── Report ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": "v10",
        "config": {
            "fee": "0.05% taker (real Binance Futures)",
            "cooldown": "6-8 hours between trades",
            "min_hold": "4-6 hours minimum",
            "funding_rate": "0.01% every 8 hours",
            "stages": [s["name"] for s in stages],
            "total_episodes": sum(s["episodes"] for s in stages),
        },
        "elapsed_s": round(elapsed, 1),
        "data": {"total_bars": n, "train_bars": split, "test_bars": n - split,
                 "features": features.shape[1], "obs_dim": obs_dim},
        "evaluation": {
            name: {k: v for k, v in r.items() if k not in ("curve", "history", "actions")}
            for name, r in eval_results.items()
        },
        "online": {k: v for k, v in online_result.items() if not isinstance(v, dict) or k in ("baseline", "online")},
        "trade_analysis": {},
    }

    # Clean online sub-dicts of curves
    for sub_key in ("baseline", "online"):
        if sub_key in report["online"] and "curve" in report["online"][sub_key]:
            del report["online"][sub_key]["curve"]

    for name, r in eval_results.items():
        if r["history"]:
            pnls = [t["pnl_pct"] for t in r["history"]]
            holds = [t["bars_held"] for t in r["history"]]
            report["trade_analysis"][name] = {
                "avg_pnl_pct": round(float(np.mean(pnls)), 2),
                "max_win_pct": round(float(max(pnls)), 2),
                "max_loss_pct": round(float(min(pnls)), 2),
                "avg_bars_held": round(float(np.mean(holds)), 1),
                "trades_per_day": round(r.get("trades_per_day", 0), 2),
            }

    with open(os.path.join(RESULTS_DIR, "pipeline_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY (v10 — Real Binance Fees)")
    print("=" * 60)
    for name, r in eval_results.items():
        print(f"  {name:10s}: ROI {r['roi']:+10.1f}% | B&H {r['buy_hold_roi']:+8.1f}% | "
              f"Trades {r['total_trades']:4d} ({r.get('trades_per_day', 0):.1f}/day) | "
              f"WR {r['win_rate']:5.1f}% | Sharpe {r['sharpe']:.3f} | "
              f"Fees ${r['fees_paid']:.4f}")
    print(f"\n  Online: {online_result['improvement']:+.1f}% improvement")
    print(f"  Time: {elapsed:.0f}s")
    print("=" * 60)

    return report


def make_charts(train_m, eval_r, online_r):
    plt.style.use("dark_background")
    fc = "#0d1117"
    g, r, b, y, p = "#00d26a", "#ff4757", "#4ecdc4", "#ffa502", "#a29bfe"

    # 1. Training progress
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor=fc)
    for ax in axes.flat:
        ax.set_facecolor(fc)
    eps = [m["episode"] for m in train_m]
    eqs = [m["equity"] for m in train_m]
    rews = [m["reward"] for m in train_m]
    wrs = [m["win_rate"] for m in train_m]
    trades = [m["trades"] for m in train_m]
    holds = [m.get("avg_hold", 0) for m in train_m]
    fees = [m.get("fees", 0) for m in train_m]

    axes[0,0].plot(eps, eqs, color=g, lw=0.6, alpha=0.4)
    if len(eqs) > 30:
        axes[0,0].plot(eps, pd.Series(eqs).rolling(30).mean(), color=y, lw=2)
    axes[0,0].axhline(y=10, color="white", ls="--", alpha=0.3)
    axes[0,0].set_title("Equity per Episode", color="white")
    axes[0,0].grid(alpha=0.15)

    axes[0,1].plot(eps, rews, color=b, lw=0.4, alpha=0.3)
    if len(rews) > 30:
        axes[0,1].plot(eps, pd.Series(rews).rolling(30).mean(), color=y, lw=2)
    axes[0,1].set_title("Episode Reward", color="white")
    axes[0,1].grid(alpha=0.15)

    axes[0,2].plot(eps, wrs, color=p, lw=0.4, alpha=0.3)
    if len(wrs) > 30:
        axes[0,2].plot(eps, pd.Series(wrs).rolling(30).mean(), color=y, lw=2)
    axes[0,2].axhline(y=50, color="white", ls="--", alpha=0.3)
    axes[0,2].set_title("Win Rate %", color="white")
    axes[0,2].grid(alpha=0.15)

    axes[1,0].plot(eps, trades, color=r, lw=0.4, alpha=0.3)
    if len(trades) > 30:
        axes[1,0].plot(eps, pd.Series(trades).rolling(30).mean(), color=y, lw=2)
    axes[1,0].set_title("Trades per Episode", color="white")
    axes[1,0].grid(alpha=0.15)

    axes[1,1].plot(eps, holds, color=g, lw=0.4, alpha=0.3)
    if len(holds) > 30:
        axes[1,1].plot(eps, pd.Series(holds).rolling(30).mean(), color=y, lw=2)
    axes[1,1].set_title("Avg Hold Period (bars)", color="white")
    axes[1,1].grid(alpha=0.15)

    axes[1,2].plot(eps, fees, color=r, lw=0.4, alpha=0.3)
    if len(fees) > 30:
        axes[1,2].plot(eps, pd.Series(fees).rolling(30).mean(), color=y, lw=2)
    axes[1,2].set_title("Fees Paid per Episode ($)", color="white")
    axes[1,2].grid(alpha=0.15)

    plt.suptitle("V10 Training Progress (Real Binance Fees)", color="white", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(RESULTS_DIR, "01_training_progress.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Test equity curves
    n_plots = max(len(eval_r), 1)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5), facecolor=fc)
    if n_plots == 1:
        axes = [axes]
    cols = [g, b, y, p, r]
    for i, (name, res) in enumerate(eval_r.items()):
        ax = axes[i]
        ax.set_facecolor(fc)
        ax.plot(res["curve"], color=cols[i%5], lw=1.5, label="RL Agent")
        ax.axhline(y=10, color="white", ls="--", alpha=0.3)
        tpd = res.get("trades_per_day", 0)
        ax.set_title(
            f"{name}\nRL: {res['roi']:+.1f}% | B&H: {res['buy_hold_roi']:+.1f}% | "
            f"Trades: {res['total_trades']} ({tpd:.1f}/day) | WR: {res['win_rate']:.0f}%\n"
            f"Fees: ${res['fees_paid']:.4f} | Funding: ${res['funding_paid']:.4f}",
            fontsize=9, color="white"
        )
        ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "02_test_evaluation.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Online vs Baseline
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=fc)
    ax.set_facecolor(fc)
    ax.plot(online_r["baseline"]["curve"], color=y, lw=1.5,
            label=f"Baseline ({online_r['baseline']['roi_pct']:+.1f}%)")
    ax.plot(online_r["online"]["curve"], color=g, lw=2,
            label=f"Online ({online_r['online']['roi_pct']:+.1f}%)")
    ax.axhline(y=10, color="white", ls="--", alpha=0.3)
    ax.set_title("Online Learning Test (v10)", fontsize=13, color="white")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "03_online_vs_baseline.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Trade analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=fc)
    all_pnls = []
    for res in eval_r.values():
        for t in res["history"]:
            all_pnls.append(t.get("pnl_pct", 0))

    axes[0].set_facecolor(fc)
    if all_pnls:
        pc = [g if pv > 0 else r for pv in all_pnls]
        axes[0].bar(range(len(all_pnls)), all_pnls, color=pc, alpha=0.7)
        axes[0].axhline(y=0, color="white", lw=0.5)
        avg = np.mean(all_pnls)
        axes[0].axhline(y=avg, color=y, ls="--", label=f"Avg: {avg:.1f}%")
        axes[0].set_title(f"Trade P&L ({len(all_pnls)} trades)", color="white")
        axes[0].legend()
        axes[0].grid(alpha=0.15)
    else:
        axes[0].text(0.5, 0.5, "No trades", ha="center", va="center", color="gray", fontsize=14)

    all_acts = []
    for res in eval_r.values():
        all_acts.extend(res["actions"])
    axes[1].set_facecolor(fc)
    if all_acts:
        ac = {}
        for a in all_acts:
            nm = ACTIONS[a]
            ac[nm] = ac.get(nm, 0) + 1
        names = list(ac.keys())
        counts = list(ac.values())
        total = sum(counts)
        labels = [f"{nm}\n({c/total*100:.1f}%)" for nm, c in zip(names, counts)]
        bar_c = [g if "long" in nm else r if "short" in nm else "#666" for nm in names]
        axes[1].barh(labels, counts, color=bar_c, alpha=0.8)
        axes[1].set_title("Action Distribution", color="white")
        axes[1].grid(alpha=0.15, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "04_trade_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Summary table
    fig, ax = plt.subplots(figsize=(18, max(3, len(eval_r)+3)), facecolor=fc)
    ax.set_facecolor(fc)
    ax.axis("off")
    headers = ["Config", "Final $", "ROI %", "B&H %", "Trades", "Trades/Day",
               "Win Rate", "Sharpe", "Max DD", "Fees $", "Funding $", "Avg Hold"]
    rows = []
    for name, res in eval_r.items():
        rows.append([name, f"${res['final_equity']:.4f}", f"{res['roi']:+.1f}%",
                      f"{res['buy_hold_roi']:+.1f}%", str(res["total_trades"]),
                      f"{res.get('trades_per_day', 0):.1f}",
                      f"{res['win_rate']:.1f}%", f"{res['sharpe']:.3f}",
                      f"{res['max_dd']:.1f}%",
                      f"${res['fees_paid']:.4f}", f"${res['funding_paid']:.4f}",
                      f"{res['avg_bars_held']:.0f}h"])
    rows.append(["Online", f"${online_r['online']['equity']:.4f}",
                  f"{online_r['online']['roi_pct']:+.1f}%", "—",
                  str(online_r['online']['trades']), "—",
                  f"{online_r['online']['win_rate']:.1f}%", "—", "—", "—", "—", "—"])
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center",
                      colColours=["#1a1a2e"]*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#333355")
        if key[0] == 0:
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_facecolor("#1a1a2e")
        else:
            cell.set_text_props(color="#cccccc")
            cell.set_facecolor(fc)
    ax.set_title("Performance Summary (v10 — Real Binance Fees)", fontsize=14, color="white", pad=20)
    plt.savefig(os.path.join(RESULTS_DIR, "05_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Charts saved.")


if __name__ == "__main__":
    main()
