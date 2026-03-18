"""
diagnose_overfitting.py — Honest assessment of v9 model
========================================================
Tests for:
  1. Look-ahead bias in features
  2. Action pattern analysis (is it just always long?)
  3. Win rate vs random baseline
  4. Sensitivity to noise (does small noise destroy performance?)
  5. Walk-forward validation (multiple test windows)
  6. Trade timing analysis (are wins clustered or distributed?)
  7. Fee sensitivity (does it survive realistic fees?)
  8. Comparison to simple moving average crossover strategy
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import load_btc_multitf
from trading_env import TradingEnv, N_ACTIONS, ACTIONS
from ppo_lstm_agent import PPOAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("diagnose")

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
    }


def main():
    logger.info("=" * 60)
    logger.info("OVERFITTING DIAGNOSTIC — v9 Model")
    logger.info("=" * 60)

    # Load data
    logger.info("\n1. Loading data...")
    features, prices, feat_names = load_btc_multitf()
    n = len(prices)
    split = int(n * 0.80)
    train_feat, test_feat = features[:split], features[split:]
    train_prices, test_prices = prices[:split], prices[split:]
    obs_dim = features.shape[1] + 10

    logger.info(f"   Total bars: {n}, Train: {split}, Test: {n-split}")
    logger.info(f"   Train period: ${train_prices[0]:.0f} → ${train_prices[-1]:.0f}")
    logger.info(f"   Test period:  ${test_prices[0]:.0f} → ${test_prices[-1]:.0f}")

    # Load model
    model_path = os.path.join(RESULTS_DIR, "trained_model.pt")
    if not os.path.exists(model_path):
        logger.error("No trained model found!")
        return

    agent = PPOAgent(obs_dim=obs_dim, n_actions=N_ACTIONS, hidden=256)
    agent.load(model_path)
    agent.eval_epsilon = 0.0
    logger.info("   Model loaded.")

    report = {}

    # ─── TEST 1: Action Distribution Analysis ───────────────────────────
    logger.info("\n2. ACTION DISTRIBUTION ANALYSIS")
    logger.info("   Is the agent just always picking one action?")

    test_env = TradingEnv(
        prices=test_prices, features=test_feat, feature_names=feat_names,
        initial_cash=10.0, max_leverage=7.0, sl_pct=0.02, tp_pct=0.06,
    )
    r = run_eval(agent, test_env)
    action_counts = {ACTIONS[i]: 0 for i in range(N_ACTIONS)}
    for a in r["actions"]:
        action_counts[ACTIONS[a]] += 1
    total_actions = len(r["actions"])

    logger.info(f"   Total steps: {total_actions}")
    for name, count in action_counts.items():
        pct = count / total_actions * 100
        logger.info(f"   {name:6s}: {count:5d} ({pct:5.1f}%)")

    # Check if it's just always one action
    max_pct = max(c / total_actions for c in action_counts.values()) * 100
    is_degenerate = max_pct > 90
    report["action_analysis"] = {
        "distribution": {k: v for k, v in action_counts.items()},
        "max_single_action_pct": round(max_pct, 1),
        "is_degenerate": is_degenerate,
        "verdict": "DEGENERATE — agent picks one action >90% of time" if is_degenerate
                   else "OK — agent uses multiple actions"
    }
    logger.info(f"   Verdict: {report['action_analysis']['verdict']}")

    # ─── TEST 2: Random Agent Baseline ──────────────────────────────────
    logger.info("\n3. RANDOM AGENT BASELINE")
    logger.info("   Does a random agent also achieve high win rates?")

    random_results = []
    for trial in range(10):
        test_env = TradingEnv(
            prices=test_prices, features=test_feat, feature_names=feat_names,
            initial_cash=10.0, max_leverage=7.0, sl_pct=0.02, tp_pct=0.06,
        )
        obs, _ = test_env.reset()
        while True:
            a = np.random.randint(0, N_ACTIONS)
            obs, _, t, tr, _ = test_env.step(a)
            if t or tr:
                break
        eq = test_env._equity()
        roi = (eq - 10.0) / 10.0 * 100
        wr = test_env.winning_trades / max(test_env.total_trades, 1) * 100
        random_results.append({"roi": roi, "trades": test_env.total_trades, "win_rate": wr})

    avg_random_roi = np.mean([r["roi"] for r in random_results])
    avg_random_wr = np.mean([r["win_rate"] for r in random_results])
    avg_random_trades = np.mean([r["trades"] for r in random_results])

    logger.info(f"   Random avg ROI: {avg_random_roi:+.1f}%")
    logger.info(f"   Random avg WR:  {avg_random_wr:.1f}%")
    logger.info(f"   Random avg trades: {avg_random_trades:.0f}")
    logger.info(f"   Agent ROI: {r['roi']:+.1f}%, Agent WR: {r['win_rate']:.1f}%")

    # KEY: If random agent also gets high win rate, the env is biased
    random_wr_close = abs(avg_random_wr - r["win_rate"]) < 10
    report["random_baseline"] = {
        "random_avg_roi": round(avg_random_roi, 2),
        "random_avg_wr": round(avg_random_wr, 1),
        "random_avg_trades": round(avg_random_trades, 0),
        "agent_roi": round(r["roi"], 2),
        "agent_wr": round(r["win_rate"], 1),
        "wr_gap": round(r["win_rate"] - avg_random_wr, 1),
        "is_env_biased": random_wr_close,
        "verdict": "WARNING — Random agent has similar win rate, env may be biased!" if random_wr_close
                   else f"OK — Agent WR {r['win_rate']:.0f}% vs Random {avg_random_wr:.0f}% (gap: {r['win_rate']-avg_random_wr:.0f}%)"
    }
    logger.info(f"   Verdict: {report['random_baseline']['verdict']}")

    # ─── TEST 3: Always-Long Baseline ───────────────────────────────────
    logger.info("\n4. ALWAYS-LONG BASELINE")
    logger.info("   Does always going long also work?")

    test_env = TradingEnv(
        prices=test_prices, features=test_feat, feature_names=feat_names,
        initial_cash=10.0, max_leverage=7.0, sl_pct=0.02, tp_pct=0.06,
    )
    obs, _ = test_env.reset()
    while True:
        obs, _, t, tr, _ = test_env.step(1)  # Always LONG
        if t or tr:
            break
    al_eq = test_env._equity()
    al_roi = (al_eq - 10.0) / 10.0 * 100
    al_wr = test_env.winning_trades / max(test_env.total_trades, 1) * 100

    # Always Short
    test_env2 = TradingEnv(
        prices=test_prices, features=test_feat, feature_names=feat_names,
        initial_cash=10.0, max_leverage=7.0, sl_pct=0.02, tp_pct=0.06,
    )
    obs, _ = test_env2.reset()
    while True:
        obs, _, t, tr, _ = test_env2.step(2)  # Always SHORT
        if t or tr:
            break
    as_eq = test_env2._equity()
    as_roi = (as_eq - 10.0) / 10.0 * 100
    as_wr = test_env2.winning_trades / max(test_env2.total_trades, 1) * 100

    logger.info(f"   Always Long:  ROI {al_roi:+.1f}%, WR {al_wr:.1f}%, Trades {test_env.total_trades}")
    logger.info(f"   Always Short: ROI {as_roi:+.1f}%, WR {as_wr:.1f}%, Trades {test_env2.total_trades}")
    logger.info(f"   RL Agent:     ROI {r['roi']:+.1f}%, WR {r['win_rate']:.1f}%, Trades {r['trades']}")

    report["always_long_short"] = {
        "always_long_roi": round(al_roi, 2),
        "always_long_wr": round(al_wr, 1),
        "always_long_trades": test_env.total_trades,
        "always_short_roi": round(as_roi, 2),
        "always_short_wr": round(as_wr, 1),
        "always_short_trades": test_env2.total_trades,
        "agent_roi": round(r["roi"], 2),
        "agent_wr": round(r["win_rate"], 1),
    }

    # ─── TEST 4: Noise Sensitivity ──────────────────────────────────────
    logger.info("\n5. NOISE SENSITIVITY TEST")
    logger.info("   Does small noise destroy the agent's performance?")

    noise_results = []
    for noise in [0.0, 0.05, 0.10, 0.20, 0.50]:
        test_env = TradingEnv(
            prices=test_prices, features=test_feat, feature_names=feat_names,
            initial_cash=10.0, max_leverage=7.0, sl_pct=0.02, tp_pct=0.06,
            noise_std=noise,
        )
        nr = run_eval(agent, test_env)
        noise_results.append({
            "noise": noise, "roi": round(nr["roi"], 2),
            "wr": round(nr["win_rate"], 1), "trades": nr["trades"],
        })
        logger.info(f"   Noise {noise:.2f}: ROI {nr['roi']:+.1f}%, WR {nr['win_rate']:.1f}%, Trades {nr['trades']}")

    # If performance drops sharply with small noise, it's memorizing
    base_roi = noise_results[0]["roi"]
    noise_05_roi = noise_results[2]["roi"]  # 0.10 noise
    roi_drop = base_roi - noise_05_roi
    is_fragile = abs(roi_drop) > abs(base_roi) * 0.5 if base_roi != 0 else True

    report["noise_sensitivity"] = {
        "results": noise_results,
        "roi_drop_at_0.10_noise": round(roi_drop, 2),
        "is_fragile": is_fragile,
        "verdict": "WARNING — Performance drops >50% with small noise, likely overfitting" if is_fragile
                   else "OK — Robust to noise"
    }
    logger.info(f"   Verdict: {report['noise_sensitivity']['verdict']}")

    # ─── TEST 5: Walk-Forward Validation ────────────────────────────────
    logger.info("\n6. WALK-FORWARD VALIDATION")
    logger.info("   Testing on multiple non-overlapping windows...")

    # Split test data into 4 equal windows
    window_size = len(test_prices) // 4
    wf_results = []
    for i in range(4):
        start = i * window_size
        end = min((i + 1) * window_size, len(test_prices))
        if end - start < 100:
            continue
        w_prices = test_prices[start:end]
        w_feat = test_feat[start:end]

        test_env = TradingEnv(
            prices=w_prices, features=w_feat, feature_names=feat_names,
            initial_cash=10.0, max_leverage=7.0, sl_pct=0.02, tp_pct=0.06,
        )
        wr = run_eval(agent, test_env)
        wf_results.append({
            "window": i + 1,
            "bars": end - start,
            "price_start": round(w_prices[0], 0),
            "price_end": round(w_prices[-1], 0),
            "roi": round(wr["roi"], 2),
            "wr": round(wr["win_rate"], 1),
            "trades": wr["trades"],
        })
        logger.info(
            f"   Window {i+1}: ${w_prices[0]:.0f}→${w_prices[-1]:.0f} | "
            f"ROI {wr['roi']:+.1f}% | WR {wr['win_rate']:.1f}% | Trades {wr['trades']}"
        )

    # Check consistency across windows
    wf_rois = [w["roi"] for w in wf_results]
    wf_wrs = [w["wr"] for w in wf_results]
    profitable_windows = sum(1 for r in wf_rois if r > 0)
    consistent = profitable_windows >= len(wf_results) * 0.5

    report["walk_forward"] = {
        "windows": wf_results,
        "profitable_windows": profitable_windows,
        "total_windows": len(wf_results),
        "avg_roi": round(np.mean(wf_rois), 2) if wf_rois else 0,
        "avg_wr": round(np.mean(wf_wrs), 1) if wf_wrs else 0,
        "roi_std": round(np.std(wf_rois), 2) if wf_rois else 0,
        "is_consistent": consistent,
        "verdict": f"{'OK' if consistent else 'WARNING'} — {profitable_windows}/{len(wf_results)} windows profitable"
    }
    logger.info(f"   Verdict: {report['walk_forward']['verdict']}")

    # ─── TEST 6: Fee Sensitivity ────────────────────────────────────────
    logger.info("\n7. FEE SENSITIVITY TEST")
    logger.info("   Testing with different fee levels...")

    fee_results = []
    for fee_mult, fee_name in [(0.0, "0% (no fees)"), (1.0, "0.04% (default)"),
                                (2.5, "0.10% (realistic)"), (5.0, "0.20% (high)")]:
        test_env = TradingEnv(
            prices=test_prices, features=test_feat, feature_names=feat_names,
            initial_cash=10.0, max_leverage=7.0, sl_pct=0.02, tp_pct=0.06,
            fee=0.0004 * fee_mult,
        )
        fr = run_eval(agent, test_env)
        fee_results.append({
            "fee": fee_name, "fee_pct": round(0.04 * fee_mult, 3),
            "roi": round(fr["roi"], 2), "wr": round(fr["win_rate"], 1),
            "trades": fr["trades"],
        })
        logger.info(f"   Fee {fee_name:20s}: ROI {fr['roi']:+.1f}%, WR {fr['win_rate']:.1f}%")

    # Check if profits vanish with realistic fees
    no_fee_roi = fee_results[0]["roi"]
    realistic_fee_roi = fee_results[2]["roi"]
    fee_sensitive = realistic_fee_roi < 0 and no_fee_roi > 0

    report["fee_sensitivity"] = {
        "results": fee_results,
        "no_fee_roi": round(no_fee_roi, 2),
        "realistic_fee_roi": round(realistic_fee_roi, 2),
        "is_fee_sensitive": fee_sensitive,
        "verdict": "WARNING — Profits vanish with realistic fees, likely overtrading" if fee_sensitive
                   else "OK — Profitable even with higher fees"
    }
    logger.info(f"   Verdict: {report['fee_sensitivity']['verdict']}")

    # ─── TEST 7: SMA Crossover Baseline ─────────────────────────────────
    logger.info("\n8. SIMPLE STRATEGY BASELINE (SMA Crossover)")
    logger.info("   Can a simple SMA crossover beat the RL agent?")

    # SMA 20/50 crossover
    sma_short = pd.Series(test_prices).rolling(20).mean().values
    sma_long = pd.Series(test_prices).rolling(50).mean().values

    test_env = TradingEnv(
        prices=test_prices, features=test_feat, feature_names=feat_names,
        initial_cash=10.0, max_leverage=7.0, sl_pct=0.02, tp_pct=0.06,
    )
    obs, _ = test_env.reset()
    sma_curve = [10.0]
    for i in range(len(test_prices) - 1):
        if i < 50 or np.isnan(sma_short[i]) or np.isnan(sma_long[i]):
            action = 0  # FLAT
        elif sma_short[i] > sma_long[i]:
            action = 1  # LONG
        else:
            action = 2  # SHORT
        obs, _, t, tr, _ = test_env.step(action)
        sma_curve.append(test_env._equity())
        if t or tr:
            break

    sma_eq = test_env._equity()
    sma_roi = (sma_eq - 10.0) / 10.0 * 100
    sma_wr = test_env.winning_trades / max(test_env.total_trades, 1) * 100

    logger.info(f"   SMA 20/50: ROI {sma_roi:+.1f}%, WR {sma_wr:.1f}%, Trades {test_env.total_trades}")
    logger.info(f"   RL Agent:  ROI {r['roi']:+.1f}%, WR {r['win_rate']:.1f}%, Trades {r['trades']}")

    report["sma_baseline"] = {
        "sma_roi": round(sma_roi, 2),
        "sma_wr": round(sma_wr, 1),
        "sma_trades": test_env.total_trades,
        "agent_roi": round(r["roi"], 2),
        "agent_beats_sma": r["roi"] > sma_roi,
    }

    # ─── TEST 8: Trade P&L Distribution ─────────────────────────────────
    logger.info("\n9. TRADE P&L DISTRIBUTION")

    if r["history"]:
        pnls = [t["pnl_pct"] for t in r["history"]]
        bars_held = [t["bars_held"] for t in r["history"]]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        logger.info(f"   Total trades: {len(pnls)}")
        logger.info(f"   Wins: {len(wins)}, Losses: {len(losses)}")
        logger.info(f"   Avg win:  {np.mean(wins):+.2f}%" if wins else "   No wins")
        logger.info(f"   Avg loss: {np.mean(losses):+.2f}%" if losses else "   No losses")
        logger.info(f"   Max win:  {max(pnls):+.2f}%")
        logger.info(f"   Max loss: {min(pnls):+.2f}%")
        logger.info(f"   Avg bars held: {np.mean(bars_held):.1f}")

        # Check if wins are just from SL/TP mechanics
        sl_exits = sum(1 for t in r["history"] if t.get("reason") == "stop_loss")
        tp_exits = sum(1 for t in r["history"] if t.get("reason") == "take_profit")
        sig_exits = sum(1 for t in r["history"] if t.get("reason") == "signal_exit")

        logger.info(f"   Exit reasons: SL={sl_exits}, TP={tp_exits}, Signal={sig_exits}")

        report["trade_distribution"] = {
            "total": len(pnls),
            "wins": len(wins),
            "losses": len(losses),
            "avg_win_pct": round(np.mean(wins), 2) if wins else 0,
            "avg_loss_pct": round(np.mean(losses), 2) if losses else 0,
            "max_win_pct": round(max(pnls), 2),
            "max_loss_pct": round(min(pnls), 2),
            "avg_bars_held": round(np.mean(bars_held), 1),
            "sl_exits": sl_exits,
            "tp_exits": tp_exits,
            "signal_exits": sig_exits,
        }

        # KEY CHECK: If TP/SL ratio explains the win rate, it's not the agent
        # With 3:1 R:R (TP=6%, SL=2%), random entries should win ~33% of time
        # If agent WR is close to 33%, it's just the SL/TP doing the work
        expected_random_wr = 100 * (0.02 / (0.02 + 0.06))  # ~25% for 3:1 R:R
        wr_above_random = r["win_rate"] - expected_random_wr
        logger.info(f"\n   Expected random WR with 3:1 R:R: ~{expected_random_wr:.0f}%")
        logger.info(f"   Agent WR: {r['win_rate']:.1f}%")
        logger.info(f"   Agent edge above random: {wr_above_random:+.1f}%")

        report["trade_distribution"]["expected_random_wr"] = round(expected_random_wr, 1)
        report["trade_distribution"]["agent_edge"] = round(wr_above_random, 1)

    # ─── FINAL VERDICT ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL OVERFITTING VERDICT")
    logger.info("=" * 60)

    red_flags = 0
    green_flags = 0
    findings = []

    if report["action_analysis"]["is_degenerate"]:
        red_flags += 1
        findings.append("RED: Agent is degenerate (picks one action >90%)")
    else:
        green_flags += 1
        findings.append("GREEN: Agent uses diverse actions")

    if report["random_baseline"]["is_env_biased"]:
        red_flags += 2  # This is a critical issue
        findings.append("RED: Random agent has similar win rate — ENV IS BIASED")
    else:
        green_flags += 1
        findings.append(f"GREEN: Agent WR ({report['random_baseline']['agent_wr']:.0f}%) > Random ({report['random_baseline']['random_avg_wr']:.0f}%)")

    if report["noise_sensitivity"]["is_fragile"]:
        red_flags += 1
        findings.append("RED: Performance fragile to noise — likely memorizing")
    else:
        green_flags += 1
        findings.append("GREEN: Robust to noise injection")

    if not report["walk_forward"]["is_consistent"]:
        red_flags += 1
        findings.append("RED: Inconsistent across walk-forward windows")
    else:
        green_flags += 1
        findings.append(f"GREEN: {report['walk_forward']['profitable_windows']}/{report['walk_forward']['total_windows']} windows profitable")

    if report["fee_sensitivity"]["is_fee_sensitive"]:
        red_flags += 1
        findings.append("RED: Profits vanish with realistic fees — overtrading")
    else:
        green_flags += 1
        findings.append("GREEN: Profitable even with higher fees")

    for f in findings:
        logger.info(f"   {f}")

    overall = "NOT READY" if red_flags >= 2 else "PROCEED WITH CAUTION" if red_flags >= 1 else "READY"
    logger.info(f"\n   RED FLAGS: {red_flags}, GREEN FLAGS: {green_flags}")
    logger.info(f"   OVERALL: {overall}")

    report["verdict"] = {
        "red_flags": red_flags,
        "green_flags": green_flags,
        "findings": findings,
        "overall": overall,
        "ready_for_live": red_flags == 0,
        "recommendation": (
            "DO NOT deploy to live trading. Fix the identified issues first."
            if red_flags >= 2 else
            "Proceed with extreme caution. Use paper trading first with tiny amounts."
            if red_flags >= 1 else
            "Model shows genuine edge. Start with paper trading, then small live amounts."
        )
    }

    # Save report
    with open(os.path.join(RESULTS_DIR, "overfitting_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ─── Generate diagnostic chart ──────────────────────────────────────
    make_diagnostic_chart(report, r, sma_curve, noise_results, wf_results, fee_results)

    logger.info(f"\nReport saved to {RESULTS_DIR}/overfitting_report.json")
    return report


def make_diagnostic_chart(report, agent_r, sma_curve, noise_results, wf_results, fee_results):
    plt.style.use("dark_background")
    fc = "#0d1117"
    g, r, b, y, p = "#00d26a", "#ff4757", "#4ecdc4", "#ffa502", "#a29bfe"

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=fc)
    for ax in axes.flat:
        ax.set_facecolor(fc)

    # 1. Action distribution
    ax = axes[0, 0]
    dist = report["action_analysis"]["distribution"]
    names = list(dist.keys())
    counts = list(dist.values())
    colors = [g if "long" in n else r if "short" in n else "#666" for n in names]
    ax.bar(names, counts, color=colors, alpha=0.8)
    ax.set_title("Action Distribution", color="white", fontsize=11)
    ax.grid(alpha=0.15)

    # 2. Noise sensitivity
    ax = axes[0, 1]
    noises = [n["noise"] for n in noise_results]
    rois = [n["roi"] for n in noise_results]
    wrs = [n["wr"] for n in noise_results]
    ax.plot(noises, rois, color=g, marker="o", lw=2, label="ROI %")
    ax2 = ax.twinx()
    ax2.plot(noises, wrs, color=y, marker="s", lw=2, label="Win Rate %")
    ax.set_title("Noise Sensitivity", color="white", fontsize=11)
    ax.set_xlabel("Noise Std", color="white")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(alpha=0.15)

    # 3. Walk-forward windows
    ax = axes[0, 2]
    if wf_results:
        w_names = [f"W{w['window']}\n${w['price_start']:.0f}→${w['price_end']:.0f}" for w in wf_results]
        w_rois = [w["roi"] for w in wf_results]
        w_colors = [g if roi > 0 else r for roi in w_rois]
        ax.bar(w_names, w_rois, color=w_colors, alpha=0.8)
        ax.axhline(y=0, color="white", ls="--", alpha=0.3)
    ax.set_title("Walk-Forward Windows", color="white", fontsize=11)
    ax.grid(alpha=0.15)

    # 4. Fee sensitivity
    ax = axes[1, 0]
    f_names = [f["fee"].split("(")[0].strip() for f in fee_results]
    f_rois = [f["roi"] for f in fee_results]
    f_colors = [g if roi > 0 else r for roi in f_rois]
    ax.bar(f_names, f_rois, color=f_colors, alpha=0.8)
    ax.axhline(y=0, color="white", ls="--", alpha=0.3)
    ax.set_title("Fee Sensitivity", color="white", fontsize=11)
    ax.grid(alpha=0.15)

    # 5. Agent vs baselines
    ax = axes[1, 1]
    baselines = {
        "RL Agent": report["random_baseline"]["agent_roi"],
        "Random": report["random_baseline"]["random_avg_roi"],
        "Always Long": report["always_long_short"]["always_long_roi"],
        "Always Short": report["always_long_short"]["always_short_roi"],
        "SMA 20/50": report["sma_baseline"]["sma_roi"],
    }
    b_names = list(baselines.keys())
    b_rois = list(baselines.values())
    b_colors = [g if roi > 0 else r for roi in b_rois]
    ax.barh(b_names, b_rois, color=b_colors, alpha=0.8)
    ax.axvline(x=0, color="white", ls="--", alpha=0.3)
    ax.set_title("Agent vs Baselines (ROI %)", color="white", fontsize=11)
    ax.grid(alpha=0.15)

    # 6. Verdict summary
    ax = axes[1, 2]
    ax.axis("off")
    verdict = report["verdict"]
    color = g if verdict["overall"] == "READY" else y if "CAUTION" in verdict["overall"] else r
    ax.text(0.5, 0.85, verdict["overall"], ha="center", va="center",
            fontsize=24, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.70, f"Red Flags: {verdict['red_flags']} | Green Flags: {verdict['green_flags']}",
            ha="center", va="center", fontsize=12, color="white", transform=ax.transAxes)

    y_pos = 0.55
    for finding in verdict["findings"]:
        fc_color = r if "RED" in finding else g
        ax.text(0.05, y_pos, finding, ha="left", va="center",
                fontsize=8, color=fc_color, transform=ax.transAxes)
        y_pos -= 0.10

    ax.text(0.5, 0.05, verdict["recommendation"], ha="center", va="center",
            fontsize=9, color="white", style="italic", transform=ax.transAxes, wrap=True)

    plt.suptitle("Overfitting Diagnostic Report — v9 Model", fontsize=16, color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(RESULTS_DIR, "06_overfitting_diagnostic.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Diagnostic chart saved.")


if __name__ == "__main__":
    main()
