import { z } from "zod";
import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import { invokeLLM } from "./_core/llm";
import { notifyOwner } from "./_core/notification";
import { storagePut } from "./storage";
import * as fs from "fs";
import * as path from "path";
import {
  getRecentTrades, getOpenTrades, getTradeStats, insertTrade,
  getRecentSnapshots, insertSnapshot,
  getTrainingRuns, insertTrainingRun, updateTrainingRun,
  getModelVersions, getActiveModel, insertModelVersion,
  getAgentLogs, insertAgentLog,
} from "./db";

const ENGINE_URL = "http://127.0.0.1:8100";

async function engineFetch(path: string, options?: RequestInit) {
  try {
    const resp = await fetch(`${ENGINE_URL}${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...options?.headers },
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Engine error ${resp.status}: ${text}`);
    }
    return resp.json();
  } catch (e: any) {
    if (e.cause?.code === "ECONNREFUSED") {
      throw new Error("RL Engine not running. Start it with: cd engine && python3 api_server.py");
    }
    throw e;
  }
}

// ─── Trading router ──────────────────────────────────────────────────────────

const tradingRouter = router({
  recentTrades: publicProcedure
    .input(z.object({ limit: z.number().default(50) }).optional())
    .query(async ({ input }) => {
      return getRecentTrades(input?.limit ?? 50);
    }),

  openTrades: publicProcedure.query(async () => {
    return getOpenTrades();
  }),

  stats: publicProcedure.query(async () => {
    return getTradeStats();
  }),

  recordTrade: protectedProcedure
    .input(z.object({
      asset: z.string(),
      side: z.enum(["long", "short"]),
      entryPrice: z.number(),
      quantity: z.number(),
      leverage: z.number(),
      mode: z.enum(["swing", "scalp"]).default("swing"),
      confidence: z.number().optional(),
      sentimentScore: z.number().optional(),
      entryReason: z.string().optional(),
    }))
    .mutation(async ({ input }) => {
      const result = await insertTrade(input);

      // Notify owner for high-conviction trades
      if (input.confidence && input.confidence > 0.8) {
        await notifyOwner({
          title: `High-Conviction ${input.side.toUpperCase()} on ${input.asset}`,
          content: `Agent opened ${input.side} on ${input.asset} at $${input.entryPrice} with ${input.leverage}x leverage. Confidence: ${(input.confidence * 100).toFixed(1)}%`,
        });
      }

      return { success: true };
    }),
});

// ─── Portfolio router ────────────────────────────────────────────────────────

const portfolioRouter = router({
  current: publicProcedure.query(async () => {
    try {
      return await engineFetch("/portfolio");
    } catch {
      return { cash: 10, equity: 10, positions: [], peak: 10, drawdown: 0 };
    }
  }),

  snapshots: publicProcedure
    .input(z.object({ limit: z.number().default(500) }).optional())
    .query(async ({ input }) => {
      return getRecentSnapshots(input?.limit ?? 500);
    }),

  recordSnapshot: protectedProcedure
    .input(z.object({
      totalValue: z.number(),
      cash: z.number(),
      unrealizedPnl: z.number(),
      realizedPnl: z.number(),
      drawdown: z.number(),
      sharpeRatio: z.number().optional(),
      winRate: z.number().optional(),
      totalTrades: z.number(),
      openPositions: z.number(),
    }))
    .mutation(async ({ input }) => {
      await insertSnapshot(input);
      return { success: true };
    }),
});

// ─── Training router ─────────────────────────────────────────────────────────

const trainingRouter = router({
  runs: publicProcedure.query(async () => {
    return getTrainingRuns();
  }),

  start: protectedProcedure
    .input(z.object({
      n_episodes: z.number().default(50),
      mode: z.enum(["swing", "scalp"]).default("swing"),
      initial_cash: z.number().default(10),
      max_leverage: z.number().default(30),
      data_start: z.string().default("2018-01-01"),
      lr: z.number().default(0.0003),
    }))
    .mutation(async ({ input }) => {
      // Record training run in DB
      await insertTrainingRun({
        runName: `${input.mode}_${new Date().toISOString().slice(0, 10)}`,
        mode: "offline",
        totalEpisodes: input.n_episodes,
        hyperparameters: input,
      });

      // Start training via engine
      const result = await engineFetch("/train/start", {
        method: "POST",
        body: JSON.stringify(input),
      });

      return result;
    }),

  status: publicProcedure.query(async () => {
    try {
      return await engineFetch("/train/status");
    } catch {
      return { active: false, current: null, history_count: 0 };
    }
  }),

  metrics: publicProcedure
    .input(z.object({ last_n: z.number().default(100) }).optional())
    .query(async ({ input }) => {
      try {
        return await engineFetch(`/train/metrics?last_n=${input?.last_n ?? 100}`);
      } catch {
        return { total: 0, metrics: [] };
      }
    }),
});

// ─── Engine router ───────────────────────────────────────────────────────────

const engineRouter = router({
  health: publicProcedure.query(async () => {
    try {
      return await engineFetch("/health");
    } catch {
      return { status: "offline", agent_loaded: false, data_loaded: false };
    }
  }),

  loadData: protectedProcedure
    .input(z.object({ start: z.string().default("2018-01-01") }))
    .mutation(async ({ input }) => {
      return engineFetch("/data/load?start=" + input.start, { method: "POST" });
    }),

  dataSummary: publicProcedure.query(async () => {
    try {
      return await engineFetch("/data/summary");
    } catch {
      return { status: "no_data" };
    }
  }),

  predict: protectedProcedure
    .input(z.object({ state: z.array(z.number()) }))
    .mutation(async ({ input }) => {
      return engineFetch("/predict", {
        method: "POST",
        body: JSON.stringify(input),
      });
    }),

  evaluate: protectedProcedure
    .input(z.object({ asset: z.string().default("BTCUSDT") }))
    .mutation(async ({ input }) => {
      return engineFetch(`/evaluate/${input.asset}`);
    }),
});

// ─── Sentiment router (LLM-powered) ─────────────────────────────────────────

const sentimentRouter = router({
  analyze: protectedProcedure
    .input(z.object({
      headlines: z.array(z.string()),
      asset: z.string().default("BTC"),
    }))
    .mutation(async ({ input }) => {
      const response = await invokeLLM({
        messages: [
          {
            role: "system",
            content: `You are a financial sentiment analysis expert. Analyze the following news headlines about ${input.asset} and return a JSON object with: sentiment_score (-1.0 to 1.0), confidence (0-1), key_factors (array of strings), market_impact (bullish/bearish/neutral), volatility_expectation (low/medium/high).`,
          },
          {
            role: "user",
            content: `Headlines:\n${input.headlines.map((h, i) => `${i + 1}. ${h}`).join("\n")}`,
          },
        ],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "sentiment_analysis",
            strict: true,
            schema: {
              type: "object",
              properties: {
                sentiment_score: { type: "number", description: "Overall sentiment from -1 (bearish) to 1 (bullish)" },
                confidence: { type: "number", description: "Confidence in the analysis from 0 to 1" },
                key_factors: { type: "array", items: { type: "string" }, description: "Key factors driving sentiment" },
                market_impact: { type: "string", description: "Expected market impact" },
                volatility_expectation: { type: "string", description: "Expected volatility level" },
              },
              required: ["sentiment_score", "confidence", "key_factors", "market_impact", "volatility_expectation"],
              additionalProperties: false,
            },
          },
        },
      });

      const rawContent = response.choices?.[0]?.message?.content;
      const content = typeof rawContent === "string" ? rawContent : JSON.stringify(rawContent);
      try {
        return JSON.parse(content || "{}");
      } catch {
        return { sentiment_score: 0, confidence: 0, key_factors: [], market_impact: "neutral", volatility_expectation: "medium" };
      }
    }),
});

// ─── Model management router ─────────────────────────────────────────────────

const modelRouter = router({
  versions: publicProcedure.query(async () => {
    return getModelVersions();
  }),

  active: publicProcedure.query(async () => {
    return getActiveModel();
  }),

  backupToS3: protectedProcedure.mutation(async () => {
    try {
      const modelData = await engineFetch("/model/bytes");
      if (!modelData.data_b64) throw new Error("No model data");

      const buffer = Buffer.from(modelData.data_b64, "base64");
      const key = `rl-models/model_${Date.now()}.pt`;
      const { url } = await storagePut(key, buffer, "application/octet-stream");

      await insertModelVersion({
        version: `v_${Date.now()}`,
        isActive: true,
        s3Path: url,
        metrics: { size: buffer.length, backed_up_at: new Date().toISOString() },
      });

      await notifyOwner({
        title: "Model Backed Up to S3",
        content: `Model weights (${(buffer.length / 1024).toFixed(1)} KB) saved to S3: ${key}`,
      });

      return { success: true, url, size: buffer.length };
    } catch (e: any) {
      throw new Error(`S3 backup failed: ${e.message}`);
    }
  }),
});

// ─── Live Trading router ────────────────────────────────────────────────────

const liveRouter = router({
  status: publicProcedure.query(async () => {
    try {
      return await engineFetch("/live/status");
    } catch {
      return {
        running: false, symbol: "BTCUSDT", leverage: 7, testnet: true,
        dry_run: false, balance: 0, initial_balance: 0, roi_pct: 0,
        current_position: 0, entry_price: 0, total_trades: 0,
        winning_trades: 0, win_rate: 0, trade_history: [],
      };
    }
  }),

  start: protectedProcedure
    .input(z.object({
      leverage: z.number().default(7),
      testnet: z.boolean().default(true),
      dry_run: z.boolean().default(false),
    }))
    .mutation(async ({ input }) => {
      const result = await engineFetch("/live/start", {
        method: "POST",
        body: JSON.stringify(input),
      });

      await notifyOwner({
        title: "Live Trading Started",
        content: `RL Agent started trading ${input.testnet ? "(TESTNET)" : "(LIVE)"} with ${input.leverage}x leverage`,
      });

      return result;
    }),

  stop: protectedProcedure.mutation(async () => {
    const result = await engineFetch("/live/stop", { method: "POST" });

    await notifyOwner({
      title: "Live Trading Stopped",
      content: "RL Agent has been stopped.",
    });

    return result;
  }),

  tradeHistory: publicProcedure
    .input(z.object({ limit: z.number().default(50) }).optional())
    .query(async ({ input }) => {
      try {
        return await engineFetch(`/live/trades?limit=${input?.limit ?? 50}`);
      } catch {
        return { trades: [] };
      }
    }),
});

// ─── Telegram router ────────────────────────────────────────────────────────

const telegramRouter = router({
  testMessage: protectedProcedure
    .input(z.object({ message: z.string().default("Test signal from RL Trading Bot") }))
    .mutation(async ({ input }) => {
      try {
        const result = await engineFetch("/telegram/test", {
          method: "POST",
          body: JSON.stringify({ message: input.message }),
        });
        return result;
      } catch (e: any) {
        return { success: false, error: e.message };
      }
    }),

  config: publicProcedure.query(async () => {
    try {
      return await engineFetch("/telegram/config");
    } catch {
      return { configured: false, chat_id: "", bot_username: "" };
    }
  }),
});

// ─── Results router (serves training report) ────────────────────────────────

const resultsRouter = router({
  report: publicProcedure.query(async () => {
    try {
      const reportPath = path.join(process.cwd(), "engine", "results", "pipeline_report.json");
      if (fs.existsSync(reportPath)) {
        const data = JSON.parse(fs.readFileSync(reportPath, "utf-8"));
        return { available: true, ...data };
      }
      return { available: false };
    } catch {
      return { available: false };
    }
  }),

  charts: publicProcedure.query(async () => {
    return {
      training_progress: "https://d2xsxph8kpxj0f.cloudfront.net/310519663449982452/USQVpVuYrG9ewP9aE57FZB/01_training_progress_de994c55.png",
      test_evaluation: "https://d2xsxph8kpxj0f.cloudfront.net/310519663449982452/USQVpVuYrG9ewP9aE57FZB/02_test_evaluation_9eaa8251.png",
      online_vs_baseline: "https://d2xsxph8kpxj0f.cloudfront.net/310519663449982452/USQVpVuYrG9ewP9aE57FZB/03_online_vs_baseline_d168e64a.png",
      trade_analysis: "https://d2xsxph8kpxj0f.cloudfront.net/310519663449982452/USQVpVuYrG9ewP9aE57FZB/04_trade_analysis_cf2dc8dc.png",
      summary: "https://d2xsxph8kpxj0f.cloudfront.net/310519663449982452/USQVpVuYrG9ewP9aE57FZB/05_summary_4fa96924.png",
    };
  }),
});

// ─── Logs router ─────────────────────────────────────────────────────────────

const logsRouter = router({
  recent: publicProcedure
    .input(z.object({ limit: z.number().default(100) }).optional())
    .query(async ({ input }) => {
      return getAgentLogs(input?.limit ?? 100);
    }),

  add: protectedProcedure
    .input(z.object({
      level: z.enum(["info", "warn", "error", "trade", "risk"]),
      message: z.string(),
      metadata: z.any().optional(),
    }))
    .mutation(async ({ input }) => {
      await insertAgentLog(input);

      // Notify on risk alerts
      if (input.level === "risk") {
        await notifyOwner({
          title: "Risk Alert",
          content: input.message,
        });
      }

      return { success: true };
    }),
});

// ─── Main router ─────────────────────────────────────────────────────────────

export const appRouter = router({
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return { success: true } as const;
    }),
  }),
  trading: tradingRouter,
  portfolio: portfolioRouter,
  training: trainingRouter,
  engine: engineRouter,
  sentiment: sentimentRouter,
  model: modelRouter,
  logs: logsRouter,
  results: resultsRouter,
  live: liveRouter,
  telegram: telegramRouter,
});

export type AppRouter = typeof appRouter;
