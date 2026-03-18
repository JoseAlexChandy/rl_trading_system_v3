import { describe, expect, it, vi, beforeEach } from "vitest";
import { appRouter } from "./routers";
import { COOKIE_NAME } from "../shared/const";
import type { TrpcContext } from "./_core/context";

// ─── Test helpers ────────────────────────────────────────────────────────────

type CookieCall = { name: string; options: Record<string, unknown> };
type AuthenticatedUser = NonNullable<TrpcContext["user"]>;

function createAuthContext(): { ctx: TrpcContext; clearedCookies: CookieCall[] } {
  const clearedCookies: CookieCall[] = [];
  const user: AuthenticatedUser = {
    id: 1, openId: "test-user", email: "test@example.com",
    name: "Test User", loginMethod: "manus", role: "admin",
    createdAt: new Date(), updatedAt: new Date(), lastSignedIn: new Date(),
  };
  const ctx: TrpcContext = {
    user,
    req: { protocol: "https", headers: {} } as TrpcContext["req"],
    res: {
      clearCookie: (name: string, options: Record<string, unknown>) => {
        clearedCookies.push({ name, options });
      },
    } as TrpcContext["res"],
  };
  return { ctx, clearedCookies };
}

function createPublicContext(): TrpcContext {
  return {
    user: null,
    req: { protocol: "https", headers: {} } as TrpcContext["req"],
    res: {
      clearCookie: () => {},
    } as TrpcContext["res"],
  };
}

// ─── Auth tests ──────────────────────────────────────────────────────────────

describe("auth.me", () => {
  it("returns null for unauthenticated users", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.auth.me();
    expect(result).toBeNull();
  });

  it("returns user for authenticated users", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.auth.me();
    expect(result).toBeDefined();
    expect(result?.openId).toBe("test-user");
    expect(result?.name).toBe("Test User");
    expect(result?.role).toBe("admin");
  });
});

describe("auth.logout", () => {
  it("clears the session cookie and reports success", async () => {
    const { ctx, clearedCookies } = createAuthContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.auth.logout();
    expect(result).toEqual({ success: true });
    expect(clearedCookies).toHaveLength(1);
    expect(clearedCookies[0]?.name).toBe(COOKIE_NAME);
    expect(clearedCookies[0]?.options).toMatchObject({ maxAge: -1 });
  });
});

// ─── Trading router tests ────────────────────────────────────────────────────

describe("trading.recentTrades", () => {
  it("returns an array (possibly empty)", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.trading.recentTrades({ limit: 10 });
    expect(Array.isArray(result)).toBe(true);
  });
});

describe("trading.stats", () => {
  it("returns stats object or null", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.trading.stats();
    // Stats can be null if no trades exist
    expect(result === null || typeof result === "object").toBe(true);
  });
});

// ─── Portfolio router tests ──────────────────────────────────────────────────

describe("portfolio.current", () => {
  it("returns portfolio with default values when engine is offline", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.portfolio.current();
    expect(result).toBeDefined();
    expect(typeof result.cash).toBe("number");
    expect(typeof result.equity).toBe("number");
  });
});

describe("portfolio.snapshots", () => {
  it("returns an array", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.portfolio.snapshots({ limit: 10 });
    expect(Array.isArray(result)).toBe(true);
  });
});

// ─── Training router tests ──────────────────────────────────────────────────

describe("training.status", () => {
  it("returns status object even when engine is offline", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.training.status();
    expect(result).toBeDefined();
    expect(typeof result.active).toBe("boolean");
  });
});

describe("training.metrics", () => {
  it("returns metrics structure even when engine is offline", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.training.metrics({ last_n: 10 });
    expect(result).toBeDefined();
    expect(typeof result.total).toBe("number");
    expect(Array.isArray(result.metrics)).toBe(true);
  });
});

// ─── Engine router tests ─────────────────────────────────────────────────────

describe("engine.health", () => {
  it("returns health status (offline when engine not running)", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.engine.health();
    expect(result).toBeDefined();
    // Engine is offline in test env
    expect(result.status === "ok" || result.status === "offline").toBe(true);
  });
});

describe("engine.dataSummary", () => {
  it("returns data summary or no_data", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.engine.dataSummary();
    expect(result).toBeDefined();
    expect(result.status === "ok" || result.status === "no_data").toBe(true);
  });
});

// ─── Model router tests ─────────────────────────────────────────────────────

describe("model.versions", () => {
  it("returns an array", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.model.versions();
    expect(Array.isArray(result)).toBe(true);
  });
});

describe("model.active", () => {
  it("returns null when no active model", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.model.active();
    expect(result === null || typeof result === "object").toBe(true);
  });
});

// ─── Logs router tests ──────────────────────────────────────────────────────

describe("logs.recent", () => {
  it("returns an array", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.logs.recent({ limit: 10 });
    expect(Array.isArray(result)).toBe(true);
  });
});

// ─── Results router tests ───────────────────────────────────────────────────

describe("results.report", () => {
  it("returns report data or available:false", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.results.report();
    expect(result).toBeDefined();
    expect(typeof result.available).toBe("boolean");
    if (result.available) {
      expect((result as any).evaluation).toBeDefined();
    }
  });
});

describe("results.charts", () => {
  it("returns chart URLs", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.results.charts();
    expect(result).toBeDefined();
    expect(typeof result.summary).toBe("string");
    expect(typeof result.training_progress).toBe("string");
    expect(typeof result.test_evaluation).toBe("string");
    expect(typeof result.online_vs_baseline).toBe("string");
    expect(typeof result.trade_analysis).toBe("string");
  });
});

// ─── Live Trading router tests ──────────────────────────────────────────────

describe("live.status", () => {
  it("returns default status when engine is offline", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.live.status();
    expect(result).toBeDefined();
    expect(typeof result.running).toBe("boolean");
    expect(result.running).toBe(false);
    expect(typeof result.leverage).toBe("number");
    expect(typeof result.testnet).toBe("boolean");
    expect(typeof result.total_trades).toBe("number");
    expect(typeof result.win_rate).toBe("number");
  });
});

describe("live.tradeHistory", () => {
  it("returns empty trades when engine is offline", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.live.tradeHistory({ limit: 10 });
    expect(result).toBeDefined();
    expect(result.trades).toBeDefined();
    expect(Array.isArray(result.trades)).toBe(true);
  });
});

// ─── Telegram router tests ──────────────────────────────────────────────────

describe("telegram.config", () => {
  it("returns config status when engine is offline", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.telegram.config();
    expect(result).toBeDefined();
    expect(typeof result.configured).toBe("boolean");
    expect(result.configured).toBe(false);
  });
});

describe("telegram.testMessage", () => {
  it("returns error when engine is offline", async () => {
    const { ctx } = createAuthContext();
    const caller = appRouter.createCaller(ctx);
    const result = await caller.telegram.testMessage({ message: "test" });
    expect(result).toBeDefined();
    expect(result.success).toBe(false);
    expect(typeof result.error).toBe("string");
  });
});

// ─── Router structure tests ──────────────────────────────────────────────────

describe("appRouter structure", () => {
  it("has all expected sub-routers", () => {
    const procedures = Object.keys(appRouter._def.procedures);
    expect(procedures).toContain("auth.me");
    expect(procedures).toContain("auth.logout");
    expect(procedures).toContain("trading.recentTrades");
    expect(procedures).toContain("trading.stats");
    expect(procedures).toContain("portfolio.current");
    expect(procedures).toContain("training.status");
    expect(procedures).toContain("engine.health");
    expect(procedures).toContain("sentiment.analyze");
    expect(procedures).toContain("model.versions");
    expect(procedures).toContain("logs.recent");
    expect(procedures).toContain("results.report");
    expect(procedures).toContain("results.charts");
    expect(procedures).toContain("live.status");
    expect(procedures).toContain("live.start");
    expect(procedures).toContain("live.stop");
    expect(procedures).toContain("live.tradeHistory");
    expect(procedures).toContain("telegram.config");
    expect(procedures).toContain("telegram.testMessage");
  });
});
