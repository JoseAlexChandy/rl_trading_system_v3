import { int, mysqlEnum, mysqlTable, text, timestamp, varchar, float, bigint, boolean, json } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 */
export const users = mysqlTable("users", {
  id: int("id").autoincrement().primaryKey(),
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * Trades executed by the RL agent.
 */
export const trades = mysqlTable("trades", {
  id: int("id").autoincrement().primaryKey(),
  asset: varchar("asset", { length: 32 }).notNull(),
  side: mysqlEnum("side", ["long", "short"]).notNull(),
  entryPrice: float("entryPrice").notNull(),
  exitPrice: float("exitPrice"),
  quantity: float("quantity").notNull(),
  leverage: float("leverage").notNull(),
  pnl: float("pnl"),
  pnlPercent: float("pnlPercent"),
  status: mysqlEnum("status", ["open", "closed", "liquidated"]).default("open").notNull(),
  mode: mysqlEnum("mode", ["swing", "scalp"]).default("swing").notNull(),
  confidence: float("confidence"),
  sentimentScore: float("sentimentScore"),
  entryReason: text("entryReason"),
  exitReason: text("exitReason"),
  openedAt: timestamp("openedAt").defaultNow().notNull(),
  closedAt: timestamp("closedAt"),
});

export type Trade = typeof trades.$inferSelect;
export type InsertTrade = typeof trades.$inferInsert;

/**
 * Portfolio snapshots taken periodically for equity curve.
 */
export const portfolioSnapshots = mysqlTable("portfolio_snapshots", {
  id: int("id").autoincrement().primaryKey(),
  totalValue: float("totalValue").notNull(),
  cash: float("cash").notNull(),
  unrealizedPnl: float("unrealizedPnl").notNull(),
  realizedPnl: float("realizedPnl").notNull(),
  drawdown: float("drawdown").notNull(),
  sharpeRatio: float("sharpeRatio"),
  winRate: float("winRate"),
  totalTrades: int("totalTrades").notNull(),
  openPositions: int("openPositions").notNull(),
  snapshotAt: timestamp("snapshotAt").defaultNow().notNull(),
});

export type PortfolioSnapshot = typeof portfolioSnapshots.$inferSelect;
export type InsertPortfolioSnapshot = typeof portfolioSnapshots.$inferInsert;

/**
 * Training runs tracking model performance over time.
 */
export const trainingRuns = mysqlTable("training_runs", {
  id: int("id").autoincrement().primaryKey(),
  runName: varchar("runName", { length: 128 }).notNull(),
  status: mysqlEnum("status", ["running", "completed", "failed", "paused"]).default("running").notNull(),
  mode: mysqlEnum("mode", ["offline", "online"]).default("offline").notNull(),
  totalEpisodes: int("totalEpisodes").notNull(),
  completedEpisodes: int("completedEpisodes").default(0).notNull(),
  bestReward: float("bestReward"),
  avgReward: float("avgReward"),
  finalPortfolioValue: float("finalPortfolioValue"),
  hyperparameters: json("hyperparameters"),
  s3ModelPath: text("s3ModelPath"),
  s3BufferPath: text("s3BufferPath"),
  startedAt: timestamp("startedAt").defaultNow().notNull(),
  completedAt: timestamp("completedAt"),
});

export type TrainingRun = typeof trainingRuns.$inferSelect;
export type InsertTrainingRun = typeof trainingRuns.$inferInsert;

/**
 * Model versions for tracking deployed models.
 */
export const modelVersions = mysqlTable("model_versions", {
  id: int("id").autoincrement().primaryKey(),
  version: varchar("version", { length: 64 }).notNull(),
  trainingRunId: int("trainingRunId"),
  isActive: boolean("isActive").default(false).notNull(),
  s3Path: text("s3Path").notNull(),
  metrics: json("metrics"),
  deployedAt: timestamp("deployedAt").defaultNow().notNull(),
});

export type ModelVersion = typeof modelVersions.$inferSelect;
export type InsertModelVersion = typeof modelVersions.$inferInsert;

/**
 * Agent activity log for audit trail.
 */
export const agentLogs = mysqlTable("agent_logs", {
  id: int("id").autoincrement().primaryKey(),
  level: mysqlEnum("level", ["info", "warn", "error", "trade", "risk"]).default("info").notNull(),
  message: text("message").notNull(),
  metadata: json("metadata"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type AgentLog = typeof agentLogs.$inferSelect;
export type InsertAgentLog = typeof agentLogs.$inferInsert;
