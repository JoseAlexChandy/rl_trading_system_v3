import { eq, desc, sql, and, gte, lte } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import {
  InsertUser, users,
  trades, InsertTrade, Trade,
  portfolioSnapshots, InsertPortfolioSnapshot,
  trainingRuns, InsertTrainingRun,
  modelVersions, InsertModelVersion,
  agentLogs, InsertAgentLog,
} from "../drizzle/schema";
import { ENV } from './_core/env';

let _db: ReturnType<typeof drizzle> | null = null;

export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

// ─── User helpers ────────────────────────────────────────────────────────────

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) throw new Error("User openId is required for upsert");
  const db = await getDb();
  if (!db) { console.warn("[Database] Cannot upsert user: database not available"); return; }
  try {
    const values: InsertUser = { openId: user.openId };
    const updateSet: Record<string, unknown> = {};
    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];
    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };
    textFields.forEach(assignNullable);
    if (user.lastSignedIn !== undefined) { values.lastSignedIn = user.lastSignedIn; updateSet.lastSignedIn = user.lastSignedIn; }
    if (user.role !== undefined) { values.role = user.role; updateSet.role = user.role; }
    else if (user.openId === ENV.ownerOpenId) { values.role = 'admin'; updateSet.role = 'admin'; }
    if (!values.lastSignedIn) values.lastSignedIn = new Date();
    if (Object.keys(updateSet).length === 0) updateSet.lastSignedIn = new Date();
    await db.insert(users).values(values).onDuplicateKeyUpdate({ set: updateSet });
  } catch (error) { console.error("[Database] Failed to upsert user:", error); throw error; }
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) { console.warn("[Database] Cannot get user: database not available"); return undefined; }
  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);
  return result.length > 0 ? result[0] : undefined;
}

// ─── Trade helpers ───────────────────────────────────────────────────────────

export async function insertTrade(trade: InsertTrade) {
  const db = await getDb();
  if (!db) return null;
  const result = await db.insert(trades).values(trade);
  return result;
}

export async function updateTrade(id: number, updates: Partial<InsertTrade>) {
  const db = await getDb();
  if (!db) return null;
  return db.update(trades).set(updates).where(eq(trades.id, id));
}

export async function getRecentTrades(limit: number = 50) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(trades).orderBy(desc(trades.openedAt)).limit(limit);
}

export async function getOpenTrades() {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(trades).where(eq(trades.status, "open"));
}

export async function getTradeStats() {
  const db = await getDb();
  if (!db) return null;
  const result = await db.select({
    totalTrades: sql<number>`COUNT(*)`,
    winningTrades: sql<number>`SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)`,
    totalPnl: sql<number>`SUM(COALESCE(pnl, 0))`,
    avgPnlPct: sql<number>`AVG(COALESCE(pnlPercent, 0))`,
    bestTrade: sql<number>`MAX(COALESCE(pnlPercent, 0))`,
    worstTrade: sql<number>`MIN(COALESCE(pnlPercent, 0))`,
  }).from(trades).where(eq(trades.status, "closed"));
  return result[0] || null;
}

// ─── Portfolio snapshot helpers ──────────────────────────────────────────────

export async function insertSnapshot(snap: InsertPortfolioSnapshot) {
  const db = await getDb();
  if (!db) return null;
  return db.insert(portfolioSnapshots).values(snap);
}

export async function getRecentSnapshots(limit: number = 500) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(portfolioSnapshots).orderBy(desc(portfolioSnapshots.snapshotAt)).limit(limit);
}

// ─── Training run helpers ────────────────────────────────────────────────────

export async function insertTrainingRun(run: InsertTrainingRun) {
  const db = await getDb();
  if (!db) return null;
  return db.insert(trainingRuns).values(run);
}

export async function updateTrainingRun(id: number, updates: Partial<InsertTrainingRun>) {
  const db = await getDb();
  if (!db) return null;
  return db.update(trainingRuns).set(updates).where(eq(trainingRuns.id, id));
}

export async function getTrainingRuns(limit: number = 20) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(trainingRuns).orderBy(desc(trainingRuns.startedAt)).limit(limit);
}

// ─── Model version helpers ───────────────────────────────────────────────────

export async function insertModelVersion(mv: InsertModelVersion) {
  const db = await getDb();
  if (!db) return null;
  return db.insert(modelVersions).values(mv);
}

export async function getActiveModel() {
  const db = await getDb();
  if (!db) return null;
  const result = await db.select().from(modelVersions).where(eq(modelVersions.isActive, true)).limit(1);
  return result[0] || null;
}

export async function getModelVersions(limit: number = 20) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(modelVersions).orderBy(desc(modelVersions.deployedAt)).limit(limit);
}

// ─── Agent log helpers ───────────────────────────────────────────────────────

export async function insertAgentLog(log: InsertAgentLog) {
  const db = await getDb();
  if (!db) return null;
  return db.insert(agentLogs).values(log);
}

export async function getAgentLogs(limit: number = 100) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(agentLogs).orderBy(desc(agentLogs.createdAt)).limit(limit);
}
