import { useAuth } from "@/_core/hooks/useAuth";
import { getLoginUrl } from "@/const";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import {
  Activity, TrendingUp, TrendingDown, DollarSign, BarChart3,
  Brain, Zap, Shield, RefreshCw, Play, Upload, Square,
  ArrowUpRight, ArrowDownRight, Minus, Clock, Target,
  Loader2, LogIn, AlertTriangle, FileBarChart, Image,
  Send, MessageCircle, Radio, Wifi, WifiOff,
} from "lucide-react";
import { useState, useMemo } from "react";
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area,
} from "recharts";

// ─── Metric Card ─────────────────────────────────────────────────────────────

function MetricCard({ title, value, subtitle, icon: Icon, trend }: {
  title: string; value: string; subtitle?: string;
  icon: any; trend?: "up" | "down" | "neutral";
}) {
  const trendColor = trend === "up" ? "text-profit" : trend === "down" ? "text-loss" : "text-muted-foreground";
  const TrendIcon = trend === "up" ? ArrowUpRight : trend === "down" ? ArrowDownRight : Minus;
  return (
    <Card className="bg-card border-border">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-muted-foreground uppercase tracking-wider">{title}</span>
          <Icon className="h-4 w-4 text-muted-foreground" />
        </div>
        <div className="flex items-end gap-2">
          <span className="text-2xl font-bold font-mono">{value}</span>
          {trend && <TrendIcon className={`h-4 w-4 ${trendColor}`} />}
        </div>
        {subtitle && <p className={`text-xs mt-1 ${trendColor}`}>{subtitle}</p>}
      </CardContent>
    </Card>
  );
}

// ─── Engine Status Badge ─────────────────────────────────────────────────────

function EngineStatus() {
  const { data: health } = trpc.engine.health.useQuery(undefined, { refetchInterval: 5000 });
  const isOnline = health?.status === "ok";
  return (
    <div className="flex items-center gap-2">
      <div className={`w-2 h-2 rounded-full ${isOnline ? "bg-green-500 animate-pulse" : "bg-red-500"}`} />
      <span className="text-xs text-muted-foreground">
        Engine {isOnline ? "Online" : "Offline"}
      </span>
      {health?.agent_loaded && (
        <Badge variant="outline" className="text-xs border-primary/30 text-primary">
          <Brain className="h-3 w-3 mr-1" /> Agent Loaded
        </Badge>
      )}
    </div>
  );
}

// ─── Equity Curve Chart ──────────────────────────────────────────────────────

function EquityCurve() {
  const { data: metrics } = trpc.training.metrics.useQuery({ last_n: 200 }, { refetchInterval: 3000 });
  const chartData = useMemo(() => {
    if (!metrics?.metrics) return [];
    return metrics.metrics.map((m: any, i: number) => ({
      ep: m.episode || i + 1,
      equity: m.equity || 10,
      reward: m.reward || 0,
    }));
  }, [metrics]);

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        <div className="text-center">
          <BarChart3 className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No training data yet. Start training to see the equity curve.</p>
        </div>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={chartData}>
        <defs>
          <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="oklch(0.72 0.19 145)" stopOpacity={0.3} />
            <stop offset="95%" stopColor="oklch(0.72 0.19 145)" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0.01 260)" />
        <XAxis dataKey="ep" stroke="oklch(0.50 0.01 260)" fontSize={11} />
        <YAxis stroke="oklch(0.50 0.01 260)" fontSize={11} tickFormatter={(v) => `$${v.toFixed(2)}`} />
        <Tooltip
          contentStyle={{ background: "oklch(0.16 0.01 260)", border: "1px solid oklch(0.25 0.01 260)", borderRadius: "6px", color: "oklch(0.93 0.01 260)" }}
          formatter={(v: number) => [`$${v.toFixed(4)}`, "Equity"]}
        />
        <Area type="monotone" dataKey="equity" stroke="oklch(0.72 0.19 145)" fill="url(#equityGrad)" strokeWidth={2} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ─── Trade History Table ─────────────────────────────────────────────────────

function TradeHistory() {
  const { data: trades } = trpc.trading.recentTrades.useQuery({ limit: 20 }, { refetchInterval: 5000 });

  if (!trades || trades.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground">
        <div className="text-center">
          <Clock className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No trades recorded yet.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-muted-foreground">
            <th className="text-left py-2 px-3 font-medium">Asset</th>
            <th className="text-left py-2 px-3 font-medium">Side</th>
            <th className="text-right py-2 px-3 font-medium">Entry</th>
            <th className="text-right py-2 px-3 font-medium">Exit</th>
            <th className="text-right py-2 px-3 font-medium">Lev</th>
            <th className="text-right py-2 px-3 font-medium">P&L</th>
            <th className="text-right py-2 px-3 font-medium">P&L %</th>
            <th className="text-center py-2 px-3 font-medium">Status</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t: any) => (
            <tr key={t.id} className="border-b border-border/50 hover:bg-accent/30 transition-colors">
              <td className="py-2 px-3 font-mono font-medium">{t.asset}</td>
              <td className="py-2 px-3">
                <Badge variant="outline" className={t.side === "long" ? "border-green-500/30 text-green-400" : "border-red-500/30 text-red-400"}>
                  {t.side === "long" ? <TrendingUp className="h-3 w-3 mr-1" /> : <TrendingDown className="h-3 w-3 mr-1" />}
                  {t.side}
                </Badge>
              </td>
              <td className="py-2 px-3 text-right font-mono">${t.entryPrice?.toFixed(2)}</td>
              <td className="py-2 px-3 text-right font-mono">{t.exitPrice ? `$${t.exitPrice.toFixed(2)}` : "—"}</td>
              <td className="py-2 px-3 text-right font-mono">{t.leverage}x</td>
              <td className={`py-2 px-3 text-right font-mono font-medium ${(t.pnl ?? 0) >= 0 ? "text-profit" : "text-loss"}`}>
                {t.pnl != null ? `${t.pnl >= 0 ? "+" : ""}$${t.pnl.toFixed(2)}` : "—"}
              </td>
              <td className={`py-2 px-3 text-right font-mono ${(t.pnlPercent ?? 0) >= 0 ? "text-profit" : "text-loss"}`}>
                {t.pnlPercent != null ? `${t.pnlPercent >= 0 ? "+" : ""}${t.pnlPercent.toFixed(1)}%` : "—"}
              </td>
              <td className="py-2 px-3 text-center">
                <Badge variant={t.status === "open" ? "default" : t.status === "closed" ? "secondary" : "destructive"} className="text-xs">
                  {t.status}
                </Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Live Trading Panel ─────────────────────────────────────────────────────

function LiveTradingPanel() {
  const { isAuthenticated } = useAuth();
  const { data: liveStatus, refetch: refetchLive } = trpc.live.status.useQuery(undefined, { refetchInterval: 3000 });
  const { data: health } = trpc.engine.health.useQuery(undefined, { refetchInterval: 5000 });
  const [leverage, setLeverage] = useState(7);
  const [testnet, setTestnet] = useState(true);
  const [dryRun, setDryRun] = useState(false);

  const startLive = trpc.live.start.useMutation({
    onSuccess: () => { toast.success("Live trading started!"); refetchLive(); },
    onError: (e) => toast.error(e.message),
  });
  const stopLive = trpc.live.stop.useMutation({
    onSuccess: () => { toast.success("Live trading stopped."); refetchLive(); },
    onError: (e) => toast.error(e.message),
  });

  const isRunning = liveStatus?.running;
  const isEngineOnline = health?.status === "ok";

  return (
    <div className="space-y-4">
      {/* Status indicator */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isRunning ? (
            <><Radio className="h-4 w-4 text-green-400 animate-pulse" /><span className="text-sm text-green-400 font-medium">LIVE</span></>
          ) : (
            <><WifiOff className="h-4 w-4 text-muted-foreground" /><span className="text-sm text-muted-foreground">Stopped</span></>
          )}
        </div>
        {isRunning && (
          <Badge variant="outline" className="text-xs">
            {liveStatus?.testnet ? "TESTNET" : "MAINNET"} | {liveStatus?.leverage}x
          </Badge>
        )}
      </div>

      {/* Live stats when running */}
      {isRunning && liveStatus && (
        <div className="bg-secondary/50 rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Balance</span>
            <span className="font-mono text-sm">${(liveStatus.balance || 0).toFixed(2)}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">ROI</span>
            <span className={`font-mono text-sm font-bold ${(liveStatus.roi_pct || 0) >= 0 ? "text-profit" : "text-loss"}`}>
              {(liveStatus.roi_pct || 0) >= 0 ? "+" : ""}{(liveStatus.roi_pct || 0).toFixed(1)}%
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Trades</span>
            <span className="font-mono text-sm">{liveStatus.total_trades || 0}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Win Rate</span>
            <span className="font-mono text-sm">{(liveStatus.win_rate || 0).toFixed(1)}%</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Position</span>
            <span className="font-mono text-sm">
              {liveStatus.current_position > 0 ? "LONG" : liveStatus.current_position < 0 ? "SHORT" : "FLAT"}
              {liveStatus.entry_price > 0 && ` @ $${liveStatus.entry_price.toFixed(2)}`}
            </span>
          </div>
        </div>
      )}

      {/* Configuration (when not running) */}
      {!isRunning && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Leverage</span>
            <select
              value={leverage}
              onChange={(e) => setLeverage(Number(e.target.value))}
              className="bg-secondary border border-border rounded px-2 py-1 text-sm font-mono"
            >
              {[3, 5, 7, 10, 15, 20, 25, 30].map(l => (
                <option key={l} value={l}>{l}x</option>
              ))}
            </select>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm text-muted-foreground">Testnet Mode</span>
              <p className="text-xs text-muted-foreground/70">Use Binance Testnet (no real money)</p>
            </div>
            <Switch checked={testnet} onCheckedChange={setTestnet} />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm text-muted-foreground">Dry Run</span>
              <p className="text-xs text-muted-foreground/70">Simulate trades without placing orders</p>
            </div>
            <Switch checked={dryRun} onCheckedChange={setDryRun} />
          </div>
        </div>
      )}

      {/* Start/Stop buttons */}
      <div className="flex gap-2">
        {!isRunning ? (
          <Button
            onClick={() => startLive.mutate({ leverage, testnet, dry_run: dryRun })}
            disabled={!isEngineOnline || startLive.isPending}
            className="w-full bg-green-600 hover:bg-green-700 text-white"
          >
            {startLive.isPending ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
            Start Live Trading
          </Button>
        ) : (
          <Button
            onClick={() => stopLive.mutate()}
            disabled={stopLive.isPending}
            variant="destructive" className="w-full"
          >
            {stopLive.isPending ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Square className="h-4 w-4 mr-2" />}
            Stop Trading
          </Button>
        )}
      </div>

      {!testnet && !isRunning && (
        <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-3 flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
          <p className="text-xs text-destructive">
            MAINNET mode will use real funds. Make sure you've tested on Testnet first.
          </p>
        </div>
      )}

      {!isEngineOnline && (
        <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-3 flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium text-destructive">Engine Offline</p>
            <p className="text-xs text-muted-foreground mt-1">
              Start the engine: <code className="bg-secondary px-1 rounded text-xs">cd engine && python3 api_server.py</code>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Telegram Panel ─────────────────────────────────────────────────────────

function TelegramPanel() {
  const { isAuthenticated } = useAuth();
  const { data: tgConfig } = trpc.telegram.config.useQuery(undefined, { refetchInterval: 10000 });
  const testMessage = trpc.telegram.testMessage.useMutation({
    onSuccess: (data: any) => {
      if (data.success) toast.success("Test message sent to Telegram!");
      else toast.error(data.error || "Failed to send message");
    },
    onError: (e) => toast.error(e.message),
  });

  const isConfigured = tgConfig?.configured;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageCircle className="h-4 w-4 text-blue-400" />
          <span className="text-sm font-medium">Telegram Signals</span>
        </div>
        <Badge variant="outline" className={`text-xs ${isConfigured ? "border-green-500/30 text-green-400" : "border-yellow-500/30 text-yellow-400"}`}>
          {isConfigured ? "Connected" : "Not Configured"}
        </Badge>
      </div>

      {isConfigured ? (
        <>
          <div className="bg-secondary/50 rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Chat ID</span>
              <span className="font-mono text-xs">{tgConfig?.chat_id || "—"}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Bot Token</span>
              <span className="font-mono text-xs text-green-400">Configured</span>
            </div>
          </div>
          <div className="text-xs text-muted-foreground space-y-1">
            <p>The bot will automatically send signals to your group when:</p>
            <ul className="list-disc list-inside space-y-0.5 ml-1">
              <li>Agent opens a new position (LONG/SHORT)</li>
              <li>Agent closes a position (with P&L)</li>
              <li>High-conviction trade detected (&gt;20% size)</li>
              <li>Risk alert triggered (approaching liquidation)</li>
            </ul>
          </div>
          <Button
            onClick={() => testMessage.mutate({ message: "🤖 RL Trading Bot — Test Signal\n\n📊 BTCUSDT | LONG\n💰 Entry: $84,500\n🎯 TP: $87,000 (+2.96%)\n🛑 SL: $83,000 (-1.78%)\n⚡ Leverage: 7x\n📈 Confidence: 78%\n\nThis is a test message." })}
            disabled={!isAuthenticated || testMessage.isPending}
            variant="outline" className="w-full" size="sm"
          >
            {testMessage.isPending ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Send className="h-4 w-4 mr-2" />}
            Send Test Signal
          </Button>
        </>
      ) : (
        <div className="bg-secondary/50 rounded-lg p-3 space-y-2">
          <p className="text-xs text-muted-foreground">
            To enable Telegram signals, configure these environment variables:
          </p>
          <div className="space-y-1">
            <code className="text-xs bg-background px-2 py-1 rounded block">TELEGRAM_BOT_TOKEN</code>
            <code className="text-xs bg-background px-2 py-1 rounded block">TELEGRAM_CHAT_ID</code>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            1. Create a bot via <strong>@BotFather</strong> on Telegram<br/>
            2. Add the bot to your group<br/>
            3. Get the Chat ID via <strong>@userinfobot</strong> or the Telegram API
          </p>
        </div>
      )}
    </div>
  );
}

// ─── Training Control Panel ──────────────────────────────────────────────────

function TrainingPanel() {
  const { data: status } = trpc.training.status.useQuery(undefined, { refetchInterval: 2000 });
  const { data: health } = trpc.engine.health.useQuery(undefined, { refetchInterval: 5000 });
  const startTraining = trpc.training.start.useMutation({
    onSuccess: () => toast.success("Training started!"),
    onError: (e) => toast.error(e.message),
  });
  const loadData = trpc.engine.loadData.useMutation({
    onSuccess: () => toast.success("Data loaded successfully!"),
    onError: (e) => toast.error(e.message),
  });
  const backupModel = trpc.model.backupToS3.useMutation({
    onSuccess: () => toast.success("Model backed up to S3!"),
    onError: (e) => toast.error(e.message),
  });

  const isEngineOnline = health?.status === "ok";
  const isTraining = status?.active;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <Button
          onClick={() => loadData.mutate({ start: "2018-01-01" })}
          disabled={!isEngineOnline || loadData.isPending}
          variant="outline" className="w-full"
        >
          {loadData.isPending ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <RefreshCw className="h-4 w-4 mr-2" />}
          Load Data
        </Button>
        <Button
          onClick={() => startTraining.mutate({
            n_episodes: 50, mode: "swing", initial_cash: 10, max_leverage: 10, data_start: "2018-01-01", lr: 0.0003,
          })}
          disabled={!isEngineOnline || isTraining || startTraining.isPending}
          className="w-full bg-primary text-primary-foreground hover:bg-primary/90"
        >
          {isTraining ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
          {isTraining ? "Training..." : "Train Agent"}
        </Button>
      </div>

      <Button
        onClick={() => backupModel.mutate()}
        disabled={!health?.agent_loaded || backupModel.isPending}
        variant="outline" className="w-full"
      >
        {backupModel.isPending ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Upload className="h-4 w-4 mr-2" />}
        Backup Model to S3
      </Button>

      {status?.current && (
        <div className="bg-secondary/50 rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Episode</span>
            <span className="font-mono text-sm">{status.current.episode || "—"}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Equity</span>
            <span className="font-mono text-sm text-profit">${status.current.equity?.toFixed(4) || "—"}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Win Rate</span>
            <span className="font-mono text-sm">{status.current.win_rate || 0}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Sentiment Panel ─────────────────────────────────────────────────────────

function SentimentPanel() {
  const [headlines, setHeadlines] = useState("");
  const analyzeSentiment = trpc.sentiment.analyze.useMutation({
    onError: (e) => toast.error(e.message),
  });

  const handleAnalyze = () => {
    const lines = headlines.split("\n").filter(l => l.trim());
    if (lines.length === 0) { toast.error("Enter at least one headline"); return; }
    analyzeSentiment.mutate({ headlines: lines, asset: "BTC" });
  };

  return (
    <div className="space-y-3">
      <textarea
        className="w-full bg-secondary border border-border rounded-lg p-3 text-sm resize-none h-24 focus:outline-none focus:ring-1 focus:ring-primary"
        placeholder="Paste news headlines (one per line)..."
        value={headlines}
        onChange={(e) => setHeadlines(e.target.value)}
      />
      <Button onClick={handleAnalyze} disabled={analyzeSentiment.isPending} className="w-full" size="sm">
        {analyzeSentiment.isPending ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Brain className="h-4 w-4 mr-2" />}
        Analyze Sentiment
      </Button>
      {analyzeSentiment.data && (
        <div className="bg-secondary/50 rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Sentiment Score</span>
            <span className={`font-mono text-sm font-bold ${(analyzeSentiment.data as any).sentiment_score > 0 ? "text-profit" : "text-loss"}`}>
              {((analyzeSentiment.data as any).sentiment_score > 0 ? "+" : "") + (analyzeSentiment.data as any).sentiment_score?.toFixed(2)}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Market Impact</span>
            <Badge variant="outline" className="text-xs">{(analyzeSentiment.data as any).market_impact}</Badge>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Volatility</span>
            <Badge variant="outline" className="text-xs">{(analyzeSentiment.data as any).volatility_expectation}</Badge>
          </div>
          {(analyzeSentiment.data as any).key_factors?.length > 0 && (
            <div>
              <span className="text-xs text-muted-foreground block mb-1">Key Factors</span>
              <div className="flex flex-wrap gap-1">
                {(analyzeSentiment.data as any).key_factors.map((f: string, i: number) => (
                  <Badge key={i} variant="secondary" className="text-xs">{f}</Badge>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Agent Logs ──────────────────────────────────────────────────────────────

function AgentLogs() {
  const { data: logs } = trpc.logs.recent.useQuery({ limit: 30 }, { refetchInterval: 5000 });

  if (!logs || logs.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-muted-foreground text-sm">
        No agent logs yet.
      </div>
    );
  }

  const levelColors: Record<string, string> = {
    info: "text-blue-400", warn: "text-yellow-400", error: "text-red-400",
    trade: "text-profit", risk: "text-loss",
  };

  return (
    <div className="space-y-1 max-h-64 overflow-y-auto font-mono text-xs">
      {logs.map((log: any) => (
        <div key={log.id} className="flex gap-2 py-1 border-b border-border/30">
          <span className="text-muted-foreground shrink-0">
            {log.createdAt ? new Date(log.createdAt).toLocaleTimeString() : "—"}
          </span>
          <Badge variant="outline" className={`text-[10px] px-1 py-0 ${levelColors[log.level] || ""}`}>
            {log.level}
          </Badge>
          <span className="truncate">{log.message}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Training Report ────────────────────────────────────────────────────────

function TrainingReport() {
  const { data: report, isLoading } = trpc.results.report.useQuery();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!report || (report as any).error) {
    return (
      <Card className="bg-card border-border">
        <CardContent className="p-8 text-center text-muted-foreground">
          <FileBarChart className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No training report available. Run the training pipeline first.</p>
        </CardContent>
      </Card>
    );
  }

  const evaluation = (report as any).evaluation || [];
  const charts = (report as any).charts || [];
  const config = (report as any).config || {};

  return (
    <div className="space-y-4">
      {/* Evaluation Table */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            Test Evaluation Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          {evaluation.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-muted-foreground">
                    <th className="text-left py-2 px-3 font-medium">Config</th>
                    <th className="text-right py-2 px-3 font-medium">Final $</th>
                    <th className="text-right py-2 px-3 font-medium">ROI %</th>
                    <th className="text-right py-2 px-3 font-medium">Trades</th>
                    <th className="text-right py-2 px-3 font-medium">Win Rate</th>
                    <th className="text-right py-2 px-3 font-medium">Sharpe</th>
                    <th className="text-right py-2 px-3 font-medium">Max DD</th>
                  </tr>
                </thead>
                <tbody>
                  {evaluation.map((e: any, i: number) => (
                    <tr key={i} className="border-b border-border/50 hover:bg-accent/30">
                      <td className="py-2 px-3 font-mono font-medium">{e.config || e.name || `Config ${i + 1}`}</td>
                      <td className="py-2 px-3 text-right font-mono">${(e.final_equity || 0).toFixed(2)}</td>
                      <td className={`py-2 px-3 text-right font-mono font-bold ${(e.roi_pct || 0) >= 0 ? "text-profit" : "text-loss"}`}>
                        {(e.roi_pct || 0) >= 0 ? "+" : ""}{(e.roi_pct || 0).toFixed(1)}%
                      </td>
                      <td className="py-2 px-3 text-right font-mono">{e.trades || 0}</td>
                      <td className={`py-2 px-3 text-right font-mono ${(e.win_rate || 0) >= 50 ? "text-profit" : "text-loss"}`}>
                        {(e.win_rate || 0).toFixed(1)}%
                      </td>
                      <td className="py-2 px-3 text-right font-mono">{(e.sharpe || 0).toFixed(2)}</td>
                      <td className="py-2 px-3 text-right font-mono text-loss">{(e.max_drawdown || 0).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No evaluation data available.</p>
          )}
        </CardContent>
      </Card>

      {/* Charts Gallery */}
      {charts.length > 0 && (
        <Card className="bg-card border-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Image className="h-4 w-4 text-primary" />
              Training Charts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {charts.map((chart: any, i: number) => (
                <div key={i}>
                  <p className="text-xs text-muted-foreground mb-2">{chart.title || `Chart ${i + 1}`}</p>
                  <img src={chart.url} alt={chart.title || `Chart ${i + 1}`} className="w-full rounded-lg border border-border" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Honest Assessment */}
      <Card className="bg-card border-border border-yellow-500/30">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2 text-yellow-400">
            <AlertTriangle className="h-4 w-4" />
            Honest Assessment & Next Steps
          </CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground space-y-3">
          <p>
            The v9 model achieved <strong className="text-profit">65% win rate</strong> on completely unseen test data
            (7,375 hourly bars where BTC dropped 28.5%). It uses multi-timeframe features (1h/4h/1d),
            WaveTrend + StochRSI + MACD + MFI indicators, and position-based trading with automatic SL/TP.
          </p>
          <div className="bg-secondary/50 rounded-lg p-3">
            <p className="font-medium text-foreground mb-2">Key Results:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
              <div className="flex items-start gap-2">
                <span className="text-profit">+</span>
                <span>65% win rate on unseen test data (BTC dropped 28.5%)</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-profit">+</span>
                <span>Sharpe ratio 15-16 across all leverage levels</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-profit">+</span>
                <span>Consistent performance at 3x, 7x, and 12x leverage</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-yellow-400">!</span>
                <span>High trade frequency (~4000 trades) — needs cooldown for real markets</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-yellow-400">!</span>
                <span>Marginal profitability after real Binance fees (0.05% taker)</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary">-&gt;</span>
                <span>Recommended: Start with Testnet, then paper trade, then small live balance</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ─── Main Dashboard ──────────────────────────────────────────────────────────

export default function Home() {
  const { user, loading, isAuthenticated } = useAuth();
  const { data: stats } = trpc.trading.stats.useQuery(undefined, { refetchInterval: 10000 });
  const { data: portfolio } = trpc.portfolio.current.useQuery(undefined, { refetchInterval: 3000 });
  const { data: liveStatus } = trpc.live.status.useQuery(undefined, { refetchInterval: 3000 });

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  const equity = portfolio?.equity ?? 10;
  const roi = ((equity - 10) / 10 * 100);
  const totalPnl = stats?.totalPnl ?? 0;
  const winRate = stats?.totalTrades ? ((stats.winningTrades ?? 0) / stats.totalTrades * 100) : 0;
  const isLiveRunning = liveStatus?.running;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container flex items-center justify-between h-14 max-w-[1600px]">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Brain className="h-6 w-6 text-primary" />
              <h1 className="text-lg font-bold tracking-tight">RL Trading System</h1>
            </div>
            <Badge variant="outline" className="text-xs border-primary/30 text-primary hidden sm:flex">
              v9 PPO+EWC | 65% WR
            </Badge>
            {isLiveRunning && (
              <Badge className="bg-green-600 text-white text-xs animate-pulse hidden sm:flex">
                <Radio className="h-3 w-3 mr-1" /> LIVE
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-4">
            <EngineStatus />
            {!isAuthenticated ? (
              <Button size="sm" variant="outline" onClick={() => window.location.href = getLoginUrl()}>
                <LogIn className="h-4 w-4 mr-2" /> Sign In
              </Button>
            ) : (
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">{user?.name || "User"}</span>
                <div className="w-7 h-7 rounded-full bg-primary/20 flex items-center justify-center">
                  <span className="text-xs font-bold text-primary">{(user?.name || "U")[0]}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container max-w-[1600px] py-4 space-y-4">
        {/* Metric Cards Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard
            title="Portfolio Value"
            value={`$${equity.toFixed(2)}`}
            subtitle={`${roi >= 0 ? "+" : ""}${roi.toFixed(1)}% ROI`}
            icon={DollarSign}
            trend={roi > 0 ? "up" : roi < 0 ? "down" : "neutral"}
          />
          <MetricCard
            title="Total P&L"
            value={`${totalPnl >= 0 ? "+" : ""}$${totalPnl.toFixed(2)}`}
            subtitle={`${stats?.totalTrades ?? 0} trades`}
            icon={TrendingUp}
            trend={totalPnl > 0 ? "up" : totalPnl < 0 ? "down" : "neutral"}
          />
          <MetricCard
            title="Win Rate"
            value={`${winRate.toFixed(1)}%`}
            subtitle={`${stats?.winningTrades ?? 0}/${stats?.totalTrades ?? 0} wins`}
            icon={Target}
            trend={winRate > 50 ? "up" : winRate > 0 ? "down" : "neutral"}
          />
          <MetricCard
            title="Max Drawdown"
            value={`${(portfolio?.drawdown ?? 0).toFixed(1)}%`}
            subtitle="Peak-to-trough"
            icon={Shield}
            trend={(portfolio?.drawdown ?? 0) > 20 ? "down" : "neutral"}
          />
        </div>

        {/* Tabs */}
        <Tabs defaultValue="dashboard" className="w-full">
          <TabsList className="bg-secondary">
            <TabsTrigger value="dashboard" className="gap-1.5">
              <Activity className="h-3.5 w-3.5" /> Dashboard
            </TabsTrigger>
            <TabsTrigger value="live" className="gap-1.5">
              <Radio className="h-3.5 w-3.5" /> Live Trading
              {isLiveRunning && <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse ml-1" />}
            </TabsTrigger>
            <TabsTrigger value="report" className="gap-1.5">
              <FileBarChart className="h-3.5 w-3.5" /> Training Report
            </TabsTrigger>
          </TabsList>

          {/* Dashboard Tab */}
          <TabsContent value="dashboard" className="mt-4">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2 space-y-4">
                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Activity className="h-4 w-4 text-primary" />
                      Equity Curve
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <EquityCurve />
                  </CardContent>
                </Card>

                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-primary" />
                      Trade History
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <TradeHistory />
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-4">
                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Zap className="h-4 w-4 text-primary" />
                      Agent Control
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <TrainingPanel />
                  </CardContent>
                </Card>

                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Brain className="h-4 w-4 text-primary" />
                      LLM Sentiment Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <SentimentPanel />
                  </CardContent>
                </Card>

                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Activity className="h-4 w-4 text-primary" />
                      Agent Logs
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <AgentLogs />
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Live Trading Tab */}
          <TabsContent value="live" className="mt-4">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2 space-y-4">
                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Zap className="h-4 w-4 text-primary" />
                      Live Trading Control
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <LiveTradingPanel />
                  </CardContent>
                </Card>

                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-primary" />
                      Live Trade History
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <TradeHistory />
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-4">
                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <MessageCircle className="h-4 w-4 text-blue-400" />
                      Telegram Signals
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <TelegramPanel />
                  </CardContent>
                </Card>

                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Brain className="h-4 w-4 text-primary" />
                      LLM Sentiment
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <SentimentPanel />
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Training Report Tab */}
          <TabsContent value="report" className="mt-4">
            <TrainingReport />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
