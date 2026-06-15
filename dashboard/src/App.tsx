import { Volume2 } from "lucide-react";
import { useRef, useState, useEffect } from "react";
import { AgentSnapshot } from "./components/AgentPane";
import { VoiceBar } from "./components/VoiceBar";
import { apiUrl } from "./lib/api";
import { AgentName, MeterEvent, connectRunStream } from "./lib/sse";

const emptySnapshot: AgentSnapshot = {
  visionTokens: 0, contextTokens: 0, cumulativeCostInr: 0,
  elapsedSeconds: 0, currentStep: "idle", currentModel: "—",
  latencyMs: 0, status: "idle",
};

type Snapshots = Record<AgentName, AgentSnapshot>;
type Tab = "voice" | "benchmark";

type BenchmarkResult = {
  name: string;
  label: string;
  status: string;
  total_tokens: number;
  cost_inr: number;
  wall_seconds: number;
};

type TrainOption = {
  number: string;
  name: string;
  days: string;
  category: string;
  departure_time: string;
  departure_station: string;
  arrival_time: string;
  arrival_station: string;
  duration: string;
};

type RailwayResult = {
  origin: string;
  destination: string;
  travel_date: string;
  total_found: number;
  trains: TrainOption[];
};

type ShoppingEntry = {
  platform: "amazon" | "flipkart";
  title: string;
  status: string;
  price: string;
  detail: string;
};

type ShoppingActivity = {
  action: "orders" | "cart" | "wishlist" | "saved_items" | "buy_again" | "browsing_history" | "invoices";
  entries: ShoppingEntry[];
  platforms_checked: string[];
  message: string;
};

const SHOPPING_LABELS: Record<ShoppingActivity["action"], string> = {
  orders: "Recent orders",
  cart: "Shopping cart",
  wishlist: "Wishlist",
  saved_items: "Saved items",
  buy_again: "Buy again",
  browsing_history: "Browsing history",
  invoices: "Invoice availability",
};

type GroceryCartEntry = {
  requested_item: string;
  requested_amount: string;
  requested_quantity: number;
  product_name: string;
  pack_size: string;
  price: string;
  added_quantity: number;
  status: "added" | "unavailable" | "failed";
};

type GroceryPlannerResult = {
  request: string;
  mode: "meal_plan" | "missing_ingredients" | "household_restock";
  entries: GroceryCartEntry[];
  added_count: number;
  unavailable_count: number;
  message: string;
};

function ResultPanel({ answer, taskType, structured }: {
  answer: string;
  taskType: string;
  structured: unknown;
}) {
  if (taskType === "train_search" && structured) {
    const result = structured as RailwayResult;
    return (
      <section className="min-w-0 overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900 p-4 sm:p-6">
        <div className="flex flex-col gap-3 border-b border-zinc-800 pb-5 sm:flex-row sm:items-end sm:justify-between">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-widest text-zinc-600">Train Options</p>
            <h2 className="mt-2 break-words text-xl font-semibold text-zinc-100 sm:text-2xl">
              {result.origin} <span className="text-zinc-600">to</span> {result.destination}
            </h2>
            <p className="mt-1 text-sm text-zinc-500">Travel date: {result.travel_date}</p>
          </div>
          <span className="w-fit rounded-full bg-violet-500/10 px-3 py-1 text-xs font-semibold text-violet-300">
            {result.total_found} found
          </span>
        </div>

        <div className="mt-5 grid min-w-0 gap-4 lg:grid-cols-2">
          {result.trains.map((train) => (
            <article key={train.number} className="min-w-0 rounded-xl border border-zinc-800 bg-zinc-950/60 p-4">
              <div className="flex min-w-0 items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="font-mono text-xs text-violet-400">{train.number}</p>
                  <h3 className="mt-1 break-words text-sm font-semibold text-zinc-100">{train.name}</h3>
                </div>
                <span className="shrink-0 rounded-md bg-zinc-800 px-2 py-1 text-[11px] text-zinc-400">
                  {train.duration}
                </span>
              </div>

              <div className="mt-5 grid grid-cols-[1fr_auto_1fr] items-center gap-3">
                <div className="min-w-0">
                  <p className="text-xl font-semibold text-zinc-100">{train.departure_time}</p>
                  <p className="mt-1 break-words text-xs leading-5 text-zinc-500">{train.departure_station}</p>
                </div>
                <div className="h-px w-8 bg-zinc-700 sm:w-12" />
                <div className="min-w-0 text-right">
                  <p className="text-xl font-semibold text-zinc-100">{train.arrival_time}</p>
                  <p className="mt-1 break-words text-xs leading-5 text-zinc-500">{train.arrival_station}</p>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap gap-2 text-[11px] text-zinc-400">
                <span className="rounded-md bg-zinc-800 px-2 py-1">{train.days}</span>
                <span className="rounded-md bg-zinc-800 px-2 py-1">{train.category}</span>
              </div>
            </article>
          ))}
        </div>
      </section>
    );
  }

  if (taskType === "shopping_orders" && structured) {
    const result = structured as ShoppingActivity;
    return (
      <section className="min-w-0 overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900 p-4 sm:p-6">
        <div className="flex flex-col gap-3 border-b border-zinc-800 pb-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-zinc-600">Shopping Activity</p>
            <h2 className="mt-2 text-xl font-semibold text-zinc-100">{SHOPPING_LABELS[result.action]}</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            {result.platforms_checked.map((platform) => (
              <span key={platform} className="rounded-full bg-violet-500/10 px-3 py-1 text-xs font-semibold capitalize text-violet-300">
                {platform}
              </span>
            ))}
          </div>
        </div>

        {result.entries.length === 0 ? (
          <div className="py-10 text-center">
            <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-full bg-zinc-800 text-lg text-zinc-400">
              0
            </div>
            <p className="mt-4 text-base font-medium text-zinc-200">Nothing to show</p>
            <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-zinc-500">{result.message}</p>
          </div>
        ) : (
          <div className="mt-5 grid gap-4 lg:grid-cols-2">
            {result.entries.map((entry, index) => (
              <article key={`${entry.platform}-${index}`} className="min-w-0 rounded-xl border border-zinc-800 bg-zinc-950/60 p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-xs font-semibold uppercase tracking-widest text-violet-400">{entry.platform}</p>
                    <h3 className="mt-2 break-words text-sm font-semibold text-zinc-100">
                      {entry.title || SHOPPING_LABELS[result.action]}
                    </h3>
                  </div>
                  {(entry.status || entry.price) && (
                    <span className="shrink-0 rounded-md bg-emerald-500/10 px-2 py-1 text-xs text-emerald-300">
                      {entry.status || entry.price}
                    </span>
                  )}
                </div>
                {entry.detail && (
                  <p className="mt-4 line-clamp-3 break-words text-xs leading-5 text-zinc-500">{entry.detail}</p>
                )}
              </article>
            ))}
          </div>
        )}
      </section>
    );
  }

  if (taskType === "blinkit_planner" && structured) {
    const result = structured as GroceryPlannerResult;
    return (
      <section className="min-w-0 overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900 p-4 sm:p-6">
        <div className="flex flex-col gap-3 border-b border-zinc-800 pb-5 sm:flex-row sm:items-end sm:justify-between">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-widest text-zinc-600">Blinkit Grocery Plan</p>
            <h2 className="mt-2 break-words text-xl font-semibold text-zinc-100">{result.request}</h2>
            <p className="mt-1 text-sm text-zinc-500">Cart prepared only. Checkout was not opened.</p>
          </div>
          <span className="w-fit rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-300">
            {result.added_count}/{result.entries.length} added
          </span>
        </div>
        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          {result.entries.map((entry, index) => (
            <article key={`${entry.requested_item}-${index}`} className="min-w-0 rounded-xl border border-zinc-800 bg-zinc-950/60 p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-xs font-semibold uppercase tracking-widest text-violet-400">{entry.requested_item}</p>
                  <h3 className="mt-2 break-words text-sm font-semibold text-zinc-100">
                    {entry.product_name || "No matching product"}
                  </h3>
                </div>
                <span className={[
                  "shrink-0 rounded-md px-2 py-1 text-xs",
                  entry.status === "added" ? "bg-emerald-500/10 text-emerald-300" : "bg-amber-500/10 text-amber-300",
                ].join(" ")}>
                  {entry.status}
                </span>
              </div>
              <div className="mt-4 flex flex-wrap gap-2 text-xs text-zinc-400">
                {entry.requested_amount && <span className="rounded-md bg-zinc-800 px-2 py-1">Need: {entry.requested_amount}</span>}
                {entry.pack_size && <span className="rounded-md bg-zinc-800 px-2 py-1">Pack: {entry.pack_size}</span>}
                {entry.price && <span className="rounded-md bg-zinc-800 px-2 py-1">{entry.price}</span>}
                {entry.added_quantity > 0 && <span className="rounded-md bg-zinc-800 px-2 py-1">Qty: {entry.added_quantity}</span>}
              </div>
            </article>
          ))}
        </div>
      </section>
    );
  }

  return (
    <section className="min-w-0 overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900 p-4 sm:p-6">
      <p className="text-xs font-semibold uppercase tracking-widest text-zinc-600">Result</p>
      <p className="mt-3 whitespace-pre-wrap break-words text-sm leading-6 text-zinc-200">{answer}</p>
    </section>
  );
}

// ── Minimal agent row (voice tab sidebar) ─────────────────────────────────────
function AgentRow({ label, snapshot }: { label: string; snapshot: AgentSnapshot }) {
  const dot =
    snapshot.status === "completed" ? "bg-emerald-400"
    : snapshot.status === "failed"  ? "bg-red-400"
    : snapshot.status === "running" ? "bg-violet-400 animate-pulse"
    : "bg-zinc-700";
  return (
    <div className="flex items-start justify-between gap-4 py-4 border-b border-zinc-800 last:border-0">
      <div className="flex items-center gap-3 min-w-0">
        <span className={`mt-1 h-2 w-2 rounded-full flex-shrink-0 ${dot}`} />
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-widest text-zinc-500">{label}</p>
          {snapshot.status !== "idle" && (
            <p className="mt-0.5 text-sm text-zinc-300 truncate max-w-[200px]">
              {snapshot.currentStep.replace(/_/g, " ")}
            </p>
          )}
          {snapshot.status !== "idle" && snapshot.currentModel !== "—" && (
            <p className="text-xs text-zinc-600 truncate max-w-[200px]">{snapshot.currentModel}</p>
          )}
        </div>
      </div>
      {snapshot.status !== "idle" && (
        <div className="text-right flex-shrink-0">
          <p className="text-sm font-mono text-zinc-300">₹{snapshot.cumulativeCostInr.toFixed(3)}</p>
          <p className="text-xs text-zinc-600">{snapshot.elapsedSeconds.toFixed(1)}s</p>
        </div>
      )}
    </div>
  );
}

// ── Full benchmark panel (benchmark tab) ──────────────────────────────────────
function MetricBlock({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-4">
      <p className="text-xs font-semibold uppercase tracking-widest text-zinc-600">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-zinc-100">{value}</p>
      {sub && <p className="mt-0.5 text-xs text-zinc-600">{sub}</p>}
    </div>
  );
}

function BenchmarkPane({ title, subtitle, accent, snapshot }: {
  title: string; subtitle: string; accent: "violet" | "amber"; snapshot: AgentSnapshot;
}) {
  const accentColor = accent === "violet" ? "text-violet-400" : "text-amber-400";
  const badgeColor =
    snapshot.status === "completed" ? "text-emerald-400 bg-emerald-400/10"
    : snapshot.status === "failed"  ? "text-red-400 bg-red-400/10"
    : snapshot.status === "running" ? "text-violet-400 bg-violet-400/10"
    : "text-zinc-600 bg-zinc-800";
  const pct = snapshot.latencyMs > 0 ? Math.min(100, (snapshot.latencyMs / 3000) * 100) : 0;
  return (
    <div className="rounded-2xl bg-zinc-900 border border-zinc-800 p-6 space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className={`text-base font-semibold ${accentColor}`}>{title}</h2>
          <p className="mt-0.5 text-xs text-zinc-600">{subtitle}</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold capitalize ${badgeColor}`}>
          {snapshot.status}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <MetricBlock label="Vision tokens"   value={snapshot.visionTokens.toLocaleString()} />
        <MetricBlock label="Context tokens"  value={snapshot.contextTokens.toLocaleString()} />
        <MetricBlock label="Cost"            value={`₹${snapshot.cumulativeCostInr.toFixed(3)}`} />
        <MetricBlock label="Elapsed"         value={`${snapshot.elapsedSeconds.toFixed(1)}s`} />
      </div>

      <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-4 space-y-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-zinc-600">Step</p>
          <p className="mt-1 text-sm text-zinc-300 break-words">
            {snapshot.currentStep.replace(/_/g, " ")}
          </p>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-zinc-600">Model</p>
          <p className="mt-1 text-sm text-zinc-300 break-words">{snapshot.currentModel}</p>
        </div>
        <div>
          <div className="flex justify-between text-xs text-zinc-600 mb-1.5">
            <span>Latency</span>
            <span>{snapshot.latencyMs} ms</span>
          </div>
          <div className="h-1 rounded-full bg-zinc-800 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${accent === "violet" ? "bg-violet-500" : "bg-amber-500"}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function ComparisonBanner({ snapshots }: { snapshots: Snapshots }) {
  const b = snapshots.baseline, o = snapshots.optimized;
  const hasCost = b.cumulativeCostInr > 0 && o.cumulativeCostInr > 0;
  const hasTime = b.elapsedSeconds > 0 && o.elapsedSeconds > 0;
  if (!hasCost && !hasTime) return null;
  const cheaper = hasCost ? b.cumulativeCostInr / o.cumulativeCostInr : null;
  const faster  = hasTime ? b.elapsedSeconds / o.elapsedSeconds : null;
  const done    = b.status === "completed" && o.status === "completed";
  return (
    <div className="rounded-2xl bg-zinc-900 border border-zinc-800 p-6 flex flex-wrap gap-8 col-span-full">
      {cheaper !== null && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-zinc-600">
            Cost savings{!done ? " so far" : ""}
          </p>
          <p className="mt-1 text-5xl font-bold text-emerald-400">{cheaper.toFixed(1)}×</p>
          <p className="text-xs text-zinc-600 mt-1">cheaper</p>
        </div>
      )}
      {faster !== null && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-zinc-600">
            Speed gain{!done ? " so far" : ""}
          </p>
          <p className="mt-1 text-5xl font-bold text-violet-400">{faster.toFixed(1)}×</p>
          <p className="text-xs text-zinc-600 mt-1">faster</p>
        </div>
      )}
    </div>
  );
}

// ── Benchmark bar chart ───────────────────────────────────────────────────────
function BenchmarkChart({ results }: { results: BenchmarkResult[] }) {
  if (results.length === 0) {
    return (
      <div className="rounded-2xl border border-zinc-800 border-dashed p-8 text-center col-span-full">
        <p className="text-sm text-zinc-600">Run a benchmark to see results here.</p>
      </div>
    );
  }
  const maxTokens = Math.max(...results.map(r => r.total_tokens), 1);
  return (
    <div className="rounded-2xl bg-zinc-900 border border-zinc-800 p-6 col-span-full space-y-1">
      <div className="flex items-end justify-between mb-6">
        <div>
          <h2 className="text-base font-semibold text-zinc-100">Benchmark Comparison</h2>
          <p className="text-xs text-zinc-600 mt-0.5">Bars appear as each config completes.</p>
        </div>
        <span className="text-xs text-zinc-500">{results.length}/5 complete</span>
      </div>
      <div className="space-y-5">
        {results.map(r => {
          const width = Math.max(4, (r.total_tokens / maxTokens) * 100);
          const done = r.status === "completed";
          return (
            <div key={r.name}>
              <div className="flex items-center justify-between mb-2 gap-4">
                <span className="text-sm font-medium text-zinc-200 min-w-0 truncate">{r.label}</span>
                <span className="text-xs text-zinc-500 flex-shrink-0 font-mono">
                  {r.status} · {r.total_tokens.toLocaleString()} tok · ₹{r.cost_inr.toFixed(4)} · {r.wall_seconds.toFixed(1)}s
                </span>
              </div>
              <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${done ? "bg-emerald-500" : "bg-zinc-600"}`}
                  style={{ width: `${width}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main app ──────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab]                       = useState<Tab>("voice");
  const [isRunning, setIsRunning]           = useState(false);
  const [isPlayingReply, setIsPlayingReply] = useState(false);
  const [snapshots, setSnapshots]           = useState<Snapshots>({ baseline: emptySnapshot, optimized: emptySnapshot });
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[]>([]);
  const [latestAnswer, setLatestAnswer]         = useState("");
  const [latestStructured, setLatestStructured] = useState<unknown>(null);

  useEffect(() => {
    const id = window.setInterval(async () => {
      try {
        const r = await fetch(apiUrl("/benchmark-results"));
        if (r.ok) {
          const p = await r.json() as { results: BenchmarkResult[] };
          setBenchmarkResults(p.results);
        }
      } catch {}
    }, 1500);
    return () => window.clearInterval(id);
  }, []);

  const sourceRef      = useRef<EventSource | null>(null);
  const voiceMetaRef   = useRef<{ taskType: string; replyLang: string; transcript: string } | null>(null);
  const replyCalledRef = useRef<Set<string>>(new Set());

  async function triggerVoiceReply(rid: string, success: boolean, replyLang: string, taskType: string, extracted: string, question: string) {
    try {
      const resp = await fetch(apiUrl("/voice/reply"), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: rid, success, reply_language: replyLang, task_type: taskType, extracted_answer: extracted, user_question: question }),
      });
      if (!resp.ok) return;
      const url = URL.createObjectURL(await resp.blob());
      const audio = new Audio(url);
      setIsPlayingReply(true);
      audio.onended = () => { setIsPlayingReply(false); URL.revokeObjectURL(url); };
      audio.onerror = () => { setIsPlayingReply(false); URL.revokeObjectURL(url); };
      await audio.play().catch(() => setIsPlayingReply(false));
    } catch { setIsPlayingReply(false); }
  }

  function handleVoiceRunStarted(rid: string, taskType: string, replyLang: string, transcript: string) {
    voiceMetaRef.current   = { taskType, replyLang, transcript };
    replyCalledRef.current = new Set();
    sourceRef.current?.close();
    setIsRunning(true);
    setLatestAnswer("");
    setLatestStructured(null);
    setSnapshots({ baseline: emptySnapshot, optimized: { ...emptySnapshot, status: "running", currentStep: "queued" } });
    sourceRef.current = connectRunStream(rid, applyMeterEvent, () => setIsRunning(false));
  }

  function applyMeterEvent(event: MeterEvent) {
    if (event.agent === "system") return;
    setSnapshots(cur => ({
      ...cur,
      [event.agent]: {
        visionTokens: event.vision_tokens, contextTokens: event.context_tokens,
        cumulativeCostInr: event.cumulative_cost_inr, elapsedSeconds: event.elapsed_seconds,
        currentStep: event.current_step, currentModel: event.current_model,
        latencyMs: event.latency_ms,
        status: event.event === "agent_complete" ? "completed" : event.event === "error" ? "failed" : "running",
      },
    }));
    if (event.agent === "optimized" && event.event === "agent_complete" && voiceMetaRef.current && !replyCalledRef.current.has(event.run_id)) {
      replyCalledRef.current.add(event.run_id);
      const meta = voiceMetaRef.current;
      let extracted = "";
      let success = !event.current_step.includes("max_steps") && !event.current_step.includes("budget") && !event.current_step.includes("error");
      try {
        const detail = JSON.parse(event.detail ?? "{}") as {
          extracted_answer?: string;
          structured_result?: unknown;
          success?: boolean;
        };
        extracted = detail.extracted_answer ?? "";
        setLatestStructured(detail.structured_result ?? null);
        if (typeof detail.success === "boolean") success = detail.success;
      } catch {}
      setLatestAnswer(extracted);
      void triggerVoiceReply(event.run_id, success, meta.replyLang, meta.taskType, extracted, meta.transcript);
    }
  }

  return (
    <div className="min-h-screen overflow-x-hidden bg-zinc-950 text-zinc-50">
      <div className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8 xl:py-12">

        {/* Header */}
        <div className="mb-10 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">agent 101</h1>
            <p className="text-xs text-zinc-600 mt-0.5">powered by Sarvam AI</p>
          </div>
          <div className="flex items-center gap-4">
            {isPlayingReply && (
              <div className="flex items-center gap-2 text-sm text-emerald-400">
                <Volume2 size={14} className="animate-pulse" />
                Speaking…
              </div>
            )}
            {/* Tabs */}
            <div className="flex gap-1 rounded-xl bg-zinc-900 p-1 border border-zinc-800">
              {(["voice", "benchmark"] as Tab[]).map(t => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={[
                    "rounded-lg px-4 py-1.5 text-sm font-medium capitalize transition-all",
                    tab === t ? "bg-zinc-700 text-zinc-100" : "text-zinc-500 hover:text-zinc-300",
                  ].join(" ")}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Voice tab */}
        {tab === "voice" && (
          <div className="grid min-w-0 grid-cols-1 gap-10 xl:grid-cols-[minmax(0,1fr)_320px]">
            <div className="min-w-0 space-y-8">
              <VoiceBar disabled={isRunning} onRunStarted={handleVoiceRunStarted} />
              {latestAnswer && (
                <ResultPanel
                  answer={latestAnswer}
                  taskType={voiceMetaRef.current?.taskType ?? "unknown"}
                  structured={latestStructured}
                />
              )}
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-zinc-600 mb-6">Live Agent</p>
              <AgentRow label="Optimised · DeepSeek cascade" snapshot={snapshots.optimized} />
              <AgentRow label="Baseline · Qwen3-VL" snapshot={snapshots.baseline} />
              {(snapshots.optimized.cumulativeCostInr > 0 || snapshots.baseline.cumulativeCostInr > 0) && (
                <div className="pt-6 flex gap-8">
                  {snapshots.optimized.cumulativeCostInr > 0 && snapshots.baseline.cumulativeCostInr > 0 && (
                    <div>
                      <p className="text-xs text-zinc-600 uppercase tracking-widest">cheaper</p>
                      <p className="text-3xl font-bold text-emerald-400">
                        {(snapshots.baseline.cumulativeCostInr / snapshots.optimized.cumulativeCostInr).toFixed(1)}×
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Benchmark tab */}
        {tab === "benchmark" && (
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
            <BenchmarkPane
              title="Optimised · DeepSeek Cascade"
              subtitle="DOM text · summarised history · 4-tier model routing"
              accent="violet"
              snapshot={snapshots.optimized}
            />
            <BenchmarkPane
              title="Baseline · Qwen3-VL-235B"
              subtitle="Full 1080p screenshots · full history · single model"
              accent="amber"
              snapshot={snapshots.baseline}
            />
            <ComparisonBanner snapshots={snapshots} />
            <BenchmarkChart results={benchmarkResults} />
          </div>
        )}

      </div>
    </div>
  );
}
