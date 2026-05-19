import { Play, Volume2 } from "lucide-react";
import { useRef, useState, useEffect } from "react";
import { AgentPane, AgentSnapshot } from "./components/AgentPane";
import { VoiceBar } from "./components/VoiceBar";
import { AgentName, MeterEvent, connectRunStream } from "./lib/sse";

const emptySnapshot: AgentSnapshot = {
  visionTokens: 0,
  contextTokens: 0,
  cumulativeCostInr: 0,
  elapsedSeconds: 0,
  currentStep: "idle",
  currentModel: "none",
  latencyMs: 0,
  status: "idle",
};

type Snapshots = Record<AgentName, AgentSnapshot>;

type BenchmarkResult = {
  name: string;
  label: string;
  status: string;
  total_tokens: number;
  cost_inr: number;
  wall_seconds: number;
};

function ComparisonBanner({ snapshots }: { snapshots: Snapshots }) {
  const b = snapshots.baseline;
  const o = snapshots.optimized;
  const hasCost = b.cumulativeCostInr > 0 && o.cumulativeCostInr > 0;
  const hasTime = b.elapsedSeconds > 0 && o.elapsedSeconds > 0;
  if (!hasCost && !hasTime) return null;

  const cheaper = hasCost ? b.cumulativeCostInr / o.cumulativeCostInr : null;
  const faster  = hasTime ? b.elapsedSeconds   / o.elapsedSeconds    : null;
  const bothDone =
    b.status === "completed" && o.status === "completed";
  const qualifier = bothDone ? "" : " so far";

  return (
    <div className="flex flex-wrap items-center justify-center gap-6 rounded-lg border border-line bg-panel px-6 py-4 text-center shadow-sm">
      {cheaper !== null && (
        <span className="text-base font-medium text-ink">
          Optimised is{" "}
          <span className="text-2xl font-bold text-emerald-600">
            {cheaper.toFixed(1)}×
          </span>{" "}
          cheaper{qualifier}
        </span>
      )}
      {cheaper !== null && faster !== null && (
        <span className="text-slate-300 text-xl">|</span>
      )}
      {faster !== null && (
        <span className="text-base font-medium text-ink">
          Optimised is{" "}
          <span className="text-2xl font-bold text-blue-600">
            {faster.toFixed(1)}×
          </span>{" "}
          faster{qualifier}
        </span>
      )}
    </div>
  );
}

export default function App() {
  const [runId, setRunId]           = useState<string | null>(null);
  const [isRunning, setIsRunning]   = useState(false);
  const [error, setError]           = useState<string | null>(null);
  const [isPlayingReply, setIsPlayingReply] = useState(false);
  const [currentTrainNumber, setCurrentTrainNumber] = useState<string>("22691");
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[]>([]);
  const [snapshots, setSnapshots] = useState<Snapshots>({
    baseline:  emptySnapshot,
    optimized: emptySnapshot,
  });
  const sourceRef       = useRef<EventSource | null>(null);
  const voiceMetaRef    = useRef<{ trainNumber: string; replyLang: string } | null>(null);
  const replyCalledRef  = useRef<Set<string>>(new Set());

  useEffect(() => {
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch("http://localhost:8000/benchmark-results");
        if (response.ok) {
          const payload = (await response.json()) as { results: BenchmarkResult[] };
          setBenchmarkResults(payload.results);
        }
      } catch {
        // Keep rendering while the backend starts.
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  async function triggerVoiceReply(
    rid: string,
    success: boolean,
    trainNumber: string,
    replyLang: string,
  ) {
    try {
      const resp = await fetch("http://localhost:8000/voice/reply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: rid, success, train_number: trainNumber, reply_language: replyLang,
        }),
      });
      if (!resp.ok) return;
      const blob  = await resp.blob();
      const url   = URL.createObjectURL(blob);
      const audio = new Audio(url);
      setIsPlayingReply(true);
      audio.onended = () => { setIsPlayingReply(false); URL.revokeObjectURL(url); };
      audio.onerror = () => { setIsPlayingReply(false); URL.revokeObjectURL(url); };
      await audio.play().catch(() => setIsPlayingReply(false));
    } catch {
      setIsPlayingReply(false);
    }
  }

  function handleVoiceRunStarted(rid: string, trainNumber: string, replyLang: string) {
    voiceMetaRef.current   = { trainNumber, replyLang };
    replyCalledRef.current = new Set();
    setCurrentTrainNumber(trainNumber);
    sourceRef.current?.close();
    setError(null);
    setIsRunning(true);
    setRunId(rid);
    setSnapshots({
      baseline:  { ...emptySnapshot, status: "running", currentStep: "queued" },
      optimized: { ...emptySnapshot, status: "running", currentStep: "queued" },
    });
    sourceRef.current = connectRunStream(rid, applyMeterEvent, () => setIsRunning(false));
  }

  async function runBoth() {
    sourceRef.current?.close();
    setError(null);
    setIsRunning(true);
    setRunId(null);
    setCurrentTrainNumber("22691");
    setSnapshots({
      baseline:  { ...emptySnapshot, status: "running", currentStep: "queued" },
      optimized: { ...emptySnapshot, status: "running", currentStep: "queued" },
    });

    try {
      const response = await fetch("http://localhost:8000/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: "ntes", headed: false }),
      });
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      const payload = (await response.json()) as { run_id: string };
      setRunId(payload.run_id);
      sourceRef.current = connectRunStream(payload.run_id, applyMeterEvent, () => {
        setIsRunning(false);
      });
    } catch (caught) {
      setIsRunning(false);
      setError(caught instanceof Error ? caught.message : "Unable to start run");
    }
  }

  function applyMeterEvent(event: MeterEvent) {
    if (event.agent === "system") return;
    setSnapshots((current) => ({
      ...current,
      [event.agent]: {
        visionTokens:       event.vision_tokens,
        contextTokens:      event.context_tokens,
        cumulativeCostInr:  event.cumulative_cost_inr,
        elapsedSeconds:     event.elapsed_seconds,
        currentStep:        event.current_step,
        currentModel:       event.current_model,
        latencyMs:          event.latency_ms,
        status:
          event.event === "agent_complete" ? "completed"
          : event.event === "error"        ? "failed"
          : "running",
      },
    }));

    // Voice reply: fire when optimized agent finishes in a voice-triggered run.
    if (
      event.agent === "optimized" &&
      event.event === "agent_complete" &&
      voiceMetaRef.current &&
      !replyCalledRef.current.has(event.run_id)
    ) {
      replyCalledRef.current.add(event.run_id);
      const meta    = voiceMetaRef.current;
      const step    = event.current_step;
      const success = !step.includes("max_steps") &&
                      !step.includes("budget_exceeded") &&
                      !step.includes("error");
      void triggerVoiceReply(event.run_id, success, meta.trainNumber, meta.replyLang);
    }
  }

  return (
    <main className="min-h-screen px-4 py-5 sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">

        <header className="flex flex-col gap-4 border-b border-line pb-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-normal text-ink">cua-bench</h1>
            <p className="mt-1 text-sm text-slate-600">
              Qwen3-VL-235B (full screenshots) vs DeepSeek-first cascade (DOM text) — Train {currentTrainNumber}
            </p>
          </div>
          <button
            className="inline-flex h-11 items-center justify-center gap-2 rounded-md bg-ink px-5 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
            disabled={isRunning}
            onClick={runBoth}
            type="button"
          >
            <Play size={17} />
            {isRunning ? "Running…" : "Run Both"}
          </button>
        </header>

        {(runId || error) && (
          <div className="min-h-8 text-sm">
            {runId && <span className="font-medium text-slate-700">Run ID: {runId}</span>}
            {error && <span className="font-medium text-red-700">{error}</span>}
          </div>
        )}

        <VoiceBar disabled={isRunning} onRunStarted={handleVoiceRunStarted} />

        {isPlayingReply && (
          <div className="flex items-center gap-2 rounded-md border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm font-medium text-emerald-700">
            <Volume2 size={16} className="animate-pulse" />
            Playing voice reply…
          </div>
        )}

        <ComparisonBanner snapshots={snapshots} />

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
          <AgentPane
            title="Baseline · Qwen3-VL-235B"
            subtitle="Full 1080p PNG every step · Full conversation history · OpenRouter"
            accent="coral"
            snapshot={snapshots.baseline}
          />
          <AgentPane
            title="Optimised · DeepSeek-first Cascade"
            subtitle="DOM text (no images) · Summarised history · DeepSeek→Qwen→Claude · Speculation"
            accent="mint"
            snapshot={snapshots.optimized}
          />
        </div>

        <section className="rounded-lg border border-line bg-panel p-5 shadow-sm">
          <div className="flex items-end justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold text-ink">Benchmark Comparison</h2>
              <p className="mt-1 text-sm text-slate-600">Bars appear as each config completes.</p>
            </div>
            <span className="text-sm font-medium text-slate-600">
              {benchmarkResults.length}/6 complete
            </span>
          </div>
          <div className="mt-5 space-y-4">
            {benchmarkResults.map((result) => {
              const maxTokens = Math.max(...benchmarkResults.map((item) => item.total_tokens), 1);
              const width = Math.max(6, (result.total_tokens / maxTokens) * 100);
              return (
                <div key={result.name} className="space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
                    <span className="font-semibold text-ink">{result.label}</span>
                    <span className="text-slate-600">
                      {result.status} · {result.total_tokens.toLocaleString()} tok · ₹
                      {result.cost_inr.toFixed(4)} · {result.wall_seconds.toFixed(1)}s
                    </span>
                  </div>
                  <div className="h-3 overflow-hidden rounded-full bg-slate-200">
                    <div
                      className="h-full rounded-full bg-mint transition-all"
                      style={{ width: `${width}%` }}
                    />
                  </div>
                </div>
              );
            })}
            {!benchmarkResults.length && (
              <div className="rounded-md border border-dashed border-line p-4 text-sm text-slate-600">
                Waiting for the first config result.
              </div>
            )}
          </div>
        </section>

      </div>
    </main>
  );
}
