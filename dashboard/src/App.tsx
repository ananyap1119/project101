import { Volume2 } from "lucide-react";
import { useRef, useState } from "react";
import { AgentPane, AgentSnapshot } from "./components/AgentPane";
import { VoiceBar } from "./components/VoiceBar";
import { AgentName, MeterEvent, connectRunStream } from "./lib/sse";

const emptySnapshot: AgentSnapshot = {
  visionTokens: 0,
  contextTokens: 0,
  cumulativeCostInr: 0,
  elapsedSeconds: 0,
  currentStep: "idle",
  currentModel: "—",
  latencyMs: 0,
  status: "idle",
};

type Snapshots = Record<AgentName, AgentSnapshot>;

function ComparisonBanner({ snapshots }: { snapshots: Snapshots }) {
  const b = snapshots.baseline;
  const o = snapshots.optimized;
  const hasCost = b.cumulativeCostInr > 0 && o.cumulativeCostInr > 0;
  const hasTime = b.elapsedSeconds > 0 && o.elapsedSeconds > 0;
  if (!hasCost && !hasTime) return null;

  const cheaper = hasCost ? b.cumulativeCostInr / o.cumulativeCostInr : null;
  const faster  = hasTime ? b.elapsedSeconds   / o.elapsedSeconds    : null;
  const bothDone = b.status === "completed" && o.status === "completed";
  const qualifier = bothDone ? "" : " so far";

  return (
    <div className="flex flex-wrap items-center justify-center gap-6 rounded-xl border border-line bg-panel px-6 py-4 text-center shadow-sm">
      {cheaper !== null && (
        <span className="text-sm font-medium text-ink">
          Optimised is{" "}
          <span className="text-3xl font-bold text-emerald-600">{cheaper.toFixed(1)}×</span>{" "}
          cheaper{qualifier}
        </span>
      )}
      {cheaper !== null && faster !== null && (
        <span className="text-slate-300 text-xl">|</span>
      )}
      {faster !== null && (
        <span className="text-sm font-medium text-ink">
          Optimised is{" "}
          <span className="text-3xl font-bold text-blue-600">{faster.toFixed(1)}×</span>{" "}
          faster{qualifier}
        </span>
      )}
    </div>
  );
}

export default function App() {
  const [runId, setRunId]         = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isPlayingReply, setIsPlayingReply] = useState(false);
  const [snapshots, setSnapshots] = useState<Snapshots>({
    baseline:  emptySnapshot,
    optimized: emptySnapshot,
  });

  const sourceRef      = useRef<EventSource | null>(null);
  const voiceMetaRef   = useRef<{ taskType: string; replyLang: string; transcript: string } | null>(null);
  const replyCalledRef = useRef<Set<string>>(new Set());

  async function triggerVoiceReply(
    rid: string,
    success: boolean,
    replyLang: string,
    taskType: string,
    extractedAnswer: string,
    userQuestion: string,
  ) {
    try {
      const resp = await fetch("http://localhost:8000/voice/reply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: rid,
          success,
          reply_language: replyLang,
          task_type: taskType,
          extracted_answer: extractedAnswer,
          user_question: userQuestion,
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

  function handleVoiceRunStarted(rid: string, taskType: string, replyLang: string, transcript: string) {
    voiceMetaRef.current   = { taskType, replyLang, transcript };
    replyCalledRef.current = new Set();
    sourceRef.current?.close();
    setIsRunning(true);
    setRunId(rid);
    setSnapshots({
      baseline:  emptySnapshot,
      optimized: { ...emptySnapshot, status: "running", currentStep: "queued" },
    });
    sourceRef.current = connectRunStream(rid, applyMeterEvent, () => setIsRunning(false));
  }

  function applyMeterEvent(event: MeterEvent) {
    if (event.agent === "system") return;
    setSnapshots((current) => ({
      ...current,
      [event.agent]: {
        visionTokens:      event.vision_tokens,
        contextTokens:     event.context_tokens,
        cumulativeCostInr: event.cumulative_cost_inr,
        elapsedSeconds:    event.elapsed_seconds,
        currentStep:       event.current_step,
        currentModel:      event.current_model,
        latencyMs:         event.latency_ms,
        status:
          event.event === "agent_complete" ? "completed"
          : event.event === "error"        ? "failed"
          : "running",
      },
    }));

    if (
      event.agent === "optimized" &&
      event.event === "agent_complete" &&
      voiceMetaRef.current &&
      !replyCalledRef.current.has(event.run_id)
    ) {
      replyCalledRef.current.add(event.run_id);
      const meta   = voiceMetaRef.current;
      const step   = event.current_step;
      const success = !step.includes("max_steps") &&
                      !step.includes("budget_exceeded") &&
                      !step.includes("error");
      let extractedAnswer = "";
      try {
        const detail = JSON.parse(event.detail ?? "{}");
        extractedAnswer = detail.extracted_answer ?? "";
      } catch { /* ignore */ }
      void triggerVoiceReply(event.run_id, success, meta.replyLang, meta.taskType, extractedAnswer, meta.transcript);
    }
  }

  return (
    <main className="min-h-screen bg-slate-50 px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">

        {/* Header */}
        <header className="mb-6 border-b border-line pb-5">
          <h1 className="text-3xl font-bold tracking-tight text-ink">cua-bench</h1>
          <p className="mt-1 text-sm text-slate-500">
            Voice-first multi-task AI agent — powered by Sarvam AI
          </p>
        </header>

        {/* Two-column layout */}
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">

          {/* ── Left: Voice ── */}
          <div className="flex flex-col gap-4">
            <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-400">
              Voice Input
            </h2>

            <VoiceBar disabled={isRunning} onRunStarted={handleVoiceRunStarted} />

            {isPlayingReply && (
              <div className="flex items-center gap-3 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-medium text-emerald-700">
                <Volume2 size={18} className="animate-pulse" />
                Speaking reply…
              </div>
            )}

            {runId && (
              <p className="text-xs text-slate-400">Run: {runId}</p>
            )}
          </div>

          {/* ── Right: Agents ── */}
          <div className="flex flex-col gap-4">
            <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-400">
              Live Agents
            </h2>

            <AgentPane
              title="Optimised · DeepSeek cascade"
              subtitle="DOM text · summarised history · 4-tier model routing"
              accent="mint"
              snapshot={snapshots.optimized}
            />

            <AgentPane
              title="Baseline · Qwen3-VL-235B"
              subtitle="Full 1080p screenshots · full history · single model"
              accent="coral"
              snapshot={snapshots.baseline}
            />

            <ComparisonBanner snapshots={snapshots} />
          </div>

        </div>
      </div>
    </main>
  );
}
