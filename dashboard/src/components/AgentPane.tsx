import { Activity, Gauge, IndianRupee, Timer } from "lucide-react";
import { LatencyBar } from "./LatencyBar";
import { TokenMeter } from "./TokenMeter";

export type AgentSnapshot = {
  visionTokens: number;
  contextTokens: number;
  cumulativeCostInr: number;
  elapsedSeconds: number;
  currentStep: string;
  currentModel: string;
  latencyMs: number;
  status: "idle" | "running" | "completed" | "failed";
};

type AgentPaneProps = {
  title: string;
  subtitle: string;
  accent: "mint" | "coral";
  snapshot: AgentSnapshot;
};

export function AgentPane({ title, subtitle, accent, snapshot }: AgentPaneProps) {
  const badgeClass =
    snapshot.status === "completed"
      ? "bg-emerald-100 text-emerald-700"
      : snapshot.status === "failed"
        ? "bg-red-100 text-red-700"
        : snapshot.status === "running"
          ? "bg-blue-100 text-blue-700"
          : "bg-slate-200 text-slate-600";

  return (
    <section className="flex min-h-[520px] flex-col rounded-lg border border-line bg-panel p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-ink">{title}</h2>
          <p className="mt-1 text-sm text-slate-600">{subtitle}</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold capitalize ${badgeClass}`}>
          {snapshot.status}
        </span>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <TokenMeter label="Vision tokens" value={snapshot.visionTokens} accent={accent} />
        <TokenMeter label="Context tokens" value={snapshot.contextTokens} accent={accent} />
      </div>

      <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded-md border border-line bg-white px-4 py-3">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-500">
            <IndianRupee size={14} />
            Cost
          </div>
          <div className="mt-1 text-2xl font-semibold text-ink">
            ₹{snapshot.cumulativeCostInr.toFixed(2)}
          </div>
        </div>
        <div className="rounded-md border border-line bg-white px-4 py-3">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-500">
            <Timer size={14} />
            Elapsed
          </div>
          <div className="mt-1 text-2xl font-semibold text-ink">
            {snapshot.elapsedSeconds.toFixed(1)}s
          </div>
        </div>
      </div>

      <div className="mt-6 space-y-4 rounded-md border border-line bg-white p-4">
        <div className="flex items-start gap-3">
          <Activity className="mt-0.5 text-slate-500" size={18} />
          <div className="min-w-0">
            <div className="text-xs font-medium uppercase tracking-wide text-slate-500">Step</div>
            <div className="mt-1 break-words text-base font-medium text-ink">
              {snapshot.currentStep}
            </div>
          </div>
        </div>
        <div className="flex items-start gap-3">
          <Gauge className="mt-0.5 text-slate-500" size={18} />
          <div className="min-w-0">
            <div className="text-xs font-medium uppercase tracking-wide text-slate-500">Model</div>
            <div className="mt-1 break-words text-base font-medium text-ink">
              {snapshot.currentModel}
            </div>
          </div>
        </div>
        <LatencyBar latencyMs={snapshot.latencyMs} />
      </div>
    </section>
  );
}
