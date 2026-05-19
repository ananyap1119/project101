type LatencyBarProps = {
  latencyMs: number;
};

export function LatencyBar({ latencyMs }: LatencyBarProps) {
  const width = Math.min(100, Math.max(6, (latencyMs / 800) * 100));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm text-slate-600">
        <span>Latency</span>
        <span className="font-medium text-ink">{latencyMs} ms</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-200">
        <div className="h-full rounded-full bg-ink transition-all" style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}
