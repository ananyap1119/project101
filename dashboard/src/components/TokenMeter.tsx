type TokenMeterProps = {
  label: string;
  value: number;
  accent: "mint" | "coral";
};

export function TokenMeter({ label, value, accent }: TokenMeterProps) {
  const accentClass = accent === "mint" ? "text-mint" : "text-coral";

  return (
    <div className="min-w-0 rounded-md border border-line bg-white px-4 py-3">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-semibold ${accentClass}`}>{value.toLocaleString()}</div>
    </div>
  );
}
