export type AgentName = "baseline" | "optimized";

export type MeterEvent = {
  run_id: string;
  agent: AgentName | "system";
  event: "usage" | "agent_complete" | "error" | "done";
  vision_tokens: number;
  context_tokens: number;
  cumulative_cost_inr: number;
  elapsed_seconds: number;
  current_step: string;
  current_model: string;
  latency_ms: number;
  detail?: string | null;
  created_at: number;
};

export function connectRunStream(
  runId: string,
  onEvent: (event: MeterEvent) => void,
  onDone: () => void,
): EventSource {
  const source = new EventSource(`http://localhost:8000/stream/${runId}`);

  const handle = (event: Event) => {
    const message = event as MessageEvent<string>;
    onEvent(JSON.parse(message.data) as MeterEvent);
  };

  source.addEventListener("usage", handle);
  source.addEventListener("agent_complete", handle);
  source.addEventListener("error", handle);
  source.addEventListener("done", (event) => {
    handle(event);
    onDone();
    source.close();
  });

  return source;
}
