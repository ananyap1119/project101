import { useCallback, useEffect, useRef, useState } from "react";
import { Mic } from "lucide-react";
import { apiUrl } from "../lib/api";

export type IntentResult = {
  run_id: string | null;
  transcript: string;
  intent: {
    task: string;
    train_number: string | null;
    origin?: string | null;
    destination?: string | null;
    travel_date?: string | null;
    platform?: "amazon" | "flipkart" | "all" | null;
    shopping_action?: "orders" | "cart" | "wishlist" | "saved_items" | "buy_again" | "browsing_history" | "invoices" | null;
    grocery_mode?: "meal_plan" | "missing_ingredients" | "household_restock" | null;
    city: string | null;
    product: string | null;
    team: string | null;
    reply_language: string;
    confidence: number;
  };
  action: "started" | "clarification_needed";
  clarification_message: string | null;
};

const TASK_META: Record<string, { icon: string; label: string }> = {
  train_search: { icon: "TR", label: "Train Journey" },
  shopping_orders: { icon: "SH", label: "Shopping Activity" },
  train_status: { icon: "🚂", label: "Train Status" },
  cricket:      { icon: "🏏", label: "Cricket" },
  weather:      { icon: "🌤", label: "Weather" },
  blinkit:      { icon: "🛒", label: "Blinkit" },
  blinkit_planner: { icon: "GP", label: "Grocery Planner" },
  unknown:      { icon: "?",  label: "Unknown" },
};

const LANGUAGES = [
  { code: "hi-IN", label: "HI" },
  { code: "kn-IN", label: "KN" },
  { code: "en-IN", label: "EN" },
  { code: "te-IN", label: "TE" },
  { code: "ta-IN", label: "TA" },
];

type Props = {
  disabled: boolean;
  onRunStarted: (runId: string, taskType: string, replyLang: string, transcript: string) => void;
};

type RecordState = "idle" | "recording" | "thinking";
type ThinkingLabel = "Transcribing…" | "Understanding…" | "Starting agents…";

export function VoiceBar({ disabled, onRunStarted }: Props) {
  const [state, setState]               = useState<RecordState>("idle");
  const [thinkingLabel, setThinkingLabel] = useState<ThinkingLabel>("Transcribing…");
  const [result, setResult]             = useState<IntentResult | null>(null);
  const [error, setError]               = useState<string | null>(null);
  const [sttLang, setSttLang]           = useState("hi-IN");
  const sttLangRef                      = useRef("hi-IN");

  const stateRef         = useRef<RecordState>("idle");
  const thinkingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const chunksRef        = useRef<Blob[]>([]);
  const animRef          = useRef<number>(0);
  const canvasRef        = useRef<HTMLCanvasElement>(null);
  const mediaRef         = useRef<{ recorder: MediaRecorder; stream: MediaStream; audioCtx: AudioContext; analyser: AnalyserNode; startedAt: number } | null>(null);

  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { sttLangRef.current = sttLang; }, [sttLang]);

  const drawBars = useCallback(() => {
    const canvas = canvasRef.current;
    const analyser = mediaRef.current?.analyser;
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    const bars = 48;
    const bw = W / bars;
    for (let i = 0; i < bars; i++) {
      const v = data[Math.floor((i / bars) * data.length)] / 255;
      const alpha = 0.4 + v * 0.6;
      ctx.fillStyle = `rgba(139, 92, 246, ${alpha})`;
      const h = Math.max(3, H * v);
      ctx.fillRect(i * bw + 1, (H - h) / 2, bw - 2, h);
    }
    animRef.current = requestAnimationFrame(drawBars);
  }, []);

  const start = useCallback(async () => {
    if (stateRef.current !== "idle" || disabled) return;
    try {
      const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new AudioContext();
      const source   = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      const PREF = ["audio/mp4", "audio/ogg;codecs=opus", "audio/webm;codecs=opus", "audio/webm"];
      const mimeType = PREF.find(m => MediaRecorder.isTypeSupported(m)) ?? "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.start(100);
      mediaRef.current = { recorder, stream, audioCtx, analyser, startedAt: Date.now() };
      setState("recording");
      requestAnimationFrame(drawBars);
    } catch {
      setError("Microphone access denied.");
    }
  }, [disabled, drawBars]);

  const stop = useCallback(async () => {
    if (stateRef.current !== "recording") return;
    cancelAnimationFrame(animRef.current);
    const media = mediaRef.current;
    if (!media) return;
    const elapsed = Date.now() - media.startedAt;
    if (elapsed < 500) await new Promise<void>(r => setTimeout(r, 500 - elapsed));
    await new Promise<void>(resolve => {
      media.recorder.onstop = () => resolve();
      if (media.recorder.state === "recording") media.recorder.requestData();
      media.recorder.stop();
    });
    media.stream.getTracks().forEach(t => t.stop());
    void media.audioCtx.close();
    mediaRef.current = null;

    setThinkingLabel("Transcribing…");
    setState("thinking");
    setResult(null);
    setError(null);
    thinkingTimerRef.current = setTimeout(() => setThinkingLabel("Understanding…"), 2000);

    const mimeType = chunksRef.current[0]?.type || "audio/webm";
    const blob = new Blob(chunksRef.current, { type: mimeType });
    if (blob.size === 0) { setError("No audio recorded."); setState("idle"); return; }
    const ext = mimeType.includes("mp4") ? "mp4" : mimeType.includes("ogg") ? "ogg" : "webm";
    const form = new FormData();
    form.append("audio", blob, `recording.${ext}`);
    form.append("language_hint", sttLangRef.current);

    try {
      const resp = await fetch(apiUrl("/voice"), { method: "POST", body: form });
      if (!resp.ok) {
        let detail = `Server error ${resp.status}`;
        try { const b = await resp.json(); if (typeof b?.detail === "string") detail = b.detail; } catch {}
        throw new Error(detail);
      }
      const data = await resp.json() as IntentResult;
      if (thinkingTimerRef.current) clearTimeout(thinkingTimerRef.current);
      setThinkingLabel("Starting agents…");
      await new Promise<void>(r => setTimeout(r, 400));
      setResult(data);
      if (data.action === "started" && data.run_id) {
        onRunStarted(data.run_id, data.intent.task, data.intent.reply_language, data.transcript);
      }
    } catch (e) {
      if (thinkingTimerRef.current) clearTimeout(thinkingTimerRef.current);
      setError(e instanceof Error ? e.message : "Voice processing failed");
    } finally {
      setState("idle");
    }
  }, [onRunStarted]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const tag = (e.target as Element)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if ((e.key === "m" || e.key === "M") && !e.repeat) void start();
    };
    const up = (e: KeyboardEvent) => { if (e.key === "m" || e.key === "M") void stop(); };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, [start, stop]);

  const isRecording = state === "recording";
  const isThinking  = state === "thinking";
  const taskMeta    = result ? (TASK_META[result.intent.task] ?? TASK_META.unknown) : null;

  return (
    <div className="flex flex-col gap-10">

      {/* Mic area */}
      <div className="flex flex-col items-start gap-8">

        {/* Language pills */}
        <div className="flex gap-2">
          {LANGUAGES.map(l => (
            <button
              key={l.code}
              type="button"
              onClick={() => { setSttLang(l.code); sttLangRef.current = l.code; }}
              className={[
                "rounded-full px-3 py-1 text-xs font-semibold tracking-wide transition-all",
                sttLang === l.code
                  ? "bg-violet-500 text-white"
                  : "bg-zinc-800 text-zinc-500 hover:text-zinc-300",
              ].join(" ")}
            >
              {l.label}
            </button>
          ))}
        </div>

        {/* Mic button + waveform */}
        <div className="flex items-center gap-8">
          <button
            type="button"
            disabled={disabled || isThinking}
            onPointerDown={() => void start()}
            onPointerUp={() => void stop()}
            onPointerLeave={() => { if (isRecording) void stop(); }}
            className={[
              "relative flex h-20 w-20 flex-shrink-0 items-center justify-center rounded-full",
              "transition-all duration-300 select-none",
              isRecording
                ? "bg-red-500/10 border-2 border-red-500 text-red-400 shadow-[0_0_40px_rgba(239,68,68,0.3)]"
                : isThinking
                  ? "bg-zinc-800 border-2 border-zinc-700 text-zinc-600 cursor-wait"
                  : disabled
                    ? "bg-zinc-900 border border-zinc-800 text-zinc-700 cursor-not-allowed"
                    : "bg-zinc-900 border border-zinc-700 text-zinc-400 hover:border-violet-500 hover:text-violet-400 hover:shadow-[0_0_30px_rgba(139,92,246,0.15)] cursor-pointer",
            ].join(" ")}
          >
            <Mic size={28} />
            {isRecording && (
              <span className="absolute inset-0 rounded-full border-2 border-red-500/40 animate-ping" />
            )}
          </button>

          {isRecording ? (
            <canvas ref={canvasRef} width={320} height={64} className="rounded-md opacity-90" />
          ) : isThinking ? (
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="h-1.5 w-1.5 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: "120ms" }} />
                <div className="h-1.5 w-1.5 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: "240ms" }} />
                <span className="text-sm text-zinc-500">{thinkingLabel}</span>
              </div>
            </div>
          ) : (
            <div>
              <p className="text-2xl font-light text-zinc-300">Hold to speak</p>
              <p className="mt-1 text-sm text-zinc-600">or press M</p>
            </div>
          )}
        </div>
      </div>

      {/* Transcript + result */}
      {result && (
        <div className="space-y-4">
          <p className="break-words text-2xl font-light leading-snug text-zinc-200 sm:text-3xl">
            "{result.transcript}"
          </p>
          <div className="flex flex-wrap items-center gap-3">
            {taskMeta && (
              <span className="text-sm font-medium text-zinc-300">
                {taskMeta.icon} {taskMeta.label}
              </span>
            )}
            <span className="text-zinc-700">·</span>
            <span className="text-sm text-zinc-500">{result.intent.reply_language}</span>
            <span className="text-zinc-700">·</span>
            <span className="text-sm text-zinc-500">{(result.intent.confidence * 100).toFixed(0)}%</span>
            {result.action === "started" && (
              <>
                <span className="text-zinc-700">·</span>
                <span className="text-sm text-emerald-400">Agents running</span>
              </>
            )}
          </div>
          {result.clarification_message && (
            <p className="text-sm text-amber-500">{result.clarification_message}</p>
          )}
        </div>
      )}

      {error && <p className="text-sm text-red-400">{error}</p>}
    </div>
  );
}
