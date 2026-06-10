import { useCallback, useEffect, useRef, useState } from "react";
import { Mic } from "lucide-react";

export type IntentResult = {
  run_id: string | null;
  transcript: string;
  intent: {
    task: string;
    train_number: string | null;
    city: string | null;
    product: string | null;
    team: string | null;
    reply_language: string;
    confidence: number;
  };
  action: "started" | "clarification_needed";
  clarification_message: string | null;
};

const TASK_META: Record<string, { icon: string; label: string; color: string }> = {
  train_status: { icon: "🚂", label: "Train Status",  color: "bg-blue-100 text-blue-700" },
  cricket:      { icon: "🏏", label: "Cricket Score", color: "bg-emerald-100 text-emerald-700" },
  weather:      { icon: "🌤", label: "Weather",       color: "bg-sky-100 text-sky-700" },
  blinkit:      { icon: "🛒", label: "Blinkit Price", color: "bg-amber-100 text-amber-700" },
  unknown:      { icon: "❓", label: "Unknown",       color: "bg-slate-100 text-slate-600" },
};

type Props = {
  disabled: boolean;
  onRunStarted: (runId: string, taskType: string, replyLang: string, transcript: string) => void;
};

type RecordState = "idle" | "recording" | "thinking";
type ThinkingLabel = "Transcribing…" | "Understanding…" | "Starting agents…";

const LANGUAGES = [
  { code: "hi-IN", label: "HI" },
  { code: "kn-IN", label: "KN" },
  { code: "en-IN", label: "EN" },
  { code: "te-IN", label: "TE" },
  { code: "ta-IN", label: "TA" },
];

export function VoiceBar({ disabled, onRunStarted }: Props) {
  const [state, setState]                   = useState<RecordState>("idle");
  const [thinkingLabel, setThinkingLabel]   = useState<ThinkingLabel>("Transcribing…");
  const [result, setResult]                 = useState<IntentResult | null>(null);
  const [sttLang, setSttLang]               = useState<string>("hi-IN");
  const sttLangRef                          = useRef<string>("hi-IN");
  const [error, setError]                   = useState<string | null>(null);

  const stateRef         = useRef<RecordState>("idle");
  const thinkingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const chunksRef        = useRef<Blob[]>([]);
  const animRef          = useRef<number>(0);
  const canvasRef        = useRef<HTMLCanvasElement>(null);
  const mediaRef         = useRef<{
    recorder: MediaRecorder;
    stream:   MediaStream;
    audioCtx: AudioContext;
    analyser: AnalyserNode;
    startedAt: number;
  } | null>(null);

  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { sttLangRef.current = sttLang; }, [sttLang]);

  const drawBars = useCallback(() => {
    const canvas   = canvasRef.current;
    const analyser = mediaRef.current?.analyser;
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    const bars = 40;
    const bw   = W / bars;
    for (let i = 0; i < bars; i++) {
      const v = data[Math.floor((i / bars) * data.length)] / 255;
      const hue = 220 + v * 60;
      ctx.fillStyle = `hsl(${hue}, 80%, 55%)`;
      ctx.fillRect(i * bw + 1, H * (1 - v), bw - 2, H * v);
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
      const mimeType = PREF.find((m) => MediaRecorder.isTypeSupported(m)) ?? "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.start(100);

      mediaRef.current = { recorder, stream, audioCtx, analyser, startedAt: Date.now() };
      setState("recording");
      requestAnimationFrame(drawBars);
    } catch {
      setError("Microphone access denied. Please allow microphone access in browser settings.");
    }
  }, [disabled, drawBars]);

  const stop = useCallback(async () => {
    if (stateRef.current !== "recording") return;
    cancelAnimationFrame(animRef.current);
    const media = mediaRef.current;
    if (!media) return;

    const elapsed = Date.now() - media.startedAt;
    if (elapsed < 500) {
      await new Promise<void>((resolve) => setTimeout(resolve, 500 - elapsed));
    }

    await new Promise<void>((resolve) => {
      media.recorder.onstop = () => resolve();
      if (media.recorder.state === "recording") {
        media.recorder.requestData();
      }
      media.recorder.stop();
    });
    media.stream.getTracks().forEach((t) => t.stop());
    void media.audioCtx.close();
    mediaRef.current = null;

    setThinkingLabel("Transcribing…");
    setState("thinking");
    setResult(null);
    setError(null);

    thinkingTimerRef.current = setTimeout(() => setThinkingLabel("Understanding…"), 2000);

    const mimeType = chunksRef.current[0]?.type || "audio/webm";
    const blob     = new Blob(chunksRef.current, { type: mimeType });
    if (blob.size === 0) {
      setError("No audio was recorded. Hold the mic while speaking, then release.");
      setState("idle");
      return;
    }
    const ext      = mimeType.includes("mp4") ? "mp4" : mimeType.includes("ogg") ? "ogg" : "webm";
    const form     = new FormData();
    form.append("audio", blob, `recording.${ext}`);
    form.append("language_hint", sttLangRef.current);

    try {
      const resp = await fetch("http://localhost:8000/voice", { method: "POST", body: form });
      if (!resp.ok) {
        let detail = `Server error ${resp.status}`;
        try {
          const body = await resp.json();
          if (typeof body?.detail === "string") detail = body.detail;
        } catch {
          // Keep the generic status message when the response is not JSON.
        }
        throw new Error(detail);
      }
      const data = (await resp.json()) as IntentResult;

      if (thinkingTimerRef.current) clearTimeout(thinkingTimerRef.current);
      setThinkingLabel("Starting agents…");
      await new Promise<void>((r) => setTimeout(r, 400));

      setResult(data);
      if (data.action === "started" && data.run_id) {
        onRunStarted(
          data.run_id,
          data.intent.task,
          data.intent.reply_language,
          data.transcript,
        );
      }
    } catch (e) {
      if (thinkingTimerRef.current) clearTimeout(thinkingTimerRef.current);
      setError(e instanceof Error ? e.message : "Voice processing failed");
    } finally {
      setState("idle");
    }
  }, [onRunStarted]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as Element)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if ((e.key === "m" || e.key === "M") && !e.repeat) void start();
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === "m" || e.key === "M") void stop();
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup",   onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup",   onKeyUp);
    };
  }, [start, stop]);

  const isRecording = state === "recording";
  const isThinking  = state === "thinking";
  const taskMeta    = result ? (TASK_META[result.intent.task] ?? TASK_META.unknown) : null;

  return (
    <div className="flex flex-col gap-4">
      {/* Mic + waveform row */}
      <div className="flex items-center gap-5 rounded-xl border border-line bg-panel p-5 shadow-sm">
        <button
          type="button"
          aria-label={isRecording ? "Recording…" : "Hold to record"}
          className={[
            "flex h-16 w-16 flex-shrink-0 select-none items-center justify-center",
            "rounded-full border-2 transition-all duration-200",
            isRecording
              ? "animate-pulse border-red-500 bg-red-50 text-red-500 shadow-lg shadow-red-100"
              : isThinking
                ? "cursor-wait border-slate-300 bg-slate-100 text-slate-400"
                : disabled
                  ? "cursor-not-allowed border-slate-200 bg-slate-50 text-slate-300"
                  : "cursor-pointer border-slate-300 bg-white text-slate-600 hover:border-indigo-400 hover:text-indigo-500 hover:shadow-md",
          ].join(" ")}
          disabled={disabled || isThinking}
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            void start();
          }}
          onPointerUp={() => void stop()}
          onPointerCancel={() => { if (isRecording) void stop(); }}
        >
          <Mic size={26} />
        </button>

        <div className="min-w-0 flex-1">
          {isRecording ? (
            <canvas ref={canvasRef} width={400} height={48} className="w-full rounded-md" />
          ) : isThinking ? (
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 animate-bounce rounded-full bg-indigo-400" style={{ animationDelay: "0ms" }} />
              <div className="h-2 w-2 animate-bounce rounded-full bg-indigo-400" style={{ animationDelay: "150ms" }} />
              <div className="h-2 w-2 animate-bounce rounded-full bg-indigo-400" style={{ animationDelay: "300ms" }} />
              <span className="text-sm italic text-slate-500">{thinkingLabel}</span>
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              <p className="text-base font-medium text-ink">Hold mic or press M to speak</p>
              <div className="flex gap-1">
                {LANGUAGES.map((l) => (
                  <button
                    key={l.code}
                    type="button"
                    onClick={() => { setSttLang(l.code); sttLangRef.current = l.code; }}
                    className={[
                      "rounded px-2 py-0.5 text-xs font-semibold transition-colors",
                      sttLang === l.code
                        ? "bg-indigo-600 text-white"
                        : "bg-slate-100 text-slate-500 hover:bg-slate-200",
                    ].join(" ")}
                  >
                    {l.label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Transcript + intent result */}
      {result && (
        <div className="space-y-3 rounded-xl border border-line bg-panel p-5 shadow-sm">
          {/* Transcript */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Transcript</p>
            <p className="mt-1 text-base text-ink">"{result.transcript || "(empty)"}"</p>
          </div>

          {/* Task + language + confidence badges */}
          {taskMeta && (
            <div className="flex flex-wrap items-center gap-2">
              <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-semibold ${taskMeta.color}`}>
                <span>{taskMeta.icon}</span>
                {taskMeta.label}
              </span>
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
                {result.intent.reply_language}
              </span>
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
                {(result.intent.confidence * 100).toFixed(0)}% confident
              </span>
              {result.action === "started" && (
                <span className="rounded-full bg-violet-100 px-3 py-1 text-xs font-semibold text-violet-700">
                  ✓ Agents running
                </span>
              )}
            </div>
          )}

          {/* Clarification message */}
          {result.clarification_message && (
            <p className="text-sm text-amber-700">{result.clarification_message}</p>
          )}
        </div>
      )}

      {error && (
        <p className="rounded-lg bg-red-50 px-4 py-2 text-sm text-red-600">{error}</p>
      )}
    </div>
  );
}
