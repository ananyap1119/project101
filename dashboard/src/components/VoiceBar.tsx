import { useCallback, useEffect, useRef, useState } from "react";
import { Mic } from "lucide-react";

export type IntentResult = {
  run_id: string | null;
  transcript: string;
  intent: {
    task: string;
    train_number: string | null;
    reply_language: string;
    confidence: number;
  };
  action: "started" | "clarification_needed";
  clarification_message: string | null;
};

type Props = {
  disabled: boolean;
  onRunStarted: (runId: string, trainNumber: string, replyLang: string) => void;
};

type RecordState = "idle" | "recording" | "thinking";
// TODO: upgrade to real SSE streaming from /voice for accurate stage transitions
type ThinkingLabel = "Transcribing…" | "Understanding…" | "Starting agents…";

export function VoiceBar({ disabled, onRunStarted }: Props) {
  const [state, setState]           = useState<RecordState>("idle");
  const [thinkingLabel, setThinkingLabel] = useState<ThinkingLabel>("Transcribing…");
  const [result, setResult]         = useState<IntentResult | null>(null);
  const [error, setError]           = useState<string | null>(null);

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
  } | null>(null);

  // Keep stateRef in sync to avoid stale closures in key handlers.
  useEffect(() => { stateRef.current = state; }, [state]);

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
    const bars = 32;
    const bw   = W / bars;
    for (let i = 0; i < bars; i++) {
      const v = data[Math.floor((i / bars) * data.length)] / 255;
      ctx.fillStyle = "#ef4444";
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

      // Prefer Chrome's stable WebM/Opus output; the backend normalizes it to WAV.
      const PREF = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/ogg", "audio/mp4"];
      const mimeType = PREF.find((m) => MediaRecorder.isTypeSupported(m)) ?? "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.start(100);

      mediaRef.current = { recorder, stream, audioCtx, analyser };
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

    await new Promise<void>((resolve) => {
      media.recorder.onstop = () => resolve();
      media.recorder.stop();
    });
    media.stream.getTracks().forEach((t) => t.stop());
    void media.audioCtx.close();
    mediaRef.current = null;

    // Stage 1 — Transcribing (0 s)
    setThinkingLabel("Transcribing…");
    setState("thinking");
    setResult(null);
    setError(null);

    // Stage 2 — Understanding (2 s after POST sent; Sarvam-M thinking takes 5–15 s)
    thinkingTimerRef.current = setTimeout(() => setThinkingLabel("Understanding…"), 2000);

    const mimeType = chunksRef.current[0]?.type || "audio/webm";
    const blob     = new Blob(chunksRef.current, { type: mimeType });
    if (blob.size === 0) {
      setError("No audio was recorded. Please hold the mic a little longer.");
      setState("idle");
      return;
    }
    const ext      = mimeType.includes("mp4") ? "mp4" : mimeType.includes("ogg") ? "ogg" : "webm";
    const form     = new FormData();
    form.append("audio", blob, `recording.${ext}`);

    try {
      const resp = await fetch("http://localhost:8000/voice", { method: "POST", body: form });
      if (!resp.ok) {
        let detail = `Server error ${resp.status}`;
        try {
          const errorBody = (await resp.json()) as { detail?: string };
          if (errorBody.detail) detail = `${detail}: ${errorBody.detail}`;
        } catch {
          // Keep the status-only message if the response is not JSON.
        }
        throw new Error(detail);
      }
      const data = (await resp.json()) as IntentResult;

      // Stage 3 — Starting agents (briefly shown when intent is confirmed)
      if (thinkingTimerRef.current) clearTimeout(thinkingTimerRef.current);
      setThinkingLabel("Starting agents…");
      await new Promise<void>((r) => setTimeout(r, 450));

      setResult(data);
      if (data.action === "started" && data.run_id) {
        onRunStarted(
          data.run_id,
          data.intent.train_number ?? "22691",
          data.intent.reply_language,
        );
      }
    } catch (e) {
      if (thinkingTimerRef.current) clearTimeout(thinkingTimerRef.current);
      setError(e instanceof Error ? e.message : "Voice processing failed");
    } finally {
      setState("idle");
    }
  }, [onRunStarted]);

  // Hold M to record, release to send.  Ignore when typing in inputs.
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

  return (
    <div className="rounded-lg border border-line bg-panel p-4 shadow-sm">
      <div className="flex items-center gap-4">
        {/* Mic button — press-and-hold */}
        <button
          type="button"
          aria-label={isRecording ? "Recording…" : "Hold to record"}
          className={[
            "flex h-12 w-12 flex-shrink-0 select-none items-center justify-center",
            "rounded-full border-2 transition-colors",
            isRecording
              ? "animate-pulse border-red-500 bg-red-50 text-red-600"
              : isThinking
                ? "cursor-wait border-slate-300 bg-slate-100 text-slate-400"
                : disabled
                  ? "cursor-not-allowed border-slate-200 bg-slate-100 text-slate-400"
                  : "cursor-pointer border-slate-300 bg-white text-slate-600 hover:border-slate-400",
          ].join(" ")}
          disabled={disabled || isThinking}
          onPointerDown={() => void start()}
          onPointerUp={() => void stop()}
          onPointerLeave={() => { if (isRecording) void stop(); }}
        >
          <Mic size={22} />
        </button>

        {/* Status / waveform */}
        <div className="min-w-0 flex-1">
          {isRecording ? (
            <canvas ref={canvasRef} width={320} height={40} className="w-full rounded" />
          ) : isThinking ? (
            <p className="text-sm italic text-slate-500">{thinkingLabel}</p>
          ) : (
            <p className="text-sm text-slate-500">
              Hold{" "}
              <kbd className="rounded border border-slate-200 bg-slate-100 px-1.5 py-0.5 font-mono text-xs">
                M
              </kbd>{" "}
              or press-and-hold the mic to speak a train query
            </p>
          )}
        </div>
      </div>

      {/* Intent card */}
      {result && (
        <div className="mt-3 space-y-1.5 rounded-md border border-slate-200 bg-white p-3 text-sm">
          <div>
            <span className="font-medium text-slate-500">Transcript: </span>
            <span className="text-ink">{result.transcript || "(empty)"}</span>
          </div>
          <div className="flex flex-wrap gap-2 text-xs">
            <span
              className={`rounded-full px-2 py-0.5 font-semibold ${
                result.intent.task === "train_status"
                  ? "bg-emerald-100 text-emerald-700"
                  : "bg-amber-100 text-amber-700"
              }`}
            >
              {result.intent.task}
            </span>
            {result.intent.train_number && (
              <span className="rounded-full bg-blue-100 px-2 py-0.5 font-semibold text-blue-700">
                Train {result.intent.train_number}
              </span>
            )}
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-slate-600">
              {result.intent.reply_language} · {(result.intent.confidence * 100).toFixed(0)}%
            </span>
            {result.action === "started" && (
              <span className="rounded-full bg-violet-100 px-2 py-0.5 font-semibold text-violet-700">
                ✓ Benchmark started
              </span>
            )}
          </div>
          {result.clarification_message && (
            <p className="text-slate-600">{result.clarification_message}</p>
          )}
        </div>
      )}

      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </div>
  );
}
