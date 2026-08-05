"""Prometheus metrics registry shared across the whole platform."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
RTF_BUCKETS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)

# --- STT ------------------------------------------------------------------
stt_requests_total = Counter("stt_requests_total", "STT requests", ["status", "channel"])
stt_audio_seconds_total = Counter("stt_audio_seconds_total", "Total audio seconds processed by STT", ["channel"])
stt_latency_seconds = Histogram("stt_latency_seconds", "STT wall-clock latency", buckets=LATENCY_BUCKETS)
stt_rtf = Histogram("stt_rtf", "STT real-time factor (processing / audio duration)", buckets=RTF_BUCKETS)
stt_confidence = Histogram("stt_confidence", "STT confidence proxy (exp(avg_logprob))", buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
# Eval gauges (set by `speechai evaluate` runs) - used by Prometheus alerts.
stt_wer = Gauge("stt_wer", "STT word error rate from evaluation runs", ["eval_set"])
stt_cer = Gauge("stt_cer", "STT character error rate from evaluation runs", ["eval_set"])
stt_rtf_mean = Gauge("stt_rtf_mean", "STT mean real-time factor from evaluation runs", ["eval_set"])

# --- TTS ------------------------------------------------------------------
tts_requests_total = Counter("tts_requests_total", "TTS requests", ["status", "channel"])
tts_chars_total = Counter("tts_chars_total", "Characters synthesized", ["channel"])
tts_latency_seconds = Histogram("tts_latency_seconds", "TTS wall-clock latency", buckets=LATENCY_BUCKETS)
tts_rtf = Histogram("tts_rtf", "TTS real-time factor", buckets=RTF_BUCKETS)

# --- Models ---------------------------------------------------------------
model_loaded = Gauge("speech_model_loaded", "Whether an engine model is loaded", ["engine"])
model_load_seconds = Histogram("speech_model_load_seconds", "Model load duration", buckets=LATENCY_BUCKETS)

# --- Batch pipeline -------------------------------------------------------
speech_jobs_total = Counter("speech_jobs_total", "Batch jobs processed", ["type", "status"])
speech_jobs_active = Gauge("speech_jobs_active", "Jobs currently running")
speech_queue_depth = Gauge("speech_queue_depth", "Pending jobs in the queue")

# --- HTTP / WebSocket -----------------------------------------------------
http_requests_total = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
http_latency_seconds = Histogram("http_latency_seconds", "HTTP request latency", buckets=LATENCY_BUCKETS)
ws_active = Gauge("speech_ws_active", "Active WebSocket connections", ["kind"])

# --- Errors ---------------------------------------------------------------
errors_total = Counter("speech_errors_total", "Errors raised", ["component", "type"])


def render() -> bytes:
    """Render the full metric exposition format."""
    return generate_latest()
