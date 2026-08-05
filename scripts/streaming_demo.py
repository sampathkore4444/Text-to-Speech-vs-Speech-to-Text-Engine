"""Live streaming demo client for the Bank Speech AI WebSocket API.

Transcribe from your microphone (partial + final events print live)::

    python scripts/streaming_demo.py --transcribe
    python scripts/streaming_demo.py --transcribe --url ws://localhost:8000 --language en

Transcribe from a 16 kHz mono WAV instead of the mic (handy on no-mic machines
or for CI)::

    python scripts/streaming_demo.py --transcribe --file call.wav

Synthesize text and play it back sentence-by-sentence as chunks arrive::

    python scripts/streaming_demo.py --synthesize "Your balance is one thousand dollars." --speed 1.1

Or save the synthesized audio to disk instead of playing it::

    python scripts/streaming_demo.py --synthesize "Hello from the bank." --output out.wav

Extra demo-only dependencies (``websockets`` ships with ``uvicorn[standard]``)::

    pip install sounddevice websockets

Requires the API to be running: ``uvicorn speechai.api.app:app --host 0.0.0.0 --port 8000``
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np

DEFAULT_URL = "ws://localhost:8000"
SAMPLE_RATE = 16000
CHUNK_FRAMES = 1600  # 100 ms @ 16 kHz


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _require(module: str, hint: str) -> object:
    try:
        return __import__(module)
    except ImportError as exc:
        raise SystemExit(f"missing dependency: {module} - {hint}") from exc


def _print_partial(text: str) -> None:
    sys.stdout.write(f"\r\x1b[2m▍ {text}\x1b[0m")
    sys.stdout.flush()


def _print_final(event: dict) -> None:
    sys.stdout.write("\r\x1b[K")  # clear the partial line
    index = event.get("utterance_index", "?")
    confidence = event.get("confidence")
    conf = f"  (conf {confidence:.2f})" if confidence is not None else ""
    print(f"[{index}] {event.get('text', '')}{conf}")


def _wav_chunk_to_pcm(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV blob to mono int16 PCM samples + sample rate."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return raw, rate


def _write_pcm_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())


# ---------------------------------------------------------------------------
# Streaming transcription (microphone or file -> /v1/ws/transcribe)
# ---------------------------------------------------------------------------
async def _send_mic(ws: object, sample_rate: int) -> None:
    sounddevice = _require("sounddevice", "pip install sounddevice")
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue()

    def callback(indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            print(f"\n[mic] {status}", file=sys.stderr)
        loop.call_soon_threadsafe(queue.put_nowait, indata.copy().tobytes())

    with sounddevice.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=sample_rate // 10,  # 100 ms
        callback=callback,
    ):
        print(f"🎙 Recording @ {sample_rate} Hz - speak now, Ctrl+C to stop.\n")
        while True:
            chunk = await queue.get()
            await ws.send(chunk)


async def _send_file(ws: object, path: Path, sample_rate: int) -> None:
    with wave.open(str(path), "rb") as wav:
        if (
            wav.getframerate() != sample_rate
            or wav.getsampwidth() != 2
            or wav.getnchannels() != 1
        ):
            raise SystemExit(
                f"{path}: expected {sample_rate} Hz mono 16-bit PCM WAV "
                "(re-encode e.g. with `speechai` or ffmpeg)"
            )
        while True:
            frames = wav.readframes(CHUNK_FRAMES)
            if not frames:
                break
            await ws.send(frames)
    await ws.send(json.dumps({"action": "stop"}))


async def run_transcribe(args: argparse.Namespace) -> int:
    websockets = _require("websockets", "ships with uvicorn[standard]; or `pip install websockets`")
    url = f"{args.url}/v1/ws/transcribe"
    config: dict = {"sample_rate": args.sample_rate, "language": args.language}
    if args.api_key:
        config["api_key"] = args.api_key

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(config))
        if args.file:
            sender = asyncio.create_task(_send_file(ws, Path(args.file), args.sample_rate))
        else:
            sender = asyncio.create_task(_send_mic(ws, args.sample_rate))
        try:
            async for message in ws:
                _handle_transcribe_event(json.loads(message))
        except KeyboardInterrupt:
            pass
        finally:
            sender.cancel()
            try:
                await sender
            except (asyncio.CancelledError, Exception):
                pass
            try:
                await ws.send(json.dumps({"action": "stop"}))
            except Exception:  # connection may already be gone
                pass
    print("\n✓ stream closed")
    return 0


def _handle_transcribe_event(event: dict) -> None:
    if event.get("type") == "partial":
        _print_partial(event.get("text", ""))
    elif event.get("type") == "final":
        _print_final(event)


# ---------------------------------------------------------------------------
# Streaming synthesis (text -> /v1/ws/synthesize -> live playback)
# ---------------------------------------------------------------------------
async def run_synthesize(args: argparse.Namespace) -> int:
    websockets = _require("websockets", "ships with uvicorn[standard]; or `pip install websockets`")
    url = f"{args.url}/v1/ws/synthesize"
    payload: dict = {"text": args.synthesize, "speed": args.speed}
    if args.api_key:
        payload["api_key"] = args.api_key

    parts: list[np.ndarray] = []
    rate: int | None = None
    chunks = 0
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(payload))
        async for message in ws:
            if isinstance(message, str):
                data = json.loads(message)
                if data.get("type") == "done":
                    print(f"✓ done - {data.get('chunks', chunks)} chunk(s)")
                    break
                continue
            chunks += 1
            samples, chunk_rate = _wav_chunk_to_pcm(message)
            rate = chunk_rate
            parts.append(samples)
            print(f"▶ received sentence {chunks} ({samples.size / chunk_rate:.2f}s)")
            if args.output is None:
                sounddevice = _require("sounddevice", "pip install sounddevice")
                sounddevice.play(samples, samplerate=chunk_rate)
                sounddevice.wait()

    if args.output and rate is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_pcm_wav(out, np.concatenate(parts), rate)
        print(f"💾 saved {out} ({sum(p.size for p in parts) / rate:.2f}s audio)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streaming_demo",
        description="Live streaming demo client for the Bank Speech AI WebSocket API.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"server origin (default {DEFAULT_URL})")
    parser.add_argument("--api-key", default=None, help="API key if the server requires auth")
    parser.add_argument("--language", default=None, help="STT language code (default: server auto-detect)")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="audio sample rate (default 16000)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--transcribe", action="store_true", help="stream speech to /v1/ws/transcribe (default)")
    mode.add_argument("--synthesize", metavar="TEXT", help="synthesize text via /v1/ws/synthesize and play it back")

    parser.add_argument("--file", metavar="WAV", help="stream a 16 kHz mono WAV instead of the microphone")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speed, 0.5-2.0 (with --synthesize)")
    parser.add_argument("--output", default=None, help="save synthesized audio to a WAV file instead of playing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.synthesize:
            return asyncio.run(run_synthesize(args))
        return asyncio.run(run_transcribe(args))
    except KeyboardInterrupt:
        print("\nstopped.")
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
