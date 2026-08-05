"""Bank Speech AI platform - production-grade STT + TTS for banking.

Public package marker; the platform is split into:
- ``speechai.core``    configuration, logging, metrics, errors
- ``speechai.audio``   audio IO, resampling, VAD
- ``speechai.stt``     speech-to-text engines + streaming transcription
- ``speechai.tts``     text-to-speech engines + text normalization
- ``speechai.redaction`` banking PII redaction
- ``speechai.pipeline`` batch jobs and queuing
- ``speechai.eval``    WER/CER/RTF/latency evaluation harness
- ``speechai.api``     FastAPI REST + WebSocket service
- ``speechai.cli``     command-line tools
"""

__version__ = "0.1.0"
