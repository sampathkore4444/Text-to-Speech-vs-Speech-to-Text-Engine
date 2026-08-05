"""Audio layer: IO/resampling/encoding and voice activity detection."""

from speechai.audio.io import (
    ASR_SAMPLE_RATE,
    AudioBuffer,
    generate_sine,
    load_audio,
    pcm16_bytes,
    resample,
    to_asr_audio,
    write_wav,
)
from speechai.audio.vad import (
    EnergyVAD,
    StreamingVAD,
    Utterance,
    WebRTCVAD,
    build_vad,
)

__all__ = [
    "ASR_SAMPLE_RATE",
    "AudioBuffer",
    "EnergyVAD",
    "StreamingVAD",
    "Utterance",
    "WebRTCVAD",
    "build_vad",
    "generate_sine",
    "load_audio",
    "pcm16_bytes",
    "resample",
    "to_asr_audio",
    "write_wav",
]
