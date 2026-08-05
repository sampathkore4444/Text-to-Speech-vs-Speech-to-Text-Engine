"""TTS package: engines, text normalization and streaming."""

from speechai.tts.base import SynthesisResult, TTSEngine, TTSOptions, build_tts_engine
from speechai.tts.piper_engine import PiperTTSEngine
from speechai.tts.streaming import StreamingSynthesizer
from speechai.tts.textnorm import NormalizedText, TextNormalizer, split_sentences

__all__ = [
    "NormalizedText",
    "PiperTTSEngine",
    "StreamingSynthesizer",
    "SynthesisResult",
    "TTSEngine",
    "TTSOptions",
    "TextNormalizer",
    "build_tts_engine",
    "split_sentences",
]
