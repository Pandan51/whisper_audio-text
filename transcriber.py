"""
transcriber.py - turn an audio file into text with Whisper.

One input (a path to an audio file), one output (a TranscriptResult holding
clean text plus timed segments). No web framework, no API: this is the core
that a service, a queue worker or a command line can call.

Two models are supported and chosen per recording:

    turbo        Stock Whisper large-v3-turbo. Multilingual, produces
                 timestamps. Used for everything that is not Czech.
    czechspeech  moravianai/CzechSpeech, a Czech fine-tune of the same model.
                 Much more accurate on Czech, but Czech only and trained
                 WITHOUT timestamps (non-commercial licence, CC BY-NC 4.0).

With model="auto" (the default) the language is detected first and Czech
audio is routed to czechspeech, everything else to turbo.


SETUP
-----
1.  pip install faster-whisper "av<17"
    (The pin matters. faster-whisper 1.2.1 only asks for av>=11, so pip
    installs the newest PyAV, and decoding then fails with "unexpected
    keyword argument 'metadata_errors'". Tested working: faster-whisper
    1.2.1 with av 16.1.0. Put both versions in requirements.txt.)

2.  turbo downloads itself on first use (about 1.6 GB, cached afterwards).

3.  czechspeech has to be converted once to the CTranslate2 format:

        pip install transformers torch
        ct2-transformers-converter --model moravianai/CzechSpeech \
            --output_dir czechspeech-ct2 \
            --copy_files tokenizer.json preprocessor_config.json \
            --quantization int8

    Point CZECHSPEECH_MODEL_DIR at the resulting folder if it is not in the
    working directory. transformers and torch are only needed for this
    conversion, not for running the service.


USAGE
-----
From Python:

    from transcriber import Transcriber

    t = Transcriber()                       # models load lazily, once
    result = t.transcribe("lecture.m4a")    # auto-picks the model
    print(result.text)
    result.save("lecture.txt")              # or .json for text + segments

From the command line:

    python transcriber.py lecture.m4a
    python transcriber.py lecture.m4a --model turbo --language en
    python transcriber.py lecture.m4a --out lecture.json


HOW IT WORKS
------------
1.  Decode   The file (m4a, mp3, wav, ...) is decoded to 16 kHz mono samples:
             a NumPy array with 16 000 floats per second of audio.
2.  Route    If no language is given, turbo listens to the first stretches of
             speech and names the language. Czech goes to czechspeech.
3.  Transcribe, with the strategy that fits the model:
             - "sequential" (turbo): a 30 s window slides through the audio.
               The model's own timestamps say where the last full phrase
               ended, and the next window starts there.
             - "batched" (czechspeech): a voice-activity detector cuts the
               audio at pauses into pieces of up to 30 s, and the pieces are
               transcribed independently, several at once. No timestamps are
               needed from the model, which is why this model uses it.
4.  Assemble Segments are cleaned and joined into paragraphs (a new paragraph
             starts after a long pause).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio

log = logging.getLogger("transcriber")

SAMPLE_RATE = 16_000  # Whisper was trained on 16 kHz audio


# --------------------------------------------------------------------------
# Errors: one base class so callers can catch everything from this module.
# --------------------------------------------------------------------------

class TranscriptionError(Exception):
    """Base class for every error this module raises on purpose."""


class AudioDecodeError(TranscriptionError):
    """The file is missing, unreadable, or contains no audio."""


class ModelNotAvailableError(TranscriptionError):
    """A model could not be loaded (not converted, not downloaded, ...)."""


# --------------------------------------------------------------------------
# Model registry: everything that differs between the two models lives here.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    key: str                     # short name used in code and on the CLI
    source: str                  # faster-whisper model name or local folder
    languages: Optional[tuple]   # languages it handles; None means any
    strategy: str                # "sequential" or "batched" (see module doc)
    is_local: bool               # True: `source` must be an existing folder


MODELS = {
    "turbo": ModelSpec(
        key="turbo",
        source=os.environ.get("TURBO_MODEL", "large-v3-turbo"),
        languages=None,
        strategy="sequential",
        is_local=False,
    ),
    "czechspeech": ModelSpec(
        key="czechspeech",
        source=os.environ.get("CZECHSPEECH_MODEL_DIR", "czechspeech-ct2"),
        languages=("cs",),
        strategy="batched",
        is_local=True,
    ),
}

# Which model "auto" picks for a language; anything not listed uses turbo.
LANGUAGE_ROUTES = {"cs": "czechspeech"}
DEFAULT_MODEL = "turbo"


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------

@dataclass
class Segment:
    start: float  # seconds from the beginning of the recording
    end: float
    text: str


@dataclass
class TranscriptResult:
    text: str                 # the full transcript, in paragraphs
    segments: list            # list[Segment], in order
    language: str             # e.g. "cs"
    model: str                # registry key of the model that was used
    audio_seconds: float      # length of the recording
    elapsed_seconds: float    # how long transcription took
    language_probability: Optional[float] = None  # None if language was given
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> Path:
        """Write .json (everything) or any other suffix (plain text)."""
        path = Path(path)
        if path.suffix.lower() == ".json":
            path.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            path.write_text(self.text + "\n", encoding="utf-8")
        return path


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_audio(path) -> np.ndarray:
    """Decode any audio file into 16 kHz mono float32 samples."""
    path = Path(path)
    if not path.is_file():
        raise AudioDecodeError(f"File not found: {path}")
    try:
        audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    except Exception as exc:  # PyAV raises many different error types
        raise AudioDecodeError(f"Could not decode {path.name}: {exc}") from exc
    if len(audio) == 0:
        raise AudioDecodeError(f"{path.name} contains no audio")
    return audio


def build_text(segments, paragraph_gap: float = 2.0) -> str:
    """Join segments into paragraphs.

    Consecutive segments go on one line. A silence of `paragraph_gap` seconds
    or more between two segments starts a new paragraph, which keeps the text
    readable and gives later steps (summaries, quizzes) natural break points.
    """
    paragraphs, current, previous_end = [], [], None
    for seg in segments:
        if previous_end is not None and seg.start - previous_end >= paragraph_gap:
            if current:
                paragraphs.append(" ".join(current))
                current = []
        current.append(seg.text)
        previous_end = seg.end
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _is_noise(text: str) -> bool:
    """True for empty output or the '...' Whisper emits on non-speech."""
    return not text.strip(" .…-")


# --------------------------------------------------------------------------
# The transcriber
# --------------------------------------------------------------------------

class Transcriber:
    """Loads the models once and transcribes files with them.

    Create ONE instance and reuse it: loading a model takes seconds and about
    1.5 GB of RAM, transcribing with an already loaded model does not pay that
    again. Not safe to call from several threads at once; run one job at a
    time per instance.

    device / compute_type
        "cpu" + "int8" is the CPU setting. With an NVIDIA GPU use
        "cuda" + "float16".
    cpu_threads
        0 lets the engine decide (it uses the physical cores).
    batch_size
        How many chunks the batched strategy transcribes at once. Higher is
        faster and uses more RAM.
    """

    def __init__(
        self,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        batch_size: int = 4,
        beam_size: int = 5,
    ):
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.batch_size = batch_size
        self.beam_size = beam_size
        self._loaded: dict = {}  # registry key -> WhisperModel

    @classmethod
    def from_env(cls, **overrides) -> "Transcriber":
        """Settings from env variables; any override that is not None wins."""
        settings = {
            "cpu_threads": int(os.environ.get("WHISPER_CPU_THREADS", "0")),  # 0 = auto
            "batch_size": int(os.environ.get("WHISPER_BATCH_SIZE", "4")),
            "beam_size": int(os.environ.get("WHISPER_BEAM_SIZE", "5")),
        }
        settings.update({k: v for k, v in overrides.items() if v is not None})
        log.info("Config: %s", settings)
        return cls(**settings)

    # -- model loading ------------------------------------------------------

    def is_available(self, key: str) -> bool:
        """Can this model be loaded without downloading or converting?"""
        spec = MODELS[key]
        if key in self._loaded:
            return True
        return Path(spec.source).is_dir() if spec.is_local else True

    def _load(self, key: str) -> WhisperModel:
        if key in self._loaded:
            return self._loaded[key]
        spec = MODELS[key]
        if spec.is_local and not Path(spec.source).is_dir():
            raise ModelNotAvailableError(
                f"Model '{key}' not found at '{spec.source}'. Convert it first "
                f"(see SETUP at the top of transcriber.py) or set "
                f"CZECHSPEECH_MODEL_DIR."
            )
        log.info("Loading model '%s' from %s", key, spec.source)
        try:
            model = WhisperModel(
                spec.source,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        except Exception as exc:
            raise ModelNotAvailableError(f"Could not load '{key}': {exc}") from exc
        self._loaded[key] = model
        return model

    # -- language detection -------------------------------------------------

    def detect_language(self, audio: np.ndarray):
        """Return (language code, probability) using the multilingual model.

        The VAD filter skips leading silence, and three 30 s windows are
        checked instead of one, so a quiet or noisy start does not decide
        the result.
        """
        model = self._load(DEFAULT_MODEL)
        language, probability, _ = model.detect_language(
            audio, vad_filter=True, language_detection_segments=3
        )
        return language, float(probability)

    # -- the main entry point -----------------------------------------------

    def transcribe(
        self,
        path,
        model: str = "auto",
        language: Optional[str] = None,
        on_segment: Optional[Callable[[Segment], None]] = None,
    ) -> TranscriptResult:
        """Transcribe one audio file.

        path        Audio file in any format ffmpeg understands.
        model       "auto", "turbo" or "czechspeech".
        language    Two-letter code such as "cs" or "en". Leave None to
                    detect it. Giving it skips detection and is more reliable
                    when you already know the answer.
        on_segment  Optional callback, called with each Segment as soon as it
                    is ready. Use it for progress reporting or live output.

        Raises AudioDecodeError, ModelNotAvailableError or ValueError.
        """
        if model != "auto" and model not in MODELS:
            raise ValueError(f"Unknown model '{model}'. Use auto, {', '.join(MODELS)}.")

        started = time.time()
        warnings: list = []
        audio = load_audio(path)
        audio_seconds = len(audio) / SAMPLE_RATE

        # 1. Decide language and model.
        probability = None
        if model == "czechspeech" and language is None:
            language = "cs"  # the model knows nothing else
        if language is None:
            language, probability = self.detect_language(audio)
            log.info("Detected language %s (%.2f)", language, probability)

        if model == "auto":
            key = LANGUAGE_ROUTES.get(language, DEFAULT_MODEL)
            if not self.is_available(key):
                warnings.append(
                    f"Model '{key}' is not installed; used '{DEFAULT_MODEL}' instead."
                )
                key = DEFAULT_MODEL
        else:
            key = model

        spec = MODELS[key]
        if spec.languages is not None and language not in spec.languages:
            raise ValueError(
                f"Model '{key}' only supports {', '.join(spec.languages)}, "
                f"but the language is '{language}'."
            )
        for message in warnings:
            log.warning(message)

        # 2. Transcribe with the strategy the model needs.
        whisper = self._load(key)
        if spec.strategy == "batched":
            raw_segments, _ = BatchedInferencePipeline(model=whisper).transcribe(
                audio,
                language=language,
                batch_size=self.batch_size,
                beam_size=self.beam_size,
            )
        else:
            raw_segments, _ = whisper.transcribe(
                audio,
                language=language,
                vad_filter=True,                   # skip silence and noise
                condition_on_previous_text=False,  # one bad window cannot spoil the next
                beam_size=self.beam_size,
            )

        # 3. Collect. `raw_segments` is lazy: the model runs inside this loop.
        segments = []
        for raw in raw_segments:
            if _is_noise(raw.text):
                continue
            segment = Segment(round(raw.start, 2), round(raw.end, 2), raw.text.strip())
            segments.append(segment)
            if on_segment is not None:
                on_segment(segment)

        if not segments:
            warnings.append("No speech was recognised in this recording.")

        return TranscriptResult(
            text=build_text(segments),
            segments=segments,
            language=language,
            model=key,
            audio_seconds=round(audio_seconds, 1),
            elapsed_seconds=round(time.time() - started, 1),
            language_probability=probability,
            warnings=warnings,
        )


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file to text.")
    parser.add_argument("audio", help="path to the audio file")
    parser.add_argument("--model", default="auto", choices=["auto", *MODELS])
    parser.add_argument("--language", default=None, help='e.g. "cs" or "en"; detected if omitted')
    parser.add_argument("--out", default=None, help="output file (.txt or .json); default: <audio>.txt")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--quiet", action="store_true", help="do not print segments while working")
    parser.add_argument("--threads", type=int, default=None, help="0 = auto")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def show(segment: Segment) -> None:
        print(f"[{segment.start:7.1f} -> {segment.end:7.1f}] {segment.text}", flush=True)

    try:
        transcriber = Transcriber.from_env(
            cpu_threads=args.threads,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
        )
        result = transcriber.transcribe(
            args.audio,
            model=args.model,
            language=args.language,
            on_segment=None if args.quiet else show,
        )
    except (TranscriptionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = result.save(args.out or Path(args.audio).with_suffix(".txt"))
    print(
        f"\n{result.model} / {result.language}: {result.audio_seconds / 60:.1f} min of audio "
        f"in {result.elapsed_seconds / 60:.1f} min -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
