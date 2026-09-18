"""Decoding a `.wem` to playable PCM, through an external decoder.

Every `.wem` in these games is codec tag **0xFFFF — Wwise Vorbis** (census 2026-09-17: 302/302 sampled `sfx`
members and 321/321 `streams` members). That is not standard Vorbis: the setup packet is stripped and the
codebooks live in the decoder, which is why a normal player cannot open one. ffmpeg cannot either — it reads the
header and reports `Audio: none ([255][255][0][0] / 0xFFFF) ... unknown codec`.

Decoding it properly means rebuilding the Vorbis headers (what `ww2ogg` does) and then decoding Vorbis. Neither
is in the standard library, and this project takes no new Python dependencies, so nothing here decodes audio
itself. Instead it drives **vgmstream**, which handles Wwise Vorbis natively, and refuses with a message naming
what is missing when vgmstream is not installed — the same call already made for BC textures, where the answer
was to point at `texconv` rather than ship an encoder.

Decoding goes through **pyvgmstream**, a dependency of this project (`requirements-gui.txt`). It is
BSD-3-Clause — PyPI carries that in `license_expression`, where PEP 639 puts it; the older `license` field and the
classifiers are both empty, so a check that reads only those wrongly concludes it is unlicensed — and it ships
the licence files for vgmstream and pybind11 alongside its own.

An earlier version also drove a `vgmstream-cli` executable as a fallback. That is gone: with the binding
installed by default nothing reached it, and an unused code path is worse than no code path.
"""
from __future__ import annotations

import importlib
import importlib.util
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import UnsupportedError

#: WAVE format tag of every wem in these games. 0xFFFF is "extensible/other"; here it means Wwise Vorbis.
WWISE_VORBIS = 0xFFFF

#: WAVE format tags of the decoder's *output*.
WAVE_PCM = 1
WAVE_FLOAT = 3

#: The optional in-process backend. Not a dependency: `pip install pyvgmstream` is the user's own call.
PY_MODULE = "pyvgmstream"

INSTALL_HINT = (f"{PY_MODULE} is not installed, so this sound cannot be decoded. It is in "
                "requirements-gui.txt; re-run setup.bat, or pip install it into the venv.")


@dataclass
class WemInfo:
    """What the RIFF header says, without decoding anything."""
    codec: int
    channels: int
    sample_rate: int
    data_bytes: int

    @property
    def is_wwise_vorbis(self) -> bool:
        return self.codec == WWISE_VORBIS

    def to_json(self) -> dict:
        return {"codec": f"0x{self.codec:04X}", "wwise_vorbis": self.is_wwise_vorbis,
                "channels": self.channels, "sample_rate": self.sample_rate, "data_bytes": self.data_bytes}


def read_info(data: bytes) -> WemInfo | None:
    """Parse the RIFF chunks of a `.wem`. None when it is not RIFF at all."""
    if len(data) < 12 or data[:4] != b"RIFF":
        return None
    pos, fmt, data_bytes = 12, None, 0
    while pos + 8 <= len(data):
        tag = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if tag == b"fmt " and fmt is None and pos + 8 + 8 <= len(data):
            fmt = struct.unpack_from("<HHI", data, pos + 8)
        elif tag == b"data":
            data_bytes = size
        pos += 8 + size + (size & 1)
    if fmt is None:
        return None
    return WemInfo(fmt[0], fmt[1], fmt[2], data_bytes)


def to_pcm16_wav(wav: bytes) -> bytes:
    """A wav re-encoded as 16-bit integer PCM, if it is not already.

    pyvgmstream hands back IEEE float32 (`fmt` tag 3, 32-bit), which several playback backends accept and others
    silently refuse - the failure looks like "Play does nothing", which is the worst kind. Anything already
    16-bit PCM, or in a shape this does not recognise, is returned untouched rather than mangled.
    """
    if len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        return wav
    fmt = data = None
    pos = 12
    while pos + 8 <= len(wav):
        tag = wav[pos:pos + 4]
        size = struct.unpack_from("<I", wav, pos + 4)[0]
        if tag == b"fmt " and fmt is None:
            fmt = struct.unpack_from("<HHIIHH", wav, pos + 8)
        elif tag == b"data" and data is None:
            data = (pos + 8, size)
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        return wav
    tag_, channels, rate, _bps, _align, bits = fmt
    if tag_ == WAVE_PCM and bits == 16:
        return wav
    if tag_ != WAVE_FLOAT or bits != 32:
        return wav                                   # an unfamiliar shape is left alone, not guessed at
    off, size = data
    samples = np.frombuffer(wav[off:off + size], dtype="<f4")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    body = pcm.tobytes()
    hdr = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(body), b"WAVE", b"fmt ", 16, WAVE_PCM, channels,
                      rate, rate * channels * 2, channels * 2, 16, b"data", len(body))
    return hdr + body


def have_pyvgmstream() -> bool:
    """Whether the optional in-process backend is importable. Checked without importing it."""
    try:
        return importlib.util.find_spec(PY_MODULE) is not None
    except (ImportError, ValueError):
        return False


def decode_in_process(wem: bytes) -> bytes:
    """wav bytes via pyvgmstream. Raises `UnsupportedError` when it is absent or refuses."""
    if not have_pyvgmstream():
        raise UnsupportedError(INSTALL_HINT)
    try:
        mod = importlib.import_module(PY_MODULE)
        # pyvgmstream 0.1.1: decode_buffer_to_wav_bytes(data, filename_hint=...). The hint is how it picks the
        # format, and these members carry no extension of their own.
        return to_pcm16_wav(bytes(mod.decode_buffer_to_wav_bytes(wem, filename_hint="sound.wem")))
    except UnsupportedError:
        raise
    except Exception as exc:                       # a third-party backend must not take the tab down
        raise UnsupportedError(f"{PY_MODULE} could not decode this sound: {type(exc).__name__}: {exc}") from exc


def available() -> bool:
    """Whether a wem can be decoded here."""
    return have_pyvgmstream()


def decode_to_wav(wem: bytes, out_wav: Path) -> Path:
    """Decode *wem* to a 16-bit PCM wav at *out_wav*. Raises `UnsupportedError` with the reason on failure."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_wav.write_bytes(decode_in_process(wem))
    return out_wav
