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

Two backends, tried in that order:

1. **pyvgmstream**, if the user has installed it — a binding that decodes bytes to wav bytes in memory, no
   temporary files and no executable. It is BSD-3-Clause (PyPI `license_expression`, which is where PEP 639 puts
   it now — the older `license` field and the classifiers are both empty, so a check that reads only those
   wrongly concludes it is unlicensed), and it ships licence files for vgmstream and pybind11 alongside its own.
   It is still *not* a dependency of this project and is never installed here: adding one is a deliberate
   decision, not something to slip in behind a convenience.
2. **vgmstream-cli**, found via `$NIGHTRUNNER_VGMSTREAM`, then beside this install, then `PATH`.

Nothing is ever downloaded.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import UnsupportedError

#: Executable names vgmstream ships under, newest first. `test.exe` is its old name and is deliberately **not**
#: searched for on PATH: Git for Windows ships an unrelated `usr/bin/test.exe`, which a PATH search happily finds
#: and which would make playback look available and then fail. It is accepted only when pointed at directly or
#: found in a vgmstream folder beside this install.
EXE_NAMES = ("vgmstream-cli", "vgmstream-cli.exe", "vgmstream.exe", "test.exe")

#: The subset safe to look for on PATH.
PATH_NAMES = ("vgmstream-cli", "vgmstream-cli.exe", "vgmstream.exe")

ENV_VAR = "NIGHTRUNNER_VGMSTREAM"

#: WAVE format tag of every wem in these games. 0xFFFF is "extensible/other"; here it means Wwise Vorbis.
WWISE_VORBIS = 0xFFFF

#: WAVE format tags of the decoder's *output*.
WAVE_PCM = 1
WAVE_FLOAT = 3

#: The optional in-process backend. Not a dependency: `pip install pyvgmstream` is the user's own call.
PY_MODULE = "pyvgmstream"

INSTALL_HINT = ("No Wwise Vorbis decoder found. Either put vgmstream-cli on PATH (or set "
                f"{ENV_VAR} to it), or install the optional {PY_MODULE} package. ffmpeg cannot decode these "
                "files. Nothing is downloaded for you.")


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


def find_decoder() -> Path | None:
    """vgmstream, or None. Env first, then beside this install, then PATH."""
    env = os.environ.get(ENV_VAR)
    if env:
        p = Path(env)
        if p.is_file():
            return p
    here = Path(__file__).resolve().parents[2]
    for name in EXE_NAMES:
        for cand in (here / name, here / "tools" / name, here / "vgmstream" / name):
            if cand.is_file():
                return cand                       # a folder we own, so the generic old name is safe here
    for name in PATH_NAMES:
        hit = shutil.which(name)
        if hit:
            return Path(hit)
    return None


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
    """Whether anything can decode a wem here."""
    return have_pyvgmstream() or find_decoder() is not None


def backend() -> str:
    """Which backend a decode would use, for the UI to report. "none" when there is nothing."""
    if have_pyvgmstream():
        return PY_MODULE
    exe = find_decoder()
    return exe.name if exe is not None else "none"


def decode_to_wav(wem: bytes, out_wav: Path, decoder: Path | None = None, timeout: float = 60.0) -> Path:
    """Decode *wem* to a RIFF/PCM wav at *out_wav*. Raises `UnsupportedError` with the reason on failure.

    The bytes are written to a temporary `.wem` first: vgmstream takes a path, and the member inside a container
    has no file of its own.
    """
    if decoder is None and have_pyvgmstream():
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(decode_in_process(wem))
        return out_wav
    exe = decoder or find_decoder()
    if exe is None:
        raise UnsupportedError(INSTALL_HINT)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        fd, tmp_name = tempfile.mkstemp(suffix=".wem")
        os.close(fd)
        tmp = Path(tmp_name)
        tmp.write_bytes(wem)
        try:
            r = subprocess.run([str(exe), "-o", str(out_wav), str(tmp)], capture_output=True, text=True,
                               timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as exc:
            raise UnsupportedError(f"{exe.name} did not finish within {timeout:g}s") from exc
        except OSError as exc:
            raise UnsupportedError(f"could not run {exe}: {exc}") from exc
        if r.returncode != 0 or not out_wav.is_file():
            detail = (r.stderr or r.stdout or "").strip().splitlines()
            raise UnsupportedError(f"{exe.name} failed: {detail[-1] if detail else f'exit {r.returncode}'}")
        return out_wav
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass
