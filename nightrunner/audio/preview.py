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

vgmstream is found by, in order: `$NIGHTRUNNER_VGMSTREAM`, a `vgmstream-cli` / `test.exe` next to this install,
then `PATH`. Nothing is ever downloaded.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

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

INSTALL_HINT = ("No vgmstream found. Put vgmstream-cli on PATH or set "
                f"{ENV_VAR} to it — it is the only decoder that reads Wwise Vorbis "
                "(ffmpeg cannot). Nothing is downloaded for you.")


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


def available() -> bool:
    return find_decoder() is not None


def decode_to_wav(wem: bytes, out_wav: Path, decoder: Path | None = None, timeout: float = 60.0) -> Path:
    """Decode *wem* to a RIFF/PCM wav at *out_wav*. Raises `UnsupportedError` with the reason on failure.

    The bytes are written to a temporary `.wem` first: vgmstream takes a path, and the member inside a container
    has no file of its own.
    """
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
