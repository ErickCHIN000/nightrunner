"""Building a `.wem` the engine can play, without Wwise.

Almost everything these games ship is Wwise Vorbis, and producing that needs Audiokinetic's own authoring tool
(UTM-AIO drives `WwiseConsole.exe convert-external-source` for exactly this). That is a licensed dependency no
toolkit can vendor, and it is the reason "import your own audio" normally starts with "install Wwise".

There is a way round it, and the game itself is the evidence. Two sounds in the `fx` bank carry source plugin
`0x00010001` rather than Vorbis's `0x00040001`, and their `.wem` files are **plain 16-bit little-endian PCM** —
confirmed by decoding them (`44100 Hz, 1 ch, 3.243 s, "16-bit Little Endian PCM"`). So this engine's Wwise
runtime carries the PCM codec, and a PCM `.wem` is a shape it already loads. PCM is something `struct` can write.

    src 1045363059   fmt 24 B   hash 16 B   junk 12 B   data 286,030 B
    src  474852861   fmt 24 B   hash 16 B   junk 12 B   data 1,216,968 B

Their `fmt` chunks are byte-identical, which is what makes a template safe to reuse:

    fe ff 01 00  44 ac 00 00  88 58 01 00  02 00  10 00  06 00  00 00 01 41 00 00
    tag 0xFFFE   44100 Hz     88200 B/s    align  16-bit cbSize  <- 6 unknown bytes

The last six bytes are a Wwise extension, not Microsoft's 22-byte `WAVEFORMATEXTENSIBLE` tail. They are copied
verbatim and never interpreted (C). The Vorbis wems hold `02 31` where these hold `01 41` in the same slot, so
they look like a codec id, but nothing here depends on that reading.

**What is not known.** Both shipped PCM wems also carry a 16-byte `hash` chunk, and the two differ, so it is
derived from the contents. The algorithm is not identified and this module does not write one. If the engine
validates it, a wem built here will not load — that is hole E12, and it is the single thing standing between this
and a Wwise-free import path. Nothing here has been confirmed in a running game.

Writing a PCM wem is also only half of a replacement: a Sound object names its codec in the bank, so pointing one
at PCM audio means patching its plugin id too. See `notes/FORMATS/aesp.md`.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..errors import BuildError

#: Source plugin ids as they appear in a HIRC Sound object.
PLUGIN_PCM = 0x00010001
PLUGIN_VORBIS = 0x00040001

WAVE_EXTENSIBLE = 0xFFFE
WAVE_PCM = 1
WAVE_FLOAT = 3

#: The 6-byte Wwise `fmt` extension, taken verbatim from both shipped PCM wems (2026-09-17). Meaning unknown.
PCM_FMT_EXTENSION = bytes.fromhex("000001410000")

#: Shipped PCM wems pad between `hash` and `data` with a zero `junk` chunk this long.
JUNK_SIZE = 12

SUPPORTED_RATES = (44100, 48000)          # every rate seen in the shipped corpus


@dataclass
class WavPcm:
    """16-bit PCM lifted out of a RIFF/WAVE file."""
    channels: int
    sample_rate: int
    frames: bytes                         # interleaved int16 little-endian

    @property
    def duration(self) -> float:
        return len(self.frames) / (self.sample_rate * self.channels * 2) if self.sample_rate and self.channels else 0.0


def _chunks(data: bytes):
    pos = 12
    while pos + 8 <= len(data):
        tag = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if pos + 8 + size > len(data):
            break
        yield tag, pos + 8, size
        pos += 8 + size + (size & 1)


def read_wav(data: bytes) -> WavPcm:
    """A WAV as 16-bit PCM, converting from float32 when needed. Raises `BuildError` on anything else."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise BuildError("not a RIFF/WAVE file")
    fmt = body = None
    for tag, off, size in _chunks(data):
        if tag == b"fmt " and fmt is None:
            fmt = struct.unpack_from("<HHIIHH", data, off)
        elif tag == b"data" and body is None:
            body = data[off:off + size]
    if fmt is None or body is None:
        raise BuildError("WAV has no fmt or data chunk")
    tag_, channels, rate, _bps, _align, bits = fmt
    if channels not in (1, 2):
        raise BuildError(f"{channels} channels: only mono and stereo are supported")
    if tag_ == WAVE_PCM and bits == 16:
        pcm = body
    elif tag_ == WAVE_FLOAT and bits == 32:
        samples = np.frombuffer(body[:len(body) // 4 * 4], dtype="<f4")
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    elif tag_ == WAVE_PCM and bits == 8:
        samples = np.frombuffer(body, dtype=np.uint8).astype("<i2")
        pcm = ((samples - 128) << 8).astype("<i2").tobytes()
    else:
        raise BuildError(f"WAV format tag {tag_} at {bits}-bit is not supported; use 16-bit PCM or 32-bit float")
    return WavPcm(channels, rate, pcm)


def build_pcm_wem(pcm: WavPcm, *, junk: bool = True) -> bytes:
    """A PCM `.wem` carrying *pcm*, shaped like the two the game ships.

    No `hash` chunk is written: the shipped ones differ per file and the algorithm is not identified (E12).
    """
    if pcm.channels not in (1, 2):
        raise BuildError(f"{pcm.channels} channels: only mono and stereo are supported")
    if not pcm.frames:
        raise BuildError("no audio data")
    if len(pcm.frames) % (2 * pcm.channels):
        raise BuildError("PCM data is not a whole number of frames")
    if pcm.sample_rate not in SUPPORTED_RATES:
        raise BuildError(f"{pcm.sample_rate} Hz is outside the rates the shipped corpus uses "
                         f"({', '.join(str(r) for r in SUPPORTED_RATES)}); resample first")
    block = pcm.channels * 2
    fmt = struct.pack("<HHIIHHH", WAVE_EXTENSIBLE, pcm.channels, pcm.sample_rate,
                      pcm.sample_rate * block, block, 16, len(PCM_FMT_EXTENSION)) + PCM_FMT_EXTENSION
    body = bytearray()
    body += b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if junk:
        body += b"junk" + struct.pack("<I", JUNK_SIZE) + bytes(JUNK_SIZE)
    body += b"data" + struct.pack("<I", len(pcm.frames)) + pcm.frames
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + bytes(body)


def wav_to_wem(wav: bytes, *, junk: bool = True) -> bytes:
    """A WAV file's bytes as a PCM `.wem`."""
    return build_pcm_wem(read_wav(wav), junk=junk)


def describe(wem: bytes) -> dict:
    """What a `.wem` is, for reporting. Cheap: reads the chunk table only."""
    out: dict = {"size": len(wem), "chunks": {}, "codec": None}
    if len(wem) < 12 or wem[:4] != b"RIFF":
        return out
    for tag, off, size in _chunks(wem):
        out["chunks"][tag.decode("ascii", "replace").strip()] = size
        if tag == b"fmt ":
            tag_, channels, rate, _bps, _align, bits = struct.unpack_from("<HHIIHH", wem, off)
            out.update({"format_tag": f"0x{tag_:04X}", "channels": channels, "sample_rate": rate, "bits": bits})
            out["codec"] = {WAVE_EXTENSIBLE: "pcm", 0xFFFF: "wwise_vorbis"}.get(tag_, f"0x{tag_:04X}")
    data = out["chunks"].get("data", 0)
    if out.get("codec") == "pcm" and out.get("sample_rate") and out.get("channels"):
        out["duration"] = data / (out["sample_rate"] * out["channels"] * out["bits"] // 8)
    return out
