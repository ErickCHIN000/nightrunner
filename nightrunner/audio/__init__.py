"""Audio: the AESP container, Wwise soundbanks and the `wwisepinhead` registry. Read-only.

`notes/FORMATS/aesp.md` has the measurements and the open holes. There is no writer: injecting audio needs the
`mods/audio/` mount confirmed first (hole E7), and a `.wem` encoder, which this project does not have.
"""
from .aesp import Aesp, Member, audio_dir, expected_id, open_all, wwise_id
from .bnk import Bank, Sound, is_bank
from .pinhead import Pinhead, from_container

__all__ = ["Aesp", "Member", "Bank", "Sound", "Pinhead", "audio_dir", "expected_id", "from_container",
           "is_bank", "open_all", "wwise_id"]
