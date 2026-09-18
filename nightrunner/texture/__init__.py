"""Texture layer: IMGC header/payload ↔ DDS (DX10) ↔ PNG preview/import.

Modules
-------
formats.py   native IL::Format enum (70 names) with DXGI / DDS FourCC / block-size table and provenance tiers
imgc.py      ImgcHeader (80-byte fixed part + extension), mip-major padded level layout, split/join payload
dds.py       DDS reader (DX10 + legacy FourCC + unambiguous RGB masks) and DX10 writer; face-major ↔ mip-major
png.py       preview decode (Pillow BCn for BC1/2/3/6H/7 — preview only; own numpy BC4/BC5; uncompressed dtypes)
             and PNG → IMGC import (RGBA8 / R8 / RG8_SNORM with box-filtered mip chains)
codec.py     TextureCodec registered for logical type 0x20 (extract / build / roundtrip)
cli.py       `nr texture info|export|census|ddsinfo`

Evidence: notes/survey/03-textures-materials-sdb.md (§1–§6, §10) and notes/FORMATS/imgc.md, dds-mapping.md.
"""
