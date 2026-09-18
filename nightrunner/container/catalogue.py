"""Native resource type catalogue (EResType).

Source: ResourceCore_x64_rwdi.dll +0x876D0, 39 rows of stride 40 (name/short/pretty pointers, u32 type @+24,
u16 memcat @+28, u16 version @+30), dumped by Binary Ninja on 2026-09-10 (survey 01/04). The `version` column is
what every storage record's version field must equal for that type in shipped packs (47/47 packs agree).

`PART_SHAPES` records the logical part sequences observed in the shipped corpus (survey 01 §12 / 04 §1);
they are descriptive, not a validation rule — the writer only enforces 1..15 parts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResType:
    id: int
    version: int
    memcat: int
    name: str
    short: str
    pretty: str
    gpu: bool = False        # GPU-resident predicate true for 0x21, 0x22, 0x31, 0xF0..0xF3
    on_disk: bool = True     # observed as a storage type in at least one shipped pack


_ROWS = [
    # id, version, memcat, name, short, pretty, gpu, on_disk
    (0x00, 1, 0, "_INVALID_", "INVALID", "Invalid", False, False),
    (0x10, 60, 84, "_MESH_", "MESH", "Mesh", False, True),
    (0x11, 60, 84, "_MESH_FIXUPS_", "MESH_FIX", "MeshFixups", False, True),
    (0x12, 13, 85, "_SKIN_", "SKIN", "Skin", False, True),
    (0x18, 3, 83, "_MODEL_", "MODEL", "Model", False, False),
    (0x20, 11, 109, "_TEXTURE_", "TEXTURE", "Texture", False, True),
    (0x21, 11, 116, "_TEXTURE_BITMAP_DATA_", "BITMAP", "TextureBitmapData", True, True),
    (0x22, 11, 116, "_TEXTURE_MIP_BITMAP_DATA_", "STRMBMP", "TextureMipBitmapData", True, False),
    (0x30, 13, 82, "_MATERIAL_", "MATERIAL", "Material", False, False),
    (0x31, 13, 114, "_SHADER_", "SHADER", "Shader", True, False),
    (0x40, 4, 5, "_ANIMATION_", "ANIM", "Animation", False, True),
    (0x41, 4, 5, "_ANIMATION_STREAM_", "ANIMSTRM", "AnimationStream", False, False),
    (0x42, 4, 8, "_ANIMATION_SCR_", "ANIMSCR", "AnimationScr", False, True),
    (0x43, 4, 8, "_ANIMATION_SCRFIXUPS_", "ANIMSFIX", "AnimationScrFixups", False, True),
    (0x44, 2, 5, "_ANM2_METADATA_", "ANM_META", "ANM2Header", False, True),
    (0x45, 2, 5, "_ANM2_PAYLOAD_", "ANM_DATA", "ANM2Payload", False, True),
    (0x46, 2, 5, "_ANM2_FALLBACK_", "ANM_FLBK", "ANM2Fallback", False, False),
    (0x47, 140, 11, "_ANIM_GRAPH_BANK_", "ANMGRAPH", "AnimGraphBank", False, True),
    (0x48, 140, 11, "_ANIM_GRAPH_BANK_FIXUPS_", "AGRPHFIX", "AnimGraphBankFixups", False, True),
    (0x49, 4, 14, "_ANIM_CUSTOM_RESOURCE_", "ACSTMRES", "AnimCustomResource", False, True),
    (0x4A, 4, 14, "_ANIM_CUSTOM_RESOURCE_FIXUPS_", "ACRESFIX", "AnimCustomResourceFixups", False, True),
    (0x51, 2, 178, "_GPUFX_", "GPUFX", "GpuFx", False, False),
    (0x55, 2, 77, "_ENV_BIN_", "ENV_BIN", "EnvprobeBin", False, True),
    (0x56, 2, 77, "_VXL_BIN_", "VXL_BIN", "VoxelizerBin", False, True),
    (0x5A, 2, 122, "_AREA_", "AREA", "Area", False, True),
    (0x60, 2, 148, "_PREFAB_TEXT_", "PRFBTXT", "PrefabText", False, False),
    (0x61, 8, 148, "_PREFAB_", "PREFAB", "Prefab", False, True),
    (0x62, 8, 148, "_PREFAB_DATA_FIXUPS_", "PRFBFXUP", "PrefabFixUps", False, True),
    (0x65, 2, 18, "_SOUND_", "SOUND", "Sound", False, False),
    (0x66, 2, 18, "_SOUND_MUSIC_", "MUSIC", "Music", False, False),
    (0x67, 2, 18, "_SOUND_SPEECH_", "SPEECH", "Speech", False, False),
    (0x68, 2, 18, "_SOUND_STREAM_", "SNDSTRM", "SFX_stream", False, False),
    (0x69, 2, 18, "_SOUND_LOCAL_", "SNDLOCAL", "SFX_local", False, False),
    (0xF0, 5, 115, "_VERTEX_DATA_", "VERTEXES", "VertexData", True, True),
    (0xF1, 4, 115, "_INDEX_DATA_", "INDEXES", "IndexData", True, True),
    (0xF2, 4, 115, "_GEOMETRY_DATA_", "GEOMETRY", "GeometryData", True, False),
    (0xF3, 2, 115, "_CLOTH_DATA_", "CLOTH", "ClothData", True, True),
    (0xF8, 8, 75, "_TINY_OBJECTS_", "TINYOBJS", "TinyObjects", False, False),
    (0xFF, 2, 107, "_BUILDER_INFORMATION_", "BUILDER", "BuilderInformation", False, False),
]

TYPES: dict[int, ResType] = {r[0]: ResType(*r) for r in _ROWS}

# Logical types that appear as *logical* resources on disk (as opposed to part-only types).
LOGICAL_TYPES = {0x10, 0x20, 0x40, 0x42, 0x47, 0x49, 0x55, 0x56, 0x5A, 0x61}

# Observed part-type sequences per logical type (descriptive; survey 01 §12, 04 §1).
PART_SHAPES: dict[int, list[tuple[int, ...]]] = {
    0x10: [(0x10, 0x12, 0x11, 0xF0, 0xF1), (0x10, 0x12, 0x11), (0x10, 0x12, 0x11, 0xF0, 0xF1, 0xF3)],
    0x20: [(0x20, 0x21), (0x20,)],
    0x40: [(0x40,), (0x44, 0x45)],
    0x42: [(0x42, 0x43)],
    0x47: [(0x47, 0x48)],
    0x49: [(0x49, 0x4A, 0x49, 0x4A)],
    0x55: [(0x55,)],
    0x56: [(0x56,)],
    0x5A: [(0x5A,)],
    0x61: [(0x61, 0x62)],
}

# Folder names used by the extractor for each logical type.
FAMILY_DIR = {
    0x10: "mesh", 0x20: "texture", 0x40: "anim", 0x42: "animscr", 0x47: "animgraph", 0x49: "animcustom",
    0x55: "envprobe", 0x56: "voxelizer", 0x5A: "area", 0x61: "prefab",
}


def type_name(type_id: int) -> str:
    t = TYPES.get(type_id)
    return t.pretty if t else f"Unknown_{type_id:02X}"


def type_version(type_id: int) -> int | None:
    t = TYPES.get(type_id)
    return t.version if t else None


def family_dir(type_id: int) -> str:
    return FAMILY_DIR.get(type_id, "other")
