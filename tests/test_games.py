"""Game profiles (nightrunner/games.py): detection on fake `ph` / `ph_ft` trees, install lookup, project game id,
next free PAK per profile, the project/game mismatch warning and `nr project --game`."""
from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import tests.synth as S
from nightrunner import games as G
from nightrunner.project import GameEnv, Project, next_free_pak, output_names, validate


def fake_install(root: Path, data: str, exe: str | None = None, extra=(), paks=("data0.pak", "data1.pak")) -> Path:
    assets = root.joinpath(data, *G.ASSETS_SUB)
    assets.mkdir(parents=True)
    (assets / "runtime_dx11.sdb").write_bytes(b"")
    src = root / data / "source"
    src.mkdir(parents=True)
    for n in paks:
        (src / n).write_bytes(b"")
    if exe:
        b = root.joinpath(data, *G.EXE_SUB)
        b.mkdir(parents=True)
        (b / exe).write_bytes(b"")
    for e in extra:
        (root / e).mkdir(parents=True, exist_ok=True)
    return root


def dltb(root: Path) -> Path:
    fake_install(root, "ph_ft", "DyingLightGame_TheBeast_x64_rwdi.exe")
    (root / "ph").mkdir()                              # the real install has a stub ph/ with only fs.ini
    (root / "ph" / "fs.ini").write_text("import: ../games/fs.ini\n")
    return root


def dl2(root: Path, data: str = "ph") -> Path:
    return fake_install(root, data, "DyingLightGame_x64_rwdi.exe", extra=("DevTools", f"{data}/dlc_opera"),
                        paks=("data0.pak", "data1.pak", "data_devtools0.pak"))


class DetectTests(unittest.TestCase):
    def setUp(self):
        self._tmp = S.tmpdir("games_")
        self.d = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_dltb(self):
        r = dltb(self.d / "Dying Light The Beast")
        self.assertEqual(G.detect_profile(r).id, "dltb")
        gi = G.GameInstall(r)
        self.assertEqual((gi.id, gi.data_name), ("dltb", "ph_ft"))
        self.assertTrue(gi.valid())
        self.assertEqual(gi.assets, r / "ph_ft/work/data_platform/pc/assets")
        self.assertEqual(gi.sdb("dx12").name, "runtime_dx12.sdb")
        self.assertEqual([p.name for p in gi.paks()], ["data0.pak", "data1.pak"])
        self.assertIsNotNone(gi.exe())

    def test_dl2_ph(self):
        r = dl2(self.d / "Dying Light 2")
        gi = G.GameInstall(r)
        self.assertEqual((gi.id, gi.data_name, gi.name), ("dl2", "ph", "Dying Light 2"))
        self.assertEqual(gi.source, r / "ph" / "source")
        self.assertEqual([p.name for p in gi.paks()], ["data0.pak", "data1.pak"])   # data_devtools0 is not dataN

    def test_dl2_renamed_ph_ft(self):
        r = dl2(self.d / "Dying Light 2", "ph_ft")
        gi = G.GameInstall(r)
        self.assertEqual((gi.id, gi.data_name), ("dl2", "ph_ft"))

    def test_dl2_by_markers_only(self):
        r = fake_install(self.d / "somewhere", "ph", extra=("DevTools",))
        self.assertEqual(G.detect_profile(r).id, "dl2")
        r2 = fake_install(self.d / "other", "ph")          # bare ph: only DL2 uses it
        self.assertEqual(G.detect_profile(r2).id, "dl2")

    def test_bare_ph_ft_defaults_to_dltb(self):
        r = fake_install(self.d / "x", "ph_ft")
        self.assertEqual(G.detect_profile(r).id, "dltb")

    def test_data_folder_picked(self):
        r = dl2(self.d / "Dying Light 2")
        gi = G.GameInstall(r / "ph")
        self.assertEqual((gi.root, gi.id), (r, "dl2"))

    def test_empty_and_explicit(self):
        self.assertIsNone(G.detect_profile(self.d))
        gi = G.GameInstall(self.d / "new")                # no data yet: DLTB layout (tests create it afterwards)
        self.assertEqual((gi.id, gi.data_name), ("dltb", "ph_ft"))
        self.assertFalse(gi.valid())
        self.assertEqual(G.GameInstall(self.d / "new", "dl2").data_name, "ph")
        self.assertEqual(G.GameInstall(self.d, "dl2"), G.GameInstall(self.d, "DL2"))
        with self.assertRaises(KeyError):
            G.profile("dl3")

    def test_find(self):
        lib = self.d / "lib"
        common = lib / "steamapps" / "common"
        b = dltb(common / "Dying Light The Beast")
        t = dl2(common / "Dying Light 2")
        with mock.patch.object(G, "_libraries", lambda: [lib]), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NIGHTRUNNER_GAME_ROOT", None)
            os.environ.pop("BEASTPACK_GAME_ROOT", None)
            self.assertEqual(G.find_game().root, b)
            self.assertEqual(G.find_game(game="dl2").root, t)
            self.assertEqual(G.find_game(t).id, "dl2")             # explicit root: its own profile
            self.assertEqual(G.find_game(t, "dltb").root, b)       # explicit root of the wrong game: skipped
            found = G.find_installs()
            self.assertEqual({k: v.root for k, v in found.items()}, {"dltb": b, "dl2": t})
            os.environ["BEASTPACK_GAME_ROOT"] = str(t)
            self.assertEqual(G.find_game().id, "dl2")
            other = dl2(self.d / "elsewhere")
            self.assertEqual(G.find_installs({"dl2": str(other)})["dl2"].root, other)


class ProjectGameTests(unittest.TestCase):
    def setUp(self):
        self._tmp = S.tmpdir("games_p_")
        self.d = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_and_legacy(self):
        p = Project(name="a", game="dl2")
        f = p.save(self.d / "a.nrproj")
        self.assertEqual(json.loads(f.read_text())["game"], "dl2")
        self.assertEqual(Project.load(f).game, "dl2")
        d = json.loads(f.read_text())
        del d["game"]
        f.write_text(json.dumps(d))
        self.assertEqual(Project.load(f).game, "dltb")
        self.assertEqual(Project().game, "dltb")

    def test_next_free_pak_per_profile(self):
        src = self.d / "src"
        src.mkdir()
        self.assertEqual(next_free_pak(src), "data0.pak")                  # no profile: files only
        self.assertEqual(next_free_pak(src, profile="dltb"), "data2.pak")
        self.assertEqual(next_free_pak(src, profile=G.DL2), "data2.pak")
        (src / "data2.pak").write_bytes(b"")
        self.assertEqual(next_free_pak(src, profile="dl2"), "data3.pak")
        r = dl2(self.d / "Dying Light 2")
        env = GameEnv.from_game(G.GameInstall(r))
        self.assertEqual(env.profile.id, "dl2")
        self.assertEqual(env.data_name, "ph")
        self.assertEqual(output_names(Project(), env), ("assets_2_pc.rpack", "data2.pak"))

    def test_mismatch_warning_and_stock_names(self):
        r = dl2(self.d / "Dying Light 2")
        env = GameEnv.from_game(G.GameInstall(r))
        msgs = [p.message for p in validate(Project(game="dltb", pak_name="data1.pak"), env)]
        self.assertTrue(any("current game is Dying Light 2" in m for m in msgs), msgs)
        self.assertTrue(any("game archive name" in m for m in msgs), msgs)
        msgs = [p.message for p in validate(Project(game="dl2"), env)]
        self.assertFalse(any("current game" in m for m in msgs), msgs)

    def test_cli_game(self):
        from nightrunner import project_cli as C
        lib = self.d / "lib"
        common = lib / "steamapps" / "common"
        dltb(common / "Dying Light The Beast")
        t = dl2(common / "Dying Light 2")
        self.assertEqual(C._split_game("DL2", None), ("dl2", None))
        self.assertEqual(C._split_game(str(t), None), (None, str(t)))
        with mock.patch.object(G, "_libraries", lambda: [lib]), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NIGHTRUNNER_GAME_ROOT", None)
            os.environ.pop("BEASTPACK_GAME_ROOT", None)
            self.assertEqual(C._env("dl2").source, t / "ph" / "source")
            self.assertEqual(C._env(None, str(t / "ph")).profile.id, "dl2")
            f = self.d / "p.nrproj"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(C.run(["new", str(f), "--game", "dl2"]), 0)
            self.assertEqual(Project.load(f).game, "dl2")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(C.run(["info", str(f)]), 0)       # project's game -> DL2 install
            self.assertIn("data2.pak", out.getvalue())
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                C.run(["--game", "dltb", "info", str(f)])
            self.assertIn("project is for dl2", err.getvalue())
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                C.run(["info", str(f), "--game", "dltb"])              # flag after the command name
            self.assertIn("project is for dl2", err.getvalue())


if __name__ == "__main__":
    unittest.main()
