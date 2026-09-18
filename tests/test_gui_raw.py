"""Nightrunner, Raw tab: lazy tree shape, chunked fetching, tri-state checks, export, filter, open_gid.

Runs headless (QT_QPA_PLATFORM=offscreen) on a synthetic pack with 5,000+ resources; skips without PySide6.
"""
from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

import tests.synth as S

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEventLoop, QSettings, Qt
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False

N_TEX = 5000


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class TestRawTab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = S.tmpdir("gui_raw_")
        cls.d = Path(cls._tmp.name)
        cls.big = cls.d / "big_pc.rpack"
        res = [S.single(b" lead_space ", 0x5A, 64, seed=1), S.prefab()]
        res += [S.texture(b"tex_%05d" % i, 24, 40 + i % 7, seed=i % 200) for i in range(N_TEX)]
        res += [S.mesh(b"mesh_a"), S.mesh(b"mesh_b", seed=7)]
        S.write_pack(res, cls.big)
        cls.small = cls.d / "small_pc.rpack"
        S.write_pack([S.texture(b"tex_small"), S.single(b"area_x", 0x5A, 32)], cls.small)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        from nightrunner.gui.context import AppContext
        from nightrunner.gui.tabs.raw import Tab
        settings = QSettings(str(self.d / f"{self._testMethodName}.ini"), QSettings.IniFormat)
        self.ctx = AppContext(settings, game=None, autoload=False)
        self.ctx.catalog.load([self.big, self.small])
        self.assertTrue(self.ctx.catalog.wait())
        self.tab = Tab(self.ctx)
        self.m = self.tab.model
        self.pump(lambda: len(self.m.visible_pack_ids()) == 2)
        self.pid = next(e.id for e in self.ctx.catalog.packs if e.path == self.big)
        self.pk = self.ctx.catalog.packs[self.pid].pack

    def tearDown(self):
        self.tab.shutdown()
        self.ctx.runner.pool.waitForDone(5000)
        self.pump(lambda: True, 0.05)
        self.tab.deleteLater()
        self.ctx.close()

    def pump(self, cond, timeout: float = 20.0) -> None:
        end = time.time() + timeout
        while time.time() < end:
            self.app.processEvents(QEventLoop.AllEvents, 20)
            if cond():
                return
            time.sleep(0.005)
        self.assertTrue(cond(), "timed out waiting for the GUI")

    def filter(self, text: str = "", type_id=None) -> None:
        self.tab.type_combo.blockSignals(True)
        self.tab.type_combo.setCurrentIndex(max(0, self.tab.type_combo.findData(type_id)))
        self.tab.type_combo.blockSignals(False)
        self.tab.search.setText(text)
        self.tab._debounce.flush()
        self.pump(lambda: not self.tab._filter_busy)

    def names(self, parent) -> list[str]:
        return [self.m.data(self.m.index(r, 0, parent)) for r in range(self.m.rowCount(parent))]

    def group_index(self, t: int):
        return self.m.node_index(self.m.group_node(self.pid, t))

    # ---- shape ----------------------------------------------------------------------------------------------------
    def test_tree_shape(self):
        m = self.m
        self.assertEqual(self.names(m.index(0, 0).parent()), ["big_pc.rpack", "small_pc.rpack"])
        pi = m.pack_index(self.pid)
        h = self.pk.header
        self.assertEqual(self.names(pi)[:3], ["Header", f"Storage table ({h.storage_count})",
                                              f"Name table ({N_TEX + 4:,})"])
        kids = [m.item(m.index(r, 0, pi)) for r in range(m.rowCount(pi))]
        self.assertEqual([k["kind"] for k in kids], ["table"] * 3 + ["group"] * 4)
        self.assertEqual([k["type"] for k in kids[3:]], [0x10, 0x20, 0x5A, 0x61])
        self.assertTrue(self.names(pi)[4].startswith("Texture") and self.names(pi)[4].endswith("(5,000)"))
        area = self.group_index(0x5A)
        m.fetchMore(area)
        ri = m.index(0, 0, area)
        self.assertEqual(m.data(ri), "·lead_space·")
        self.assertEqual(m.gid_of(ri), self.ctx.catalog.packs[self.pid].base + 0)
        mesh = self.group_index(0x10)
        m.fetchMore(mesh)
        res = m.index(1, 0, mesh)
        self.assertEqual(m.data(res), "mesh_b")
        self.assertEqual(m.rowCount(res), 5)
        part = m.index(3, 0, res)
        self.assertEqual(m.item(part)["part"], 3)
        self.assertEqual(m.parent(part), res.siblingAtColumn(0))
        self.assertEqual(m.parent(res), mesh)
        self.assertEqual(m.parent(mesh), pi)
        li = N_TEX + 3
        j = self.pk.logicals[li].first_part + 3
        self.assertEqual(m.data(part.siblingAtColumn(3)), f"0x{self.pk.part_offset(j):X}")
        self.assertEqual(m.data(part.siblingAtColumn(2)), str(j))
        self.assertEqual(m.data(res.siblingAtColumn(2)), str(li))
        hdr = m.index(0, 0, pi)
        self.assertEqual(m.data(hdr.siblingAtColumn(4)), "36 B")

    def test_lazy_fetch(self):
        from nightrunner.gui.tabs.raw_model import CHUNK
        m = self.m
        gi = self.group_index(0x20)
        self.assertEqual(m.rowCount(gi), 0)
        self.assertTrue(m.canFetchMore(gi))
        m.fetchMore(gi)
        self.assertEqual(m.rowCount(gi), CHUNK)
        seen = CHUNK
        while m.canFetchMore(gi):
            m.fetchMore(gi)
            self.assertGreater(m.rowCount(gi), seen)
            seen = m.rowCount(gi)
        self.assertEqual(m.rowCount(gi), N_TEX)
        self.assertEqual(m.data(m.index(N_TEX - 1, 0, gi)), "tex_%05d" % (N_TEX - 1))
        self.assertFalse(m.index(N_TEX, 0, gi).isValid())
        # the view fetches by itself when the group is expanded and scrolled
        self.tab.resize(900, 600)
        self.tab.show()
        m2 = self.m
        g = m2.group_node(self.pid, 0x20)
        g.fetched = 0
        m2.beginResetModel()
        m2.endResetModel()
        self.tab.view.expand(m2.pack_index(self.pid))
        self.tab.view.expand(m2.node_index(g))
        self.pump(lambda: g.fetched >= CHUNK)
        sb = self.tab.view.verticalScrollBar()
        for _ in range(200):
            sb.setValue(sb.maximum())
            self.app.processEvents()
            if g.fetched == N_TEX:
                break
        self.assertEqual(g.fetched, N_TEX)

    # ---- checks ---------------------------------------------------------------------------------------------------
    def test_tristate(self):
        m = self.m
        pi = m.pack_index(self.pid)
        gi = self.group_index(0x20)
        mesh = self.group_index(0x10)
        C, P, U = Qt.Checked, Qt.PartiallyChecked, Qt.Unchecked
        self.assertEqual(m.data(pi, Qt.CheckStateRole), U)
        # whole group without fetching a row
        self.assertTrue(m.setData(gi, C, Qt.CheckStateRole))
        self.assertEqual(m.rowCount(gi), 0)
        self.assertEqual(m.check_state(gi), C)
        self.assertEqual(m.check_state(pi), P)
        self.assertEqual(m.checked_totals()[0], 2 * N_TEX)
        m.fetchMore(gi)
        r7 = m.index(7, 0, gi)
        self.assertEqual(m.check_state(r7), C)
        m.setData(m.index(1, 0, r7), U, Qt.CheckStateRole)          # one part off → resource & group partial
        self.assertEqual(m.check_state(r7), P)
        self.assertEqual(m.check_state(m.index(0, 0, r7)), C)
        self.assertEqual(m.check_state(gi), P)
        self.assertEqual(m.checked_totals()[0], 2 * N_TEX - 1)
        m.setData(m.index(1, 0, r7), C, Qt.CheckStateRole)          # back on → resource checked again
        self.assertEqual(m.check_state(r7), C)
        self.assertEqual(m.check_state(gi), C)
        # the whole pack
        m.setData(pi, C, Qt.CheckStateRole)
        self.assertEqual(m.check_state(pi), C)
        self.assertEqual(m.check_state(m.index(0, 0, pi)), C)       # header
        m.fetchMore(mesh)
        self.assertEqual(m.check_state(m.index(1, 0, mesh)), C)
        m.setData(m.index(0, 0, pi), U, Qt.CheckStateRole)          # header off → pack partial
        self.assertEqual(m.check_state(pi), P)
        m.setData(pi, U, Qt.CheckStateRole)
        self.assertEqual(m.check_state(pi), U)
        self.assertEqual(m.checked_totals(), (0, 0))
        # single part from nothing
        res = m.index(0, 0, mesh)
        m.setData(m.index(2, 0, res), C, Qt.CheckStateRole)
        self.assertEqual((m.check_state(res), m.check_state(mesh), m.check_state(pi)), (P, P, P))
        j = self.pk.logicals[N_TEX + 2].first_part + 2
        self.assertEqual(m.checked_totals(), (1, self.pk.physicals[j].size))
        self.tab.btn_none.click()
        self.assertEqual((m.check_state(res), m.check_state(pi)), (U, U))
        self.assertIn("nothing", self.tab.counter.text())

    # ---- export ---------------------------------------------------------------------------------------------------
    def test_export_checked(self):
        from nightrunner.extract import part_filename
        from nightrunner.util.names import resource_dirname
        m = self.m
        pi = m.pack_index(self.pid)
        m.setData(m.index(1, 0, pi), Qt.Checked, Qt.CheckStateRole)      # storage table
        area = self.group_index(0x5A)
        m.setData(area, Qt.Checked, Qt.CheckStateRole)
        gi = self.group_index(0x20)
        m.fetchMore(gi)
        m.setData(m.index(10, 0, gi), Qt.Checked, Qt.CheckStateRole)
        mesh = self.group_index(0x10)
        m.fetchMore(mesh)
        m.setData(m.index(3, 0, m.index(1, 0, mesh)), Qt.Checked, Qt.CheckStateRole)
        out = self.d / f"out_{self._testMethodName}"
        self.tab.start_export(self.m.checked_spec(), out)
        self.assertTrue(self.tab.wait_export())
        self.pump(lambda: self.tab._last_export is not None)
        st = self.tab._last_export
        self.assertEqual((st["written"], st["existing"], st["errors"]), (1 + 1 + 2 + 1, 0, 0), st)
        root = out / "big_pc"
        pk = self.pk
        n = pk.header.storage_count * 20
        self.assertEqual((root / "_tables" / "storage_table.bin").read_bytes(),
                         bytes(pk._data[pk.storage_offset:pk.storage_offset + n]))

        def expect(li, ks=None):
            lg = pk.logicals[li]
            fam = __import__("nightrunner.container.catalogue", fromlist=["x"]).family_dir(lg.type)
            d = root / fam / resource_dirname(li, pk.resource(li).name)
            seen = {}
            for k in range(lg.part_count):
                f = d / part_filename(k, pk.part_type(lg.first_part + k), seen)
                if ks is None or k in ks:
                    self.assertEqual(f.read_bytes(), bytes(pk.read_part(lg.first_part + k)), f)
                else:
                    self.assertFalse(f.exists(), f)

        expect(0)                   # area (leading/trailing-space name)
        expect(12)                  # tex_00010 (logicals: area, prefab, textures…)
        expect(N_TEX + 3, {3})      # one part of mesh_b
        # second run: nothing overwritten
        self.tab.start_export(self.m.checked_spec(), out)
        self.assertTrue(self.tab.wait_export())
        self.pump(lambda: self.tab._last_export is not st)
        self.assertEqual((self.tab._last_export["written"], self.tab._last_export["existing"]), (0, 5))

    def test_export_selection_and_all_specs(self):
        from nightrunner.gui.tabs.raw_model import spec_totals
        m = self.m
        gi = self.group_index(0x61)
        m.fetchMore(gi)
        spec = m.selection_spec([m.index(0, 0, gi), m.index(2, 0, m.pack_index(self.pid))])
        self.assertEqual(len(spec), 1)
        info, label, ch = spec[0]
        self.assertEqual((label, ch.tables, int(ch.mask.sum())), ("big_pc.rpack", {"names"}, 1))
        self.assertTrue(ch.mask[1])
        self.assertEqual(m.checked_totals(), (0, 0))             # selection does not check
        out = self.d / f"out_{self._testMethodName}"
        self.tab.start_export(spec, out)
        self.assertTrue(self.tab.wait_export())
        self.pump(lambda: self.tab._last_export is not None)
        self.assertEqual(self.tab._last_export["written"], 3)
        from nightrunner.gui.tabs.raw import _all_spec_job
        aspec, parts, size, _ = _all_spec_job(self.ctx.catalog, {})
        self.assertEqual(len(aspec), 2)
        pk = self.pk
        exp = len(pk.physicals) + 3 + len(self.ctx.catalog.packs[1 - self.pid].pack.physicals) + 3
        self.assertEqual(parts, exp)
        self.assertEqual((parts, size), spec_totals(aspec))

    # ---- filter / navigation --------------------------------------------------------------------------------------
    def test_filter(self):
        m = self.m
        self.filter("TEX_0123")
        self.assertTrue(m.filtering)
        self.assertEqual(m.visible_pack_ids(), [self.pid])
        pi = m.pack_index(self.pid)
        self.assertEqual(m.rowCount(pi), 1)                      # no table nodes, one group
        gi = m.index(0, 0, pi)
        m.fetchMore(gi)
        self.assertEqual(self.names(gi), ["tex_%05d" % i for i in range(1230, 1240)])
        self.filter("tex 4999")                                  # words AND
        self.assertEqual(self.names(m.index(0, 0, m.pack_index(self.pid))), ["tex_04999"])  # auto-expanded
        g = m.group_node(self.pid, 0x20)
        self.assertEqual(g.rows.tolist(), [2 + 4999])
        self.filter("#1")
        self.assertEqual(sorted((pid, n.type, n.rows.tolist()) for pid in m.visible_pack_ids()
                                for n in m.group_nodes(pid)),
                         sorted([(self.pid, 0x61, [1]), (1 - self.pid, 0x5A, [1])]))
        self.filter("", 0x5A)
        self.assertEqual(sorted(len(n.rows) for n in m.group_nodes()), [1, 1])
        self.filter("nothing_matches_this")
        self.assertEqual(m.visible_pack_ids(), [])
        self.filter("")
        self.assertFalse(m.filtering)
        self.assertEqual(len(m.visible_pack_ids()), 2)
        self.assertEqual(m.rowCount(m.pack_index(self.pid)), 7)

    def test_open_gid(self):
        m = self.m
        self.tab.resize(900, 600)
        self.tab.show()
        self.filter("mesh")
        base = self.ctx.catalog.packs[self.pid].base
        gid = base + 2 + 4500                                    # tex_04500: hidden by the filter
        self.tab.open_gid(gid)
        self.pump(lambda: self.tab._reveal_gid is None)
        self.assertEqual(self.tab.search.text(), "")
        self.assertFalse(m.filtering)
        cur = self.tab.view.currentIndex()
        self.assertEqual(m.gid_of(cur), gid)
        self.assertEqual(m.data(cur.siblingAtColumn(0)), "tex_04500")
        self.assertTrue(self.tab.view.isExpanded(m.node_index(m.group_node(self.pid, 0x20))))
        # visible without a filter change: immediate
        self.tab.open_gid(base + 1)
        self.assertIsNone(self.tab._reveal_gid)
        self.assertEqual(m.gid_of(self.tab.view.currentIndex()), base + 1)

    def test_nav_signals_from_menu_targets(self):
        """Texture / mesh rows know their gid and type (context menu → ctx.openTexture / openMesh)."""
        got = []
        self.ctx.openTexture.connect(got.append)
        m = self.m
        gi = self.group_index(0x20)
        m.fetchMore(gi)
        idx = m.index(3, 0, gi)
        it = m.item(idx)
        self.assertEqual(self.pk.logicals[it["li"]].type, 0x20)
        self.ctx.openTexture.emit(m.gid_of(idx))
        self.assertEqual(got, [self.ctx.catalog.packs[self.pid].base + 5])


if __name__ == "__main__":
    unittest.main()
