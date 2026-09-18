"""Background work for the GUI: a thread pool, result delivery on the GUI thread, stale-result dropping.

    self.runner = TaskRunner(self)
    self.runner.submit("preview", decode_fn, arg, on_done=self._show, on_error=self._fail)

Submitting again under the same key supersedes the previous job: its result is discarded when it arrives, so a
fast-scrolling list never paints an old preview. Callbacks always run on the GUI thread.
"""
from __future__ import annotations

import traceback
from typing import Any, Callable

import shiboken6

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal


class _Bridge(QObject):
    done = Signal(object, int, object)     # key, generation, result
    failed = Signal(object, int, str)


class _Job(QRunnable):
    def __init__(self, bridge: _Bridge, key, gen: int, fn: Callable, args, kwargs):
        super().__init__()
        self.setAutoDelete(True)
        self.bridge, self.key, self.gen, self.fn, self.args, self.kwargs = bridge, key, gen, fn, args, kwargs
        self.cancelled = False

    def run(self):
        if self.cancelled:
            return
        try:
            res = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001
            self.bridge.failed.emit(self.key, self.gen, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            return
        self.bridge.done.emit(self.key, self.gen, res)


class TaskRunner(QObject):
    def __init__(self, parent=None, pool: QThreadPool | None = None):
        super().__init__(parent)
        self.pool = pool or QThreadPool.globalInstance()
        self._bridge = _Bridge(self)
        self._bridge.done.connect(self._on_done)
        self._bridge.failed.connect(self._on_failed)
        self._gen: dict[Any, int] = {}     # key -> generation of the live job (global counter: never reused)
        self._seq = 0
        self._cb: dict[tuple[Any, int], tuple[Callable | None, Callable | None]] = {}
        self._jobs: dict[Any, _Job] = {}

    def submit(self, key, fn: Callable, *args, on_done: Callable | None = None, on_error: Callable | None = None,
               **kwargs) -> int:
        self._seq += 1
        gen = self._seq
        self._gen[key] = gen
        old = self._jobs.get(key)
        if old is not None:
            old.cancelled = True           # not started yet → skipped; running → result dropped
            self._try_take(old)
        self._cb = {k: v for k, v in self._cb.items() if k[0] != key}
        self._cb[(key, gen)] = (on_done, on_error)
        job = _Job(self._bridge, key, gen, fn, args, kwargs)
        self._jobs[key] = job
        self.pool.start(job)
        return gen

    def cancel(self, key) -> None:
        self._gen.pop(key, None)
        job = self._jobs.pop(key, None)
        if job is not None:
            job.cancelled = True
            self._try_take(job)
        self._cb = {k: v for k, v in self._cb.items() if k[0] != key}

    def _try_take(self, job: "_Job") -> None:
        # an auto-deleted QRunnable (already ran) must not be touched: its C++ side is gone
        try:
            if shiboken6.isValid(job):
                self.pool.tryTake(job)
        except RuntimeError:
            pass

    def _take(self, key, gen):
        if self._gen.get(key) != gen:
            return None
        self._jobs.pop(key, None)
        self._gen.pop(key, None)           # keys (e.g. per-thumbnail) must not accumulate
        return self._cb.pop((key, gen), None)

    def _on_done(self, key, gen, res):
        cb = self._take(key, gen)
        if cb and cb[0]:
            cb[0](res)

    def _on_failed(self, key, gen, err):
        cb = self._take(key, gen)
        if cb and cb[1]:
            cb[1](err)


class Debouncer(QObject):
    """Call *fn* once, *ms* after the last trigger()."""

    def __init__(self, fn: Callable, ms: int = 200, parent=None):
        super().__init__(parent)
        self._t = QTimer(self, singleShot=True, interval=ms)
        self._t.timeout.connect(fn)

    def trigger(self, *_):
        self._t.start()

    def flush(self):
        if self._t.isActive():
            self._t.stop()
            self._t.timeout.emit()
