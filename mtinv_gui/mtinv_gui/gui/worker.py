"""Tiny helper: run a blocking callable on a QThread, deliver result to slots."""
from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import QObject, QThread, pyqtSignal


class _Worker(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, fn: Callable):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self.failed.emit(exc)


def run_async(owner: QObject, fn: Callable,
              on_done: Callable, on_error: Callable) -> QThread:
    """Run ``fn()`` off the GUI thread.

    The thread + worker are parented to *owner* so they outlive the call and are
    cleaned up with it.  Returns the started :class:`QThread`.
    """
    thread = QThread(owner)
    worker = _Worker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    def _cleanup():
        thread.quit()
        thread.wait()
        worker.deleteLater()

    worker.done.connect(lambda r: (_cleanup(), on_done(r)))
    worker.failed.connect(lambda e: (_cleanup(), on_error(e)))
    if not hasattr(owner, "_bg_threads"):
        owner._bg_threads = []          # type: ignore[attr-defined]
    owner._bg_threads.append((thread, worker))  # keep refs
    thread.start()
    return thread
