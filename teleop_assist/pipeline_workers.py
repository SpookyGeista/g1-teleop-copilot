"""Small bounded workers for immutable, already captured sensor bundles.

There are no simulator, camera, model, or filesystem dependencies here. Bundles
are passed by reference, not copied: callers must finish capturing them before
submit() and must not mutate them afterwards. A callback may retain its own
references; that memory is outside the worker's queue bound.

check() and close() raise sticky PipelineWorkerError on callback failure. Work
accepted before failure is still drained in order; subsequent submit() calls
raise instead of pretending the pipeline is healthy. A failed callback is not
counted as processed. Failure diagnostics retain text rather than traceback
frames, which could otherwise retain entire RGB-D bundles.

close() stops submissions and joins after draining. It cannot cancel a callback
that is blocked in external code; such callbacks must provide their own bounds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral
import threading
import traceback
from typing import Callable, Generic, TypeVar


Bundle = TypeVar('Bundle')


@dataclass(frozen=True)
class WorkerFailure:
    """First callback failure, without strong references to failed payloads."""

    exception_type: str
    message: str
    traceback: str


class PipelineWorkerError(RuntimeError):
    """A callback failed; failure contains its original type and traceback text."""

    def __init__(self, worker_name: str, failure: WorkerFailure):
        self.worker_name = worker_name
        self.failure = failure
        super().__init__(f'{worker_name} callback failed: {failure.exception_type}: {failure.message}')


def _describe_failure(error: BaseException) -> WorkerFailure:
    # An external exception with a broken __str__ must not make the worker die
    # while formatting the original failure and thereby hide both failures.
    exception_type = f'{type(error).__module__}.{type(error).__qualname__}'
    try:
        message = str(error)[:4096]
    except BaseException:
        message = '<exception message could not be formatted>'
    try:
        formatted = ''.join(traceback.format_exception(type(error), error, error.__traceback__))[-32768:]
    except BaseException:
        formatted = f'{exception_type}: {message}\n<traceback could not be formatted>'
    return WorkerFailure(exception_type, message, formatted)


class _BoundedWorker(Generic[Bundle]):
    def __init__(self, callback: Callable[[Bundle], object], *, capacity: int,
                 replace_pending: bool, name: str):
        if not callable(callback):
            raise TypeError('callback must be callable')
        if isinstance(capacity, bool) or not isinstance(capacity, Integral):
            raise TypeError('capacity must be a positive integer')
        if capacity < 1:
            raise ValueError('capacity must be a positive integer')
        self._callback = callback
        self._capacity = int(capacity)
        self._replace_pending = replace_pending
        self._name = name
        self._condition = threading.Condition()
        self._pending: deque[Bundle] = deque()
        self._running = False
        self._closing = False
        self._closed = False
        self._failure: WorkerFailure | None = None
        self._captured = self._processed = self._dropped = self._failed = 0
        self._maximum_pending = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _raise_if_failed(self) -> None:
        # Caller holds _condition. Each check raises a fresh exception so its
        # calling stack is not accumulated in the stored worker state.
        if self._failure is not None:
            raise PipelineWorkerError(self._name, self._failure) from None

    def submit(self, bundle: Bundle) -> bool:
        """Offer a bundle without waiting for callback/queue capacity.

        False means an archive queue was full; its dropped counter increments.
        LatestBundleWorker always accepts healthy/open submissions, discarding
        the previous pending bundle when necessary. Closed workers raise.
        """
        with self._condition:
            self._raise_if_failed()
            if self._closing:
                raise RuntimeError(f'{self._name} is closing or closed')
            self._captured += 1
            if len(self._pending) == self._capacity:
                self._dropped += 1
                if not self._replace_pending:
                    return False
                self._pending.popleft()
            self._pending.append(bundle)
            self._maximum_pending = max(self._maximum_pending, len(self._pending))
            self._condition.notify()
            return True

    def check(self) -> None:
        """Raise the first callback failure, including after shutdown."""
        with self._condition:
            self._raise_if_failed()

    def statistics(self) -> dict[str, int | bool]:
        """Atomic counts; capacity excludes the one running callback.

        captured counts healthy/open submit offers, including archive rejection.
        processed counts successful callbacks, failed counts callback failures.
        Always: captured = processed + failed + dropped + pending + running.
        closing means submissions stopped; closed means the worker exited.
        """
        with self._condition:
            return {
                'captured': self._captured, 'processed': self._processed,
                'dropped': self._dropped, 'failed': self._failed,
                'running': self._running, 'pending': len(self._pending),
                'capacity': self._capacity, 'maximum_pending': self._maximum_pending,
                'closing': self._closing, 'closed': self._closed,
            }

    def close(self) -> None:
        """Drain all accepted pending work and join; safe to call repeatedly.

        Callback failures are raised only after the queue drains and thread
        exits. close() from this worker's callback is invalid (self-join).
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError('a worker callback cannot close its own worker')
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join()
        self.check()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending or self._closing)
                    if not self._pending:
                        return
                    bundle = self._pending.popleft()
                    self._running = True
                failure = None
                try:
                    self._callback(bundle)
                except BaseException as error:
                    # Even SystemExit from external callback code must not
                    # silently kill this background thread. Do not retain the
                    # exception/traceback object and its bundle references.
                    failure = _describe_failure(error)
                finally:
                    # Release the input before advertising callback completion.
                    del bundle
                with self._condition:
                    if failure is None:
                        self._processed += 1
                    else:
                        self._failed += 1
                        if self._failure is None:
                            self._failure = failure
                    self._running = False
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()


class LatestBundleWorker(_BoundedWorker[Bundle]):
    """Exactly one callback at a time and at most one replaceable pending item."""

    def __init__(self, callback: Callable[[Bundle], object]):
        super().__init__(callback, capacity=1, replace_pending=True, name='latest-bundle')


class BoundedArchiveWorker(_BoundedWorker[Bundle]):
    """Single callback with a bounded FIFO of pending items (default four).

    submit() returns False immediately when the pending FIFO is full. A running
    callback does not occupy a FIFO slot, so total worker-held bundles <= C+1.
    """

    def __init__(self, callback: Callable[[Bundle], object], capacity: int = 4):
        super().__init__(callback, capacity=capacity, replace_pending=False, name='bounded-archive')
