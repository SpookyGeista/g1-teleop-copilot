"""Isolated inference with a single replaceable pending observation.

The control loop only submits or polls; it never waits for GPU inference.
Original capture timestamps survive process transport unchanged.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading
import time
import numpy as np
from .observations import GeometryObservation


class GraspProcess:
    def __init__(self, python, *, timeout=10., startup_timeout=90., **config):
        self.timeout = timeout
        self.directory = tempfile.TemporaryDirectory(prefix='punktnull-copilot-')
        self.work = Path(self.directory.name)
        self.log = (self.work / 'worker.log').open('w+')
        self.closed = False
        self.counter = 0
        self.buffer = b''
        self.lock = threading.Lock()
        try:
            self.process = subprocess.Popen([str(python), str(Path(__file__).with_name('model_worker.py')),
                '--directory', str(self.work), '--config', json.dumps(config)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0,
                env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
        except BaseException:
            self.closed = True
            self.log.close(); self.directory.cleanup()
            raise
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        os.set_blocking(self.process.stdout.fileno(), False)
        os.set_blocking(self.process.stdin.fileno(), False)
        try:
            ready = self._read(startup_timeout)
            if ready.get('status') != 'ready':
                raise RuntimeError(str(ready.get('error', ready)))
            self.manifest = ready['manifest']
        except BaseException:
            self.close()
            raise

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Grasp inference deadline exceeded')
            if b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                if len(line) > 2 * 1024 * 1024:
                    raise ValueError('Grasp worker response exceeded its message limit')
                result = json.loads(line)
                if not isinstance(result, dict):
                    raise ValueError('Malformed worker response')
                return result
            if len(self.buffer) > 2 * 1024 * 1024:
                raise ValueError('Grasp worker response exceeded its message limit')
            if not self.selector.select(remaining):
                raise TimeoutError('Grasp inference deadline exceeded')
            try:
                data = os.read(self.process.stdout.fileno(), 65536)
            except BlockingIOError:
                continue
            if not data:
                raise RuntimeError('Grasp worker closed before a complete JSON response')
            self.buffer += data

    def _write(self, data, deadline):
        # Requests are currently small, but use the same bounded transport
        # contract as responses so a stalled reader cannot block this thread.
        with selectors.DefaultSelector() as writable:
            writable.register(self.process.stdin, selectors.EVENT_WRITE)
            offset = 0
            while offset < len(data):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not writable.select(remaining):
                    raise TimeoutError('Grasp inference request deadline exceeded')
                try:
                    written = os.write(self.process.stdin.fileno(), data[offset:offset + 65536])
                except BlockingIOError:
                    continue
                if written <= 0:
                    raise RuntimeError('Grasp worker input pipe closed')
                offset += written

    def infer(self, observation, seed=6103):
        with self.lock:
            if self.closed:
                raise RuntimeError('Grasp worker is closed')
            self.counter += 1
            path = self.work / f'{self.counter:08d}.npz'
            np.savez(path, **observation.arrays())
            try:
                deadline = time.monotonic() + self.timeout
                packet = json.dumps({'request_id': self.counter, 'seed': seed}, allow_nan=False).encode() + b'\n'
                self._write(packet, deadline)
                result = self._read(deadline - time.monotonic())
                if result.get('request_id') != self.counter:
                    raise RuntimeError('Late or mismatched grasp inference response')
                if result.get('status') != 'ok':
                    raise RuntimeError(str(result.get('error')))
                return result['result']
            except BaseException:
                self.close()
                raise
            finally:
                path.unlink(missing_ok=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()
        self.directory.cleanup()


class LatestGraspWorker:
    def __init__(self, engine):
        self.engine = engine
        self.condition = threading.Condition()
        self.pending = None
        self.result = None
        self.error = None
        self.closed = False
        self.submitted = self.completed = self.replaced = 0
        self.thread = threading.Thread(target=self._run, name='copilot-grasps', daemon=True)
        self.thread.start()

    def submit(self, observation: GeometryObservation):
        # Copy sensor buffers before returning to the acquisition loop.
        arrays = observation.arrays()
        copied = GeometryObservation(**{key: value.item() if value.ndim == 0 else value
                                       for key, value in arrays.items()})
        with self.condition:
            if self.closed or self.error is not None:
                raise RuntimeError('Grasp worker unavailable')
            if self.pending is not None:
                self.replaced += 1
            self.pending = copied
            self.submitted += 1
            self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                observation, self.pending = self.pending, None
            try:
                result = self.engine.infer(observation)
                with self.condition:
                    if not self.closed:
                        self.result = result
                        self.completed += 1
            except Exception as exc:
                with self.condition:
                    self.error = f'{type(exc).__name__}: {exc}'
                    self.pending = None
                return

    def poll(self):
        with self.condition:
            result, self.result = self.result, None
            return result

    def status(self):
        with self.condition:
            return dict(submitted=self.submitted, completed=self.completed, replaced=self.replaced,
                        pending=self.pending is not None, error=self.error, closed=self.closed)

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = self.result = None
            self.condition.notify_all()
        self.engine.close()
        self.thread.join(timeout=4)
        if self.thread.is_alive():
            raise RuntimeError('Grasp inference thread did not stop')
