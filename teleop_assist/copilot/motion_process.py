"""Bounded, isolated CPU motion validation; never opens a robot transport."""
from __future__ import annotations
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time


class MotionProcess:
    def __init__(self, python, repository, *, startup_timeout=30., timeout=1., rejection_directory=None,
                 command_format='float64', mode='guarded'):
        if mode not in ('guarded', 'unitree_manual_sim'):
            raise ValueError('Unknown motion worker mode')
        if command_format not in ('float64', 'unitree_float32'):
            raise ValueError('Unsupported command format')
        self.command_format = command_format
        self.directory = tempfile.TemporaryDirectory(prefix='punktnull-motion-guard-')
        self.log = (Path(self.directory.name)/'worker.log').open('w+')
        self.closed = False
        self.timeout = timeout
        self.sequence = 0
        self.last_call_timing = None
        self.last_call_request_bytes = None
        self.rejection_directory = Path(rejection_directory) if rejection_directory is not None else None
        self.buffer = b''
        script = Path(__file__).resolve().parents[2]/'tools/copilot/motion_worker.py'
        try:
            self.process = subprocess.Popen([str(python), str(script), '--repository', str(repository), '--mode', mode],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1'))
        except BaseException:
            self.closed = True
            self.log.close(); self.directory.cleanup()
            raise
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        os.set_blocking(self.process.stdout.fileno(), False)
        os.set_blocking(self.process.stdin.fileno(), False)
        try:
            ready = self.read(startup_timeout)
            if ready.get('status') != 'ready': raise RuntimeError(str(ready))
            self.manifest = ready['manifest']
            self.manifest['requested_command_format'] = command_format
        except BaseException:
            self.close(interrupt=True); raise

    def read(self, timeout):
        deadline = time.monotonic()+timeout
        while True:
            if b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                result = json.loads(line)
                if not isinstance(result, dict): raise ValueError('Invalid motion worker response')
                return result
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError('Motion worker response timeout')
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data: raise RuntimeError('Motion worker exited')
            self.buffer += data
            if len(self.buffer) > 4*1024*1024: raise ValueError('Motion worker response too large')

    def call(self, operation, **payload):
        if self.closed: raise RuntimeError('Motion worker closed')
        started=time.monotonic();cpu_started=time.process_time()
        self.last_call_timing = None
        self.last_call_request_bytes = None
        if operation == 'validate':
            if payload.get('command_format', self.command_format) != self.command_format:
                raise ValueError('Per-call command format cannot override worker contract')
            payload['command_format'] = self.command_format
        self.sequence += 1
        packet = {'operation': operation, 'request_id': self.sequence, **payload}
        raw = json.dumps(packet, allow_nan=False).encode()+b'\n'
        if len(raw) > 2*1024*1024: raise ValueError('Motion request too large')
        deadline = time.monotonic()+(self.timeout if operation in ('validate','original_manual_ik','original_manual_reset') else 30.)
        try:
            # A stalled worker can fill its stdin pipe before producing output.
            # Bound writing and reading by the same deadline; buffered .write()
            # or readline() would bypass the timeout on partial messages.
            with selectors.DefaultSelector() as writable:
                writable.register(self.process.stdin, selectors.EVENT_WRITE)
                sent = 0
                while sent < len(raw):
                    remaining = deadline-time.monotonic()
                    if remaining <= 0 or not writable.select(remaining):
                        raise TimeoutError('Motion worker request timeout')
                    try:
                        sent += os.write(self.process.stdin.fileno(), raw[sent:sent+65536])
                    except BlockingIOError:
                        continue
            result = self.read(max(0., deadline-time.monotonic()))
            if result.get('request_id') != self.sequence or result.get('status') != 'ok':
                raise RuntimeError(str(result))
            if (operation == 'validate' and result['result'].get('allowed') is True and
                    result['result'].get('command_format', 'float64') != self.command_format):
                raise RuntimeError('Motion worker returned a different command format')
            if (operation == 'validate' and self.rejection_directory is not None
                    and result['result'].get('allowed') is not True):
                from .mailbox import atomic_json
                atomic_json(self.rejection_directory/f'{self.sequence:08d}.json',
                    {'request': packet, 'result': result['result']})
            self.last_call_timing = dict(operation=operation, request_id=self.sequence,
                client_wall_s=time.monotonic()-started,client_process_cpu_s=time.process_time()-cpu_started,
                worker=result.get('diagnostic_timing'))
            # Keep only the already serialized, bounded bytes of this completed
            # call. Simulation diagnostics can retain them after a later lease
            # failure without serialization or disk I/O on the command path.
            self.last_call_request_bytes = raw if operation == 'validate' else None
            return result['result']
        except BaseException:
            self.close(interrupt=True); raise

    def close(self, *, interrupt=False):
        if self.closed: return
        self.closed = True
        try:
            # EOF lets the real worker finish through Python's normal shutdown.
            # Original IK imports logging_mp, whose listener ignores SIGTERM;
            # its atexit sentinel/join must run instead of orphaning the child.
            self.process.stdin.close()
            # No more replies are consumed after close. Release a worker that
            # might otherwise block flushing a final large reply into the pipe.
            self.process.stdout.close()
            if self.process.poll() is None:
                if interrupt:
                    # A failed call may still be computing or reading a partial
                    # request. Our worker handles this as a cooperative stop.
                    self.process.terminate()
                try:
                    self.process.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        # A native-code stall cannot cooperate. The launcher's
                        # bounded cleanup still owns this entire process group.
                        self.process.kill(); self.process.wait(timeout=3)
        finally:
            self.selector.close()
            self.process.stdout.close(); self.log.close()
            self.directory.cleanup()
