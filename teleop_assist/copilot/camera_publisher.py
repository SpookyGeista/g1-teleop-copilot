"""One in-flight display/perception publication, outside the physics process."""
from pathlib import Path
import multiprocessing
import os
import time


def _serve(connection, directory, assistance):
    # This is a freshly spawned CPU worker. Set limits before importing NumPy;
    # Isaac's parent process can have replaced the launcher's environment.
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[variable]='1'
    from .frames import publish_frame
    from .mailbox import read_json
    from .xr_camera import publish_operator_frame
    from .xr_grasp_overlay import build_operator_grasp_preview
    try:
        hand=None
        if assistance:
            from .preview import HandPreview
            from .collision import GraspCollisionFilter
            hand=HandPreview(GraspCollisionFilter())
        directory=Path(directory)
        connection.send(('ready',None))
        while True:
            job=connection.recv()
            if job is None:break
            started=time.monotonic()
            frame,primary,choices,selected=job
            # The operator image is available before any mesh or model work.
            publish_operator_frame(directory/'operator_camera',frame)
            if assistance:
                try:
                    preview=build_operator_grasp_preview(choices,selected,
                        read_json(directory/'preview_proposals.json',{}),frame,hand)
                except Exception as exc:
                    # A display-only mesh failure must not take down teleop.
                    preview={'preview_current':False,'error':type(exc).__name__}
                publish_operator_frame(directory/'assist_camera',frame,grasp_preview=preview)
                publish_frame(directory/'ai_frames',primary)
            if not assistance:
                (directory/'frame.jpg').write_bytes((directory/'operator_camera/frame.jpg').read_bytes())
            connection.send(('done',time.monotonic()-started))
    except Exception as exc:
        connection.send(('error',f'{type(exc).__name__}: {exc}'))
    finally:
        connection.close()


class CameraPublisher:
    """Spawn a CPU-only worker; never queue old frames or pass simulator objects."""
    def __init__(self, directory, *, assistance):
        context=multiprocessing.get_context('spawn')
        self.connection,child=context.Pipe()
        self.process=context.Process(target=_serve,args=(child,str(directory),assistance),daemon=True)
        self.busy=False;self.last_work_s=0.
        self.process.start();child.close()
        try:
            if not self.connection.poll(15.):raise RuntimeError('Camera publisher startup timeout')
            kind,value=self.connection.recv()
            if kind!='ready':raise RuntimeError('Camera publisher: '+str(value))
        except Exception:
            self.close();raise

    def poll(self):
        if self.connection.poll():
            kind,value=self.connection.recv()
            if kind!='done':raise RuntimeError('Camera publisher: '+str(value))
            self.busy=False;self.last_work_s=value
        if not self.process.is_alive():raise RuntimeError('Camera publisher exited')
        return self.last_work_s

    def submit(self,frame,primary,choices,selected):
        if self.busy:raise RuntimeError('Camera publisher already has an in-flight frame')
        self.connection.send((frame,primary,choices,selected))
        self.busy=True

    def close(self):
        try:
            if self.process.is_alive():self.connection.send(None)
        except (BrokenPipeError,EOFError,OSError):pass
        self.process.join(timeout=5.)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=2.)
        self.connection.close()
