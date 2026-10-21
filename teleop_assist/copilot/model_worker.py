"""Private stdin/stdout inference protocol; stdout is strictly JSON-lines."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from teleop_assist.copilot.observations import GeometryObservation


def send(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    work = args.directory.resolve()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from teleop_assist.copilot.grasp_model import GraspModel
            model = GraspModel(**json.loads(args.config))
        send({'status': 'ready', 'manifest': model.manifest})
    except Exception as exc:
        send({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'})
        return 1
    for line in sys.stdin:
        identifier = None
        try:
            request = json.loads(line)
            identifier = request['request_id']
            if type(identifier) is not int or identifier < 1:
                raise ValueError('Invalid request ID')
            path = work / f'{identifier:08d}.npz'
            observation = GeometryObservation.from_npz(path)
            with contextlib.redirect_stdout(sys.stderr):
                result = model.infer(observation, int(request.get('seed', 6103)))
            send({'status': 'ok', 'request_id': identifier, 'result': result})
        except Exception as exc:
            send({'status': 'error', 'request_id': identifier, 'error': f'{type(exc).__name__}: {exc}'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
