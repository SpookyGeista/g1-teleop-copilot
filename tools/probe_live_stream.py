#!/usr/bin/env python3
"""Read-only local stream load/consistency probe alongside an Isaac run.

Requests actual RGB/depth pairs, exercising HTTP encoding during simulation.
It sends no commands and measures neither headset rendering nor remote latency.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen


def read(url):
    with urlopen(url, timeout=3.) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=120.)
    parser.add_argument('--display', action='store_true', help='Request independent raw display frames; measurement state remains separate')
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 300:
        parser.error('--max-seconds must be within 1..300')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error('--output must be a new report path')
    deadline = time.monotonic() + args.max_seconds
    manifest = args.recording / 'manifest.json'
    while True:
        try:
            url = json.loads(manifest.read_text())['url']
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise TimeoutError('Isaac manifest did not appear')
            time.sleep(.05)
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('Probe accepts only the local HTTP simulation endpoint')
    frames = []
    errors = []
    expired = 0
    previous = None
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        while time.monotonic() < deadline:
            try:
                state = json.loads(read(url + '/api/state'))
                display = state.get('display_frame') if args.display else None
                if args.display and not display:
                    time.sleep(.05)
                    continue
                descriptor = display or state
                identifier = int(descriptor['frame_id'])
                if identifier == previous:
                    time.sleep(.05)
                    continue
                previous = identifier
                camera = descriptor['camera']
                width, height = int(camera['width']), int(camera['height'])
                begun = time.monotonic()
                rgb = pool.submit(read, url + f'/api/frame.jpg?frame_id={identifier}')
                depth = pool.submit(read, url + f'/api/depth.bin?frame_id={identifier}')
                rgb_bytes, depth_bytes = rgb.result(), depth.result()
                unpacked = gzip.decompress(depth_bytes)
                if not rgb_bytes.startswith(b'\xff\xd8') or not rgb_bytes.endswith(b'\xff\xd9'):
                    raise ValueError('Invalid JPEG framing')
                if len(unpacked) != width * height * 4:
                    raise ValueError('Depth dimensions do not match this frame descriptor')
                frames.append({'frame_id': identifier, 'width': width, 'height': height,
                    'measurement_frame_id':state['frame_id'],
                    'measurement_age_s':state.get('frame_age'),
                    'display_age_s':display.get('age_s') if display else None,
                    'rgb_bytes': len(rgb_bytes), 'compressed_depth_bytes': len(depth_bytes),
                    'paired_request_ms': 1000 * (time.monotonic() - begun)})
            except HTTPError as error:
                if error.code == 410:
                    expired += 1
                    continue
                errors.append(str(error))
                break
            except (OSError, URLError) as error:
                # Shutdown at the end of an automatic run is expected.
                # assist_sim closes HTTP after writing acceptance, but before
                # writing its final result.json in the cleanup block.
                finished = (args.recording / 'result.json').is_file()
                acceptance = args.recording / 'acceptance.json'
                if acceptance.is_file():
                    try:
                        finished = finished or bool(json.loads(acceptance.read_text()).get('complete'))
                    except json.JSONDecodeError:
                        pass
                if frames and finished:
                    break
                errors.append(str(error))
                break
            except (ValueError, KeyError) as error:
                errors.append(str(error))
                break
            time.sleep(.05)
    durations = sorted(frame['paired_request_ms'] for frame in frames)
    report = {'passed': len(frames) >= 10 and not errors,
        'scope': 'Read-only local paired RGB/depth requests concurrent with actual simulation',
        'recording': str(args.recording.resolve()), 'headset_verified': False,
        'stream_type': 'independent_raw_display' if args.display else 'perception_paired',
        'commands_sent': 0, 'received_frame_pairs': len(frames),
        'expired_frame_requests': expired, 'errors': errors,
        'wall_seconds': time.monotonic() - started,
        'paired_request_median_ms': statistics.median(durations) if durations else None,
        'paired_request_p95_ms': durations[min(len(durations)-1, int(.95*len(durations)))] if durations else None,
        'frames': frames,
        'limits': ['Request timings are local HTTP service measurements, not headset or motion-to-photon latency.',
                   'A valid stream report does not imply that the separate physical task passed.']}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'frames'}))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
