#!/usr/bin/env python3
"""Install pinned official ADB locally; USB forwarding is an explicit command."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / '.tools'
INSTALL = TOOLS / 'platform-tools'
MANIFEST = TOOLS / 'platform-tools-install.json'
VERSION = '37.0.1'
ARCHIVE = f'platform-tools_r{VERSION}-linux.zip'
PUBLISHED_SHA1 = '477254aa5f903c15cf51001717bdf347fb6b53e0'
PUBLISHED_SIZE = 9054187
REPOSITORY = 'https://dl.google.com/android/repository/'
METADATA_URL = REPOSITORY + 'repository2-1.xml'
RELEASE_NOTES = 'https://developer.android.com/tools/releases/platform-tools'
PORT = 'tcp:8765'


def digest(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def fetch(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        if not response.geturl().startswith(REPOSITORY):
            raise ValueError('Unexpected download redirect outside the official Android repository')
        return response.read()


def checked_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'platform-tools':
        raise ValueError('Unexpected path in Platform Tools archive')
    return path


def verify_install():
    if not INSTALL.is_dir() or not MANIFEST.is_file():
        raise ValueError('Local ADB is not installed. Run: python3 tools/setup_headset_usb.py install')
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get('version') != VERSION or manifest.get('published_archive_sha1') != PUBLISHED_SHA1:
        raise ValueError('Local manifest does not match the pinned official Platform Tools release')
    for name, expected in manifest['files_sha256'].items():
        checked_member(name)
        path = TOOLS / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError(f'Local Platform Tools hash mismatch: {name}')
    adb = INSTALL / 'adb'
    if 'platform-tools/adb' not in manifest['files_sha256'] or not os.access(adb, os.X_OK):
        raise ValueError('Verified executable adb is missing')
    return adb, manifest


def run_adb(adb, *arguments):
    env = os.environ.copy()
    env['ADB_MDNS_AUTO_CONNECT'] = '0'
    result = subprocess.run([str(adb), *arguments], text=True, capture_output=True, timeout=20, env=env)
    if result.returncode:
        raise ValueError((result.stderr or result.stdout).strip() or f'ADB exited with {result.returncode}')
    return result.stdout.strip()


def install():
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64'):
        raise ValueError('This pinned distribution is for Linux x86_64')
    if INSTALL.exists() or MANIFEST.exists():
        adb, manifest = verify_install()
        return {'already_installed': True, 'manifest': str(MANIFEST), 'adb_version': run_adb(adb, 'version')}
    metadata = fetch(METADATA_URL)
    repository = ET.fromstring(metadata)
    # The exact stable-release URL, size and published hash must all match.
    package = next((p for p in repository.findall('remotePackage') if p.get('path') == 'platform-tools'
                    and p.find('channelRef') is not None and p.find('channelRef').get('ref') == 'channel-0'), None)
    linux = next((a for a in package.findall('archives/archive') if a.findtext('host-os') == 'linux'), None) if package is not None else None
    if (linux is None or linux.findtext('complete/url') != ARCHIVE
            or linux.findtext('complete/checksum') != PUBLISHED_SHA1
            or int(linux.findtext('complete/size', '0')) != PUBLISHED_SIZE):
        raise ValueError('Official repository no longer matches the pinned release; review the new release before updating this helper')
    payload = fetch(REPOSITORY + ARCHIVE)
    if len(payload) != PUBLISHED_SIZE or hashlib.sha1(payload).hexdigest() != PUBLISHED_SHA1:
        raise ValueError('Archive does not match the published official checksum and size')
    TOOLS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='platform-tools-stage-', dir=TOOLS) as temporary:
        staging = Path(temporary)
        archive = staging / ARCHIVE
        archive.write_bytes(payload)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                checked_member(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('Unexpected archive symlink')
                bundle.extract(member, staging)
                path = staging / member.filename
                if path.is_file():
                    path.chmod(0o755 if (member.external_attr >> 16) & 0o111 else 0o644)
        staged_adb = staging / 'platform-tools' / 'adb'
        version = run_adb(staged_adb, 'version')  # No daemon or device command.
        if f'Version {VERSION}-' not in version:
            raise ValueError('Downloaded adb version differs from pinned package')
        files = {str(p.relative_to(staging)): digest(p) for p in sorted((staging / 'platform-tools').rglob('*')) if p.is_file()}
        manifest = {'schema': 1, 'version': VERSION, 'download_url': REPOSITORY + ARCHIVE,
            'official_metadata_url': METADATA_URL, 'official_release_notes': RELEASE_NOTES,
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'published_archive_sha1': PUBLISHED_SHA1, 'archive_size_bytes': len(payload),
            'archive_sha256': hashlib.sha256(payload).hexdigest(),
            'archive_sha256_source': 'computed locally after verification against official HTTPS metadata SHA-1',
            'repository_metadata_sha256': hashlib.sha256(metadata).hexdigest(),
            'files_sha256': files, 'adb_version_during_install': version,
            'device_commands_executed': False, 'headset_connection_verified': False}
        (staging / 'repository2-1.xml').write_bytes(metadata)
        (staging / 'platform-tools-install.json').write_text(json.dumps(manifest, indent=2) + '\n')
        shutil.move(str(staging / 'platform-tools'), INSTALL)
        shutil.move(str(archive), TOOLS / ARCHIVE)
        shutil.move(str(staging / 'repository2-1.xml'), TOOLS / 'platform-tools-repository.xml')
        shutil.move(str(staging / 'platform-tools-install.json'), MANIFEST)
    adb, _ = verify_install()
    manifest['adb_version'] = run_adb(adb, 'version')
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    return {'installed': str(INSTALL), 'manifest': str(MANIFEST), 'adb_version': manifest['adb_version'],
            'device_commands_executed': False, 'headset_connection_verified': False}


def require_usb_device(output, serial):
    if not serial or serial.startswith('-') or any(c.isspace() for c in serial):
        raise ValueError('Choose an explicit headset serial from the device list')
    row = next((line.split() for line in output.splitlines() if line.split() and line.split()[0] == serial), None)
    if row is None or len(row) < 2:
        raise ValueError('Requested headset is not listed; connect its USB data cable')
    if row[1] != 'device':
        raise ValueError(f'Headset state is {row[1]}; authorize this computer in the headset first')
    if not any(part.startswith('usb:') for part in row[2:]):
        raise ValueError('The selected transport is not reported as USB; this helper does not pair or connect over a network')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['install', 'version', 'devices', 'reverse', 'status', 'remove'])
    parser.add_argument('--serial', help='Explicit USB headset serial; required for reverse/status/remove')
    args = parser.parse_args(argv)
    try:
        if args.command == 'install':
            print(json.dumps(install(), indent=2)); return 0
        adb, _ = verify_install()
        if args.command == 'version':
            print(run_adb(adb, 'version')); return 0
        if args.command in ('reverse', 'status', 'remove') and not args.serial:
            raise ValueError('--serial is required; first run the devices command')
        devices = run_adb(adb, 'devices', '-l')
        if args.command == 'devices':
            print(devices); return 0
        require_usb_device(devices, args.serial)
        prefix = ('-s', args.serial, 'reverse')
        mappings = run_adb(adb, *prefix, '--list')
        current = [line.split() for line in mappings.splitlines() if len(line.split()) == 3 and line.split()[1] == PORT]
        if args.command == 'status':
            print(mappings or 'No reverse mappings'); return 0
        if current and any(row[2] != PORT for row in current):
            raise ValueError('Port 8765 already maps elsewhere; refusing to replace or remove another mapping')
        if args.command == 'reverse':
            if not current: run_adb(adb, *prefix, '--no-rebind', PORT, PORT)
            print(run_adb(adb, *prefix, '--list'))
            print('Open http://localhost:8765/ in the headset browser. WebXR must still be checked there.')
        else:
            if current: run_adb(adb, *prefix, '--remove', PORT)
            print('The selected headset has no reverse mapping for tcp:8765.')
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, ET.ParseError, zipfile.BadZipFile) as error:
        print(f'Headset USB setup: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
