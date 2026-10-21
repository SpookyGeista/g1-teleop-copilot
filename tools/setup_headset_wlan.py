#!/usr/bin/env python3
"""Reconnect one known Quest via existing WLAN ADB, preserving localhost WebXR.

No USB is needed if wireless ADB is already enabled and authorized. This helper
does not enable TCP debugging, pair devices, install trust, or open LAN servers.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import re
import socket
import subprocess
import sys

from setup_headset_usb import PORT, run_adb, verify_install


def validate_serial(serial):
    if not isinstance(serial, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', serial):
        raise ValueError('Use the known physical headset serial, not an IP address')
    return serial


def private_endpoint(value):
    """Only numeric RFC1918 IPv4 endpoints; no DNS lookup or public connection."""
    try:
        address, port = value.rsplit(':', 1)
        ip = ipaddress.IPv4Address(address)
        number = int(port)
        private = any(ip in ipaddress.ip_network(net) for net in
                      ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))
        if not private or not 1 <= number <= 65535 or str(number) != port:
            raise ValueError()
    except (AttributeError, ValueError) as exc:
        raise ValueError('WLAN endpoint must be a private IPv4 address and port') from exc
    return str(ip), number


def discover_endpoint(output, serial, requested=None):
    validate_serial(serial)
    endpoints = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == 'adb-' + serial and fields[1] == '_adb._tcp':
            private_endpoint(fields[2])
            endpoints.add(fields[2])
    if requested is not None:
        private_endpoint(requested)
        if requested not in endpoints:
            raise ValueError('Requested endpoint is not advertised for this exact headset serial')
        return requested
    if len(endpoints) != 1:
        raise ValueError('Expected one WLAN ADB advertisement for this headset. Wake it on the same WLAN; '
                         'wireless debugging must already be enabled. No device was contacted.')
    return endpoints.pop()


def device_state(output, endpoint):
    for line in output.splitlines():
        row = line.split()
        if len(row) >= 2 and row[0] == endpoint:
            return row[1]
    return 'disconnected'


def checked_mapping(output):
    mappings = [row for line in output.splitlines() if len(row := line.split()) == 3 and row[1] == PORT]
    if any(row[2] != PORT for row in mappings):
        raise ValueError('Headset port 8765 maps elsewhere; refusing to replace or remove it')
    return bool(mappings)


def reachable(endpoint):
    address, port = private_endpoint(endpoint)
    try:
        with socket.create_connection((address, port), timeout=3):
            pass
    except OSError as exc:
        raise ValueError('Known Quest WLAN endpoint is not responding. Wake it, confirm the same WLAN, '
                         'and retry; its mDNS announcement alone does not prove connectivity.') from exc


def configure(adb, serial, command, endpoint=None):
    validate_serial(serial)
    endpoint = discover_endpoint(run_adb(adb, 'mdns', 'services'), serial, endpoint)
    state = device_state(run_adb(adb, 'devices', '-l'), endpoint)
    if state != 'device' and command == 'reverse':
        reachable(endpoint)
        result = run_adb(adb, 'connect', endpoint)
        state = device_state(run_adb(adb, 'devices', '-l'), endpoint)
        if state != 'device':
            raise ValueError('Quest WLAN ADB is not authorized/online (' + state + '). '
                             'Confirm the debugging prompt in the headset and retry. ' + result[:160])
    if state != 'device':
        raise ValueError('Quest WLAN transport is ' + state + '. Use reverse to reconnect; '
                         'authorize this PC in the headset if prompted.')
    actual_serial = run_adb(adb, '-s', endpoint, 'shell', 'getprop', 'ro.serialno')
    if actual_serial != serial:
        raise ValueError('Connected device serial differs from the requested headset; no ports were changed')
    prefix = ('-s', endpoint, 'reverse')
    present = checked_mapping(run_adb(adb, *prefix, '--list'))
    if command == 'reverse' and not present:
        run_adb(adb, *prefix, '--no-rebind', PORT, PORT)
    elif command == 'remove' and present:
        run_adb(adb, *prefix, '--remove', PORT)
    present = checked_mapping(run_adb(adb, *prefix, '--list'))
    if command == 'reverse' and not present:
        raise ValueError('ADB returned without establishing the expected localhost mapping')
    if command == 'remove' and present:
        raise ValueError('ADB returned but the localhost mapping is still present')
    return {'headset_serial': serial, 'wlan_endpoint': endpoint,
            'identity_verified': True, 'reverse_ready': present,
            'headset_url': 'http://localhost:8765/', 'pc_url': 'http://127.0.0.1:8765/',
            'transport': 'existing wireless ADB', 'browser_or_vr_started': False,
            'headset_webxr_verified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['reverse', 'status', 'remove'])
    parser.add_argument('--serial', required=True, help='Physical serial of the already known headset')
    parser.add_argument('--endpoint', help='Optional exact private IPv4:port; must match this serial in mDNS')
    args = parser.parse_args(argv)
    try:
        validate_serial(args.serial)
        if args.endpoint is not None:
            private_endpoint(args.endpoint)
        adb, _ = verify_install()
        print(json.dumps(configure(adb, args.serial, args.command, args.endpoint), indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print('Headset WLAN setup: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
