"""No device I/O: WLAN identity, scoping, reconnect and mapping preservation."""
from pathlib import Path
import unittest
from unittest.mock import patch

import setup_headset_wlan as wlan


SERIAL = 'QUEST123'
ENDPOINT = '192.168.0.100:5555'
MDNS = 'List of discovered mdns services\nadb-QUEST123\t_adb._tcp\t' + ENDPOINT + '\n'
DEVICE = 'List of devices attached\n' + ENDPOINT + ' device product:test\n'
MAPPING = 'TcpFfs tcp:8765 tcp:8765'


class WlanTests(unittest.TestCase):
    def test_discovery_requires_exact_known_serial_and_one_private_endpoint(self):
        self.assertEqual(wlan.discover_endpoint(MDNS, SERIAL), ENDPOINT)
        for output, serial, requested in [
                (MDNS, 'OTHER', None), (MDNS, SERIAL, '192.168.0.101:5555'),
                (MDNS + 'adb-QUEST123 _adb._tcp 192.168.0.102:5555', SERIAL, None),
                (MDNS.replace('192.168.0.100', '8.8.8.8'), SERIAL, None)]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                wlan.discover_endpoint(output, serial, requested)

    def test_no_dns_public_loopback_or_shell_syntax(self):
        for endpoint in ['example.org:5555', '8.8.8.8:5555', '127.0.0.1:5555',
                         '192.168.0.1:0', '192.168.0.1:65536', '192.168.0.1:5555;echo']:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                wlan.private_endpoint(endpoint)
        for serial in ['', '-s', 'host:5555', 'a b']:
            with self.subTest(serial=serial), self.assertRaises(ValueError):
                wlan.validate_serial(serial)

    def configure(self, responses, command='reverse'):
        with patch.object(wlan, 'run_adb', side_effect=responses) as run, \
             patch.object(wlan, 'reachable'):
            value = wlan.configure(Path('/verified/adb'), SERIAL, command)
        return value, [call.args[1:] for call in run.call_args_list]

    def test_disconnected_headset_reconnects_and_identity_precedes_reverse(self):
        value, calls = self.configure([MDNS, '', 'connected', DEVICE, SERIAL, '', '', MAPPING])
        self.assertTrue(value['reverse_ready'])
        self.assertFalse(value['headset_webxr_verified'])
        self.assertIn(('connect', ENDPOINT), calls)
        identity = ('-s', ENDPOINT, 'shell', 'getprop', 'ro.serialno')
        reverse = ('-s', ENDPOINT, 'reverse', '--no-rebind', 'tcp:8765', 'tcp:8765')
        self.assertLess(calls.index(identity), calls.index(reverse))

    def test_existing_mapping_is_idempotent(self):
        value, calls = self.configure([MDNS, DEVICE, SERIAL, MAPPING, MAPPING])
        self.assertTrue(value['reverse_ready'])
        self.assertFalse(any('connect' in call or '--no-rebind' in call for call in calls))

    def test_mismatched_device_never_touches_reverse(self):
        with patch.object(wlan, 'run_adb', side_effect=[MDNS, DEVICE, 'OTHER']) as run:
            with self.assertRaisesRegex(ValueError, 'serial differs'):
                wlan.configure('/adb', SERIAL, 'reverse')
        self.assertFalse(any('reverse' in call.args for call in run.call_args_list))

    def test_unauthorized_device_never_receives_shell_or_forward(self):
        with patch.object(wlan, 'run_adb', side_effect=[MDNS, '', 'failed to authenticate',
                ENDPOINT + ' unauthorized']) as run, patch.object(wlan, 'reachable'):
            with self.assertRaisesRegex(ValueError, 'Confirm the debugging prompt'):
                wlan.configure('/adb', SERIAL, 'reverse')
        self.assertFalse(any('shell' in call.args or 'reverse' in call.args for call in run.call_args_list))

    def test_conflicting_mapping_is_never_replaced_or_removed(self):
        for command in ['reverse', 'remove']:
            with self.subTest(command=command), patch.object(wlan, 'run_adb',
                    side_effect=[MDNS, DEVICE, SERIAL, 'TcpFfs tcp:8765 tcp:9999']) as run:
                with self.assertRaisesRegex(ValueError, 'maps elsewhere'):
                    wlan.configure('/adb', SERIAL, command)
            self.assertFalse(any('--remove' in call.args or '--no-rebind' in call.args for call in run.call_args_list))

    def test_remove_only_expected_mapping(self):
        value, calls = self.configure([MDNS, DEVICE, SERIAL, MAPPING, '', ''], command='remove')
        self.assertFalse(value['reverse_ready'])
        self.assertIn(('-s', ENDPOINT, 'reverse', '--remove', 'tcp:8765'), calls)

    def test_status_does_not_connect_offline_device(self):
        with patch.object(wlan, 'run_adb', side_effect=[MDNS, '']) as run:
            with self.assertRaisesRegex(ValueError, 'disconnected'):
                wlan.configure('/adb', SERIAL, 'status')
        self.assertFalse(any('connect' in call.args for call in run.call_args_list))


if __name__ == '__main__':
    unittest.main()
