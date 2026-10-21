"""USB helper checks with mocked ADB; never touches a connected device."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import setup_headset_usb as usb


class UsbHelperTests(unittest.TestCase):
    def test_only_explicit_authorized_usb_transport_is_accepted(self):
        listing='List of devices attached\nQUEST device usb:1-2 product:test\nPICO unauthorized usb:1-3\nHOST:5555 device product:test\n'
        usb.require_usb_device(listing,'QUEST')
        for serial in ('', '-s', 'PICO', 'HOST:5555', 'MISSING'):
            with self.subTest(serial=serial), self.assertRaises(ValueError):
                usb.require_usb_device(listing,serial)

    def test_archive_paths_cannot_escape_install_directory(self):
        self.assertEqual(str(usb.checked_member('platform-tools/adb')),'platform-tools/adb')
        for name in ('../adb','/platform-tools/adb','platform-tools/../adb','adb'):
            with self.subTest(name=name), self.assertRaises(ValueError):usb.checked_member(name)

    def invoke(self,args,responses):
        with patch.object(usb,'verify_install',return_value=(Path('/verified/adb'),{})), \
             patch.object(usb,'run_adb',side_effect=responses) as command, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code=usb.main(args)
        return code,[call.args[1:] for call in command.call_args_list]

    def test_missing_serial_never_starts_adb(self):
        code,calls=self.invoke(['reverse'],[])
        self.assertEqual(code,1);self.assertEqual(calls,[])

    def test_reverse_is_explicit_usb_only_and_does_not_replace_other_ports(self):
        listing='List of devices attached\nQUEST device usb:1-2\n'
        code,calls=self.invoke(['reverse','--serial','QUEST'],[listing,'','','QUEST tcp:8765 tcp:8765'])
        self.assertEqual(code,0)
        self.assertIn(('-s','QUEST','reverse','--no-rebind','tcp:8765','tcp:8765'),calls)
        code,calls=self.invoke(['reverse','--serial','QUEST'],[listing,'QUEST tcp:8765 tcp:9999'])
        self.assertEqual(code,1)
        self.assertFalse(any('--no-rebind' in call for call in calls))

    def test_existing_expected_mapping_is_idempotent(self):
        listing='List of devices attached\nQUEST device usb:1-2\n'
        code,calls=self.invoke(['reverse','--serial','QUEST'],[listing,'QUEST tcp:8765 tcp:8765','QUEST tcp:8765 tcp:8765'])
        self.assertEqual(code,0)
        self.assertFalse(any('--no-rebind' in call for call in calls))

    def test_remove_only_deletes_the_expected_mapping(self):
        listing='List of devices attached\nQUEST device usb:1-2\n'
        code,calls=self.invoke(['remove','--serial','QUEST'],[listing,'QUEST tcp:8765 tcp:8765',''])
        self.assertEqual(code,0)
        self.assertIn(('-s','QUEST','reverse','--remove','tcp:8765'),calls)
        code,calls=self.invoke(['remove','--serial','QUEST'],[listing,'QUEST tcp:8765 tcp:9999'])
        self.assertEqual(code,1)
        self.assertFalse(any('--remove' in call for call in calls))

    def test_remove_does_not_touch_other_ports(self):
        listing='List of devices attached\nQUEST device usb:1-2\n'
        code,calls=self.invoke(['remove','--serial','QUEST'],[listing,'QUEST tcp:9000 tcp:9000'])
        self.assertEqual(code,0)
        self.assertFalse(any('--remove' in call for call in calls))


if __name__=='__main__':unittest.main()
