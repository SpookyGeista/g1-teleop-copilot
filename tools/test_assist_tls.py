#!/usr/bin/env python3
"""Real OpenSSL chain/SAN/permissions validation; no trust store modifications."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from assist_tls import generate, san_host


class TLSHelperTest(unittest.TestCase):
    def test_host_validation(self):
        self.assertEqual(san_host('192.168.1.12'),('IP','192.168.1.12'))
        self.assertEqual(san_host('::1'),('IP','::1'))
        self.assertEqual(san_host('sim-host.local'),('DNS','sim-host.local'))
        for bad in ['https://host', 'host:123', '*.local', 'host\n[evil]', 'host,IP:1.2.3.4', '-host', 'host_', '']:
            with self.assertRaises(ValueError):san_host(bad)

    def test_real_cert_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='assist_tls_test_') as parent:
            output=Path(parent)/'tls'
            metadata=generate(output,['192.168.1.12','sim-host.local','::1'],1)
            self.assertFalse(metadata['trust_installed'])
            self.assertIn('OK',metadata['chain_verification'])
            self.assertEqual(output.stat().st_mode & 0o777,0o700)
            for name in ['ca.key','server.key']:
                self.assertEqual((output/name).stat().st_mode & 0o777,0o600)
            before=hashlib.sha256((output/'server.key').read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError):generate(output,['different.local'],1)
            self.assertEqual(hashlib.sha256((output/'server.key').read_bytes()).hexdigest(),before)


if __name__=='__main__':unittest.main()
