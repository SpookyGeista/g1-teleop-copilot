#!/usr/bin/env python3
"""Create a private local CA and SAN certificate for the simulation viewer.

Does not install trust, change network configuration, start a service, or overwrite
any existing directory/key. Only the public ca.crt is intended for client trust.
"""
from __future__ import annotations
import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def san_host(value):
    """Return a validated OpenSSL SAN token; reject config injection and URLs."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError('Host must be an IP address or DNS name without whitespace')
    try:
        return 'IP', str(ipaddress.ip_address(value))
    except ValueError:
        pass
    if any(c in value for c in '\r\n\x00/,;=:*[]'):
        raise ValueError('Use an IP address or DNS name, without URL, port or wildcard')
    try:
        name = value.rstrip('.').encode('idna').decode('ascii').lower()
    except UnicodeError as exc:
        raise ValueError('Invalid DNS name') from exc
    if len(name) > 253 or not all(re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', label)
                                  for label in name.split('.')):
        raise ValueError('Invalid DNS name')
    return 'DNS', name


def generate(output, hosts, days=30, openssl=None):
    if not isinstance(days, int) or not 1 <= days <= 365:
        raise ValueError('Certificate lifetime must be 1..365 days')
    executable = openssl or shutil.which('openssl')
    if not executable:
        raise ValueError('OpenSSL executable not available')
    sans = list(dict.fromkeys([san_host(h) for h in ['localhost', '127.0.0.1', *hosts]]))
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    old_umask = os.umask(0o077)
    try:
        # Atomic ownership of a new folder: even an existing empty folder is refused.
        output.mkdir(mode=0o700, exist_ok=False)
        (output/'.gitignore').write_text('*\n!.gitignore\n')
        config = output/'server_extensions.cnf'
        config.write_text('''[server]
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=@names
[names]
''' + '\n'.join(f'{kind}.{i+1}={value}' for i, (kind, value) in enumerate(sans))+'\n')
        def run(*args):
            result = subprocess.run([str(executable), *map(str,args)], check=False,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode:
                raise RuntimeError('OpenSSL certificate operation failed: '+result.stderr[-1000:])
            return result.stdout.strip()
        run('req','-x509','-newkey','rsa:3072','-sha256','-nodes','-days',days+30,
            '-keyout',output/'ca.key','-out',output/'ca.crt',
            '-subj','/CN=punktNULL Local Simulation CA',
            '-addext','basicConstraints=critical,CA:TRUE,pathlen:0',
            '-addext','keyUsage=critical,keyCertSign,cRLSign')
        run('req','-new','-newkey','rsa:2048','-sha256','-nodes',
            '-keyout',output/'server.key','-out',output/'server.csr',
            '-subj','/CN=punktNULL Local Simulation Viewer')
        run('x509','-req','-in',output/'server.csr','-CA',output/'ca.crt',
            '-CAkey',output/'ca.key','-CAcreateserial','-out',output/'server.crt',
            '-days',days,'-sha256','-extfile',config,'-extensions','server')
        for key in ['ca.key','server.key']:
            (output/key).chmod(0o600)
        validation=run('verify','-CAfile',output/'ca.crt',output/'server.crt')
        for kind,host in sans:
            run('verify','-CAfile',output/'ca.crt',
                '-verify_ip' if kind=='IP' else '-verify_hostname',host,output/'server.crt')
        metadata={'purpose':'Local Isaac simulation viewer; trust installation is manual',
                  'certificate':str(output/'server.crt'),'private_key':str(output/'server.key'),
                  'trust_certificate':str(output/'ca.crt'),'days':days,
                  'subject_alternative_names':[{'type':k,'value':v} for k,v in sans],
                  'ca_sha256_fingerprint':run('x509','-in',output/'ca.crt','-noout','-fingerprint','-sha256'),
                  'chain_verification':validation,
                  'openssl_version':run('version'),'trust_installed':False}
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return metadata
    finally:
        os.umask(old_umask)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New directory; existing paths are refused')
    parser.add_argument('--host',action='append',default=[],help='Actual LAN IP or DNS name; repeat for aliases')
    parser.add_argument('--days',type=int,default=30)
    args=parser.parse_args()
    try:
        metadata=generate(args.output,args.host,args.days)
    except (ValueError,FileExistsError,OSError,RuntimeError) as exc:
        parser.exit(2,str(exc)+'\nNo trust settings were installed. Existing files were not overwritten.\n')
    print(json.dumps(metadata,indent=2))
    print('Pass --cert server.crt --key server.key to the viewer. Trust only ca.crt on the headset.\n'
          'Keep ca.key and server.key private. Creating these files alone does not enable trusted WebXR.')


if __name__=='__main__':
    main()
