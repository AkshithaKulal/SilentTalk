#!/usr/bin/env python3
"""Create a self-signed TLS cert so phone browsers can use the camera over LAN.

Browsers only allow getUserMedia on secure contexts (HTTPS or localhost).
Phone → http://192.168.x.x is blocked; https://192.168.x.x works after you accept the warning.

  python isl_recognition/make_dev_cert.py
  python app.py --https
"""

from __future__ import annotations

import datetime
import ipaddress
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERT_DIR = ROOT / "certs"
CERT_FILE = CERT_DIR / "dev-cert.pem"
KEY_FILE = CERT_DIR / "dev-key.pem"


def local_ips() -> list[str]:
    ips = {"127.0.0.1"}
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips)


def main() -> int:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        print("Installing cryptography...", flush=True)
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography", "-q"])
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ips = local_ips()
    names = [x509.DNSName("localhost")]
    for ip in ips:
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "SilentTalk-dev"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .sign(key, hashes.SHA256())
    )

    KEY_FILE.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"Wrote {CERT_FILE}")
    print(f"Wrote {KEY_FILE}")
    print(f"SANs include: {', '.join(ips)}")
    print("Next:  python app.py --https")
    print("Phone: https://<your-wifi-ip>:5000  (tap Advanced -> Proceed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
