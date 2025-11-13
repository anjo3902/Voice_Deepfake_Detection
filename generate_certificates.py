"""
Generate Self-Signed SSL Certificates for HTTPS Server
Run this script once before starting the application.
"""

from pathlib import Path
import subprocess
import sys

def generate_certificates():
    """Generate self-signed SSL certificates using OpenSSL"""
    
    cert_dir = Path(__file__).parent / "certificates"
    cert_dir.mkdir(exist_ok=True)
    
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    pfx_file = cert_dir / "cert.pfx"
    
    # Check if certificates already exist
    if cert_file.exists() and key_file.exists():
        print("✅ SSL certificates already exist!")
        print(f"   Certificate: {cert_file}")
        print(f"   Private Key: {key_file}")
        
        overwrite = input("\nDo you want to regenerate them? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("✅ Using existing certificates.")
            return True
    
    print("\n🔐 Generating self-signed SSL certificates...")
    print("=" * 60)
    
    try:
        # Generate private key and certificate
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:4096",
            "-keyout", str(key_file),
            "-out", str(cert_file),
            "-days", "365",
            "-nodes",
            "-subj", "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Error generating certificates with OpenSSL")
            print(result.stderr)
            print("\n⚠️  OpenSSL not found. Trying alternative method...")
            generate_with_cryptography()
            return True
        
        print("✅ Certificate generated successfully!")
        print(f"   Certificate: {cert_file}")
        print(f"   Private Key: {key_file}")
        print(f"   Valid for: 365 days")
        print("\n⚠️  Note: Browsers will show a security warning (expected for self-signed certs)")
        print("   Click 'Advanced' → 'Proceed to localhost' to continue")
        
        return True
        
    except FileNotFoundError:
        print("⚠️  OpenSSL not found. Trying alternative method...")
        return generate_with_cryptography()
    except Exception as e:
        print(f"❌ Error: {e}")
        return generate_with_cryptography()

def generate_with_cryptography():
    """Generate certificates using Python cryptography library (fallback)"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime
        
        print("\n🔐 Generating certificates with Python cryptography library...")
        
        cert_dir = Path(__file__).parent / "certificates"
        cert_dir.mkdir(exist_ok=True)
        
        cert_file = cert_dir / "cert.pem"
        key_file = cert_dir / "key.pem"
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )
        
        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"State"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, u"City"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Voice Deepfake Detection"),
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(u"localhost"),
                x509.DNSName(u"127.0.0.1"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Write private key
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Write certificate
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        print("✅ Certificate generated successfully!")
        print(f"   Certificate: {cert_file}")
        print(f"   Private Key: {key_file}")
        print(f"   Valid for: 365 days")
        print("\n⚠️  Note: Browsers will show a security warning (expected for self-signed certs)")
        print("   Click 'Advanced' → 'Proceed to localhost' to continue")
        
        return True
        
    except ImportError:
        print("\n❌ Python cryptography library not installed.")
        print("   Install it with: pip install cryptography")
        print("\n   OR install OpenSSL:")
        print("   - Windows: Download from https://slproweb.com/products/Win32OpenSSL.html")
        print("   - Linux: sudo apt-get install openssl")
        print("   - macOS: brew install openssl")
        return False
    except Exception as e:
        print(f"❌ Error generating certificates: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SSL CERTIFICATE GENERATOR")
    print("=" * 60)
    
    success = generate_certificates()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Setup complete! You can now run the application.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ Certificate generation failed.")
        print("=" * 60)
        sys.exit(1)
