"""
HTTPS server for frontend using Python's built-in SSL support
"""
import http.server
import ssl
from pathlib import Path

# Configuration
PORT = 3000
CERT_FILE = Path(__file__).parent / "certificates" / "cert.pem"
KEY_FILE = Path(__file__).parent / "certificates" / "key.pem"
FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"

print("=" * 60)
print("HTTPS FRONTEND SERVER")
print("=" * 60)

# Check certificate files
if not CERT_FILE.exists() or not KEY_FILE.exists():
    print("\n[ERROR] SSL certificate not found!")
    print("   Run: python generate_ssl_cert.py")
    exit(1)

# Create HTTPS server
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

# Create server
httpd = http.server.HTTPServer(('0.0.0.0', PORT), Handler)

# Wrap with SSL
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(str(CERT_FILE), str(KEY_FILE))
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print(f"\n[OK] Frontend directory: {FRONTEND_DIR}")
print(f"[OK] Certificate: {CERT_FILE}")
print(f"[OK] Private key: {KEY_FILE}")
print("\n" + "=" * 60)
print(f"[HTTPS] Server running on port {PORT}")
print("=" * 60)
print(f"\n  Local:   https://localhost:{PORT}")
print(f"  Network: https://10.10.48.66:{PORT}")
print("\n[!] You'll see a security warning (self-signed cert)")
print("    Click 'Advanced' -> 'Proceed' to continue")
print("\n" + "=" * 60)
print("\nPress Ctrl+C to stop")
print()

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\n\nShutting down...")
    httpd.shutdown()
