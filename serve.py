"""
serve.py — Run from project root: python serve.py
Serves frontend at http://localhost:3000
Fixes CSS/JS path issues for pages in subfolders.
"""
import http.server
import socketserver
import os

PORT = 3000
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def log_message(self, format, *args):
        # Only log non-404s to keep output clean
        if args[1] != '404':
            super().log_message(format, *args)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"✅ MediGraph AI Frontend running at http://localhost:{PORT}")
    print(f"   Serving from: {FRONTEND_DIR}")
    print(f"   Press Ctrl+C to stop")
    httpd.serve_forever()