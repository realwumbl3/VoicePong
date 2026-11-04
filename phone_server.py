"""Local HTTP phone microphones; audio is processed in memory only."""
import json
import os
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

import numpy as np


class PhoneServer:
    def __init__(self):
        import qrcode
        self.latest = [None, None]
        self.active = 1
        self.paused = True
        self.tokens = [secrets.token_urlsafe(18) for _ in range(2)]
        self.lock = threading.Lock()
        self.error = ''
        self.ip = os.environ.get('VOICE_PONG_HOST')
        if not self.ip:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                try:
                    probe.connect(('192.0.2.1', 80))
                    self.ip = probe.getsockname()[0]
                except OSError:
                    self.ip = socket.gethostbyname(socket.gethostname())
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body, content_type='application/json'):
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def player(self):
                token = parse_qs(urlsplit(self.path).query).get('token', [''])[0]
                return owner.tokens.index(token) if token in owner.tokens else None

            def do_GET(self):
                player = self.player()
                if player is None:
                    self.reply(403, b'Invalid player link', 'text/plain')
                elif urlsplit(self.path).path == '/status':
                    self.reply(200, json.dumps({'active': owner.active == player, 'paused': owner.paused}).encode())
                else:
                    page = Path(__file__).with_name('phone.html').read_text(encoding='utf-8')
                    self.reply(200, page.replace('__PLAYER__', str(player + 1)).encode(), 'text/html; charset=utf-8')

            def do_POST(self):
                player = self.player()
                if player is None or urlsplit(self.path).path != '/audio':
                    self.reply(403, b'{}')
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    rate = int(parse_qs(urlsplit(self.path).query).get('rate', ['0'])[0])
                    if length < 1024 or length > 32768 or length % 4 or not 8000 <= rate <= 96000:
                        raise ValueError()
                    samples = np.frombuffer(self.rfile.read(length), dtype='<f4').copy()
                    if not np.isfinite(samples).all():
                        raise ValueError()
                    # Normalize every phone's sample rate for the existing pitch/visual pipeline.
                    count = round(len(samples) * 44100 / rate)
                    samples = np.interp(np.arange(count) * rate / 44100, np.arange(len(samples)), samples)
                    with owner.lock:
                        owner.latest[player] = (samples, time.monotonic())
                    self.reply(200, b'{}')
                except (ValueError, OSError):
                    self.reply(400, b'{}')

        self.http = ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('VOICE_PONG_PORT', '8765'))), Handler)
        self.http.daemon_threads = True
        try:
            self.urls = [f'http://{self.ip}:{self.http.server_port}/?token={token}' for token in self.tokens]
            self.qrs = []
            for url in self.urls:
                qr = qrcode.QRCode(border=4)
                qr.add_data(url)
                qr.make(fit=True)
                self.qrs.append(qr.get_matrix())
            self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
            self.thread.start()
        except Exception:
            self.http.server_close()
            raise

    def connected(self, player):
        latest = self.latest[player]
        return latest is not None and time.monotonic() - latest[1] < 2

    def close(self):
        self.http.shutdown()
        self.http.server_close()
