"""Full-screen voice Pong. Run with Python 3.12+: python main.py."""

import math
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pygame

try:
    import sounddevice as sd
except ImportError:
    sd = None


SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
MIN_PITCH, MAX_PITCH = 75, 1100
COURT = (10, 15, 27)


def _dim(color, alpha=100):
    """Composite a color the way the old full-screen overlay did, at alpha 100."""
    mix = alpha / 255
    keep = 1 - mix
    return tuple(int(color[i] * mix + COURT[i] * keep) for i in range(3))


def pitch_position(pitch, neutral, height, paddle_height=0):
    """Map +/- seven semitones to the paddle's full available travel."""
    tone = max(-1., min(1., 12 * math.log2(pitch / neutral) / 7))
    return height / 2 - tone * (height - paddle_height) / 2


class VoiceVisuals:
    """Separate, turn-attributed frequency histories with software glow effects."""

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.bands = np.zeros((2, 48))
        self.energy = [0., 0.]
        self.history = [pygame.Surface((180, 48)), pygame.Surface((180, 48))]
        self.colors = ((74, 224, 226), (255, 123, 164))
        self.edges = np.geomspace(440 * 2 ** (-7 / 12), 440 * 2 ** (7 / 12), 49)
        self.dirty = [False, False]
        self.scaled = [None, None]
        self.panel_size = None

    def feed(self, samples, player, neutral=440.):
        self.edges = np.geomspace(neutral * 2 ** (-7 / 12), neutral * 2 ** (7 / 12), 49)
        signal = np.asarray(samples, dtype=float)
        signal = signal - signal.mean()
        energy = float(np.sqrt(np.mean(signal * signal)))
        fft_size = len(signal) * 4
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n=fft_size)) / len(signal)
        frequencies = np.fft.rfftfreq(fft_size, 1 / SAMPLE_RATE)
        # Interpolation keeps narrow note bands visible between FFT bins.
        centers = np.sqrt(self.edges[:-1] * self.edges[1:])
        values = np.interp(centers, frequencies, spectrum)
        values = np.clip((20 * np.log10(values + 1e-7) + 75) / 55, 0, 1)
        if energy < .008:
            values[:] = 0
        self.bands[player] = values
        self.energy[player] = min(1., energy * 12)
        self.dirty[player] = True
        # Each side records only samples received during its own turn.
        strip = self.history[player]
        strip.scroll(dx=-1)
        strip.fill((0, 0, 0), (179, 0, 1, 48))
        for band, value in enumerate(values):
            strip.set_at((179, 47 - band), tuple(int(c * value) for c in self.colors[player]))

    def _waterfall(self, player, size):
        """Smooth the spectrogram only when new audio arrives."""
        if size != self.panel_size:
            self.dirty = [self.dirty[i] or self.scaled[i] is not None for i in range(2)]
            self.panel_size = size
            self.scaled = [None, None]
        if self.dirty[player]:
            image = pygame.transform.smoothscale(self.history[player], size)
            # Mirror time so the newest audio sits on each outside edge.
            if player == 0:
                image = pygame.transform.flip(image, True, False)
            self.scaled[player] = image
            self.dirty[player] = False
        return self.scaled[player]

    def draw(self, screen, dt, active, paddles, now, pitch=None):
        w, h = self.width, self.height
        panel_size = (max(1, int(w * .48)), max(1, h))
        panel_width = panel_size[0]
        for player, color in enumerate(self.colors):
            level = self.energy[player]
            waterfall = self._waterfall(player, panel_size)
            if waterfall is not None:
                # 43 and 18 keep the old overlay brightness (source alpha, then the layer at 100).
                waterfall.set_alpha(43 if active == player else 18)
                screen.blit(waterfall, (0 if player == 0 else w - panel_width, 0))
            spacing = h / 48
            bar_h = max(1, int(spacing * .65))
            for band, value in enumerate(self.bands[player]):
                # Anchor the outer bands to the screen edges, including the last pixel.
                by = round((47 - band) * (h - bar_h) / 47)
                bar_width = max(1, int(value * panel_width))
                bx = 0 if player == 0 else w - bar_width
                rgb = tuple(int(c * (.3 + .7 * value)) for c in color)
                pygame.draw.rect(screen, _dim(rgb), (bx, by, bar_width, bar_h), border_radius=2)
                if value > .08:
                    glow = tuple(int(c * value * .45) for c in color)
                    pygame.draw.line(screen, _dim(glow), (bx, by), (bx + bar_width, by), 2)
            # A frequency-shaped halo radiates from the listening paddle.
            if level > .01:
                anchor = w * (.02 if player == 0 else .98)
                points = []
                for band, value in enumerate(self.bands[player]):
                    angle = -math.pi / 2 + band / 47 * math.pi
                    if player == 1:
                        angle += math.pi
                    radius = h * (.06 + value * .11)
                    points.append((int(anchor + math.cos(angle) * radius),
                                   int(paddles[player] + math.sin(angle) * radius)))
                pygame.draw.lines(screen, _dim(tuple(int(c * level * .6) for c in color)), False, points, 3)
            self.energy[player] *= math.exp(-dt * 6)
            self.bands[player] *= math.exp(-dt * 2.8)
            if player == active and pitch is not None:
                # The live note crosses the exact center of its paddle.
                y = round(paddles[player])
                start = 0 if player == 0 else w - panel_width
                line = tuple(min(255, int(c * 210 / 255)) for c in color)
                pygame.draw.line(screen, _dim(line), (start, y), (start + panel_width, y), 2)


def detect_pitch(samples, sample_rate=SAMPLE_RATE):
    """Return (fundamental Hz, RMS), rejecting quiet or unpitched audio."""
    signal = np.asarray(samples, dtype=np.float64).flatten()
    signal = signal - signal.mean()
    rms = float(np.sqrt(np.mean(signal * signal)))
    if rms < 0.008:
        return None, rms
    signal *= np.hanning(len(signal))
    spectrum = np.fft.rfft(signal, n=2 * len(signal))
    correlation = np.fft.irfft(spectrum * spectrum.conjugate())[:len(signal)]
    low, high = int(sample_rate / MAX_PITCH), int(sample_rate / MIN_PITCH)
    # First strong local peak avoids choosing a multiple of the true period.
    peaks = np.flatnonzero(
        (correlation[low:high] > correlation[low - 1:high - 1])
        & (correlation[low:high] >= correlation[low + 1:high + 1])
    ) + low
    if not len(peaks):
        return None, rms
    strongest = float(correlation[peaks].max())
    if strongest < correlation[0] * 0.5:
        return None, rms
    lag = int(peaks[correlation[peaks] >= strongest * 0.9][0])
    a, b, c = correlation[lag - 1:lag + 2]
    denominator = a - 2 * b + c
    offset = 0.5 * (a - c) / denominator if denominator else 0
    return sample_rate / (lag + offset), rms


class Microphone:
    def __init__(self):
        self.stream = None
        self.latest = None
        self.error = ""
        self.start()

    def start(self):
        self.close()
        self.latest = None
        if sd is None:
            self.error = "Microphone unavailable: install sounddevice (see requirements.txt)."
            return
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE,
                dtype="float32", callback=self.callback,
            )
            self.stream.start()
            self.error = ""
        except Exception as exc:
            self.error = f"Microphone unavailable: {exc}"
            self.close()

    def callback(self, indata, frames, timing, status):
        # Keep the audio thread light; pitch analysis runs in the game loop.
        self.latest = (indata[:, 0].copy(), time.monotonic())

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def main():
    pygame.init()
    try:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, vsync=1)
    except TypeError:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Voice Pong")
    pygame.mouse.set_visible(False)
    width, height = screen.get_size()
    scale = min(width / 1280, height / 720)
    font = pygame.font.SysFont("Bahnschrift,Segoe UI,Arial", max(16, int(22 * scale)))
    large = pygame.font.SysFont("Bahnschrift,Segoe UI,Arial", max(32, int(68 * scale)), bold=True)
    small = pygame.font.SysFont("Cascadia Code,Consolas", max(12, int(14 * scale)))
    label = pygame.font.SysFont("Bahnschrift,Segoe UI,Arial", max(13, int(16 * scale)), bold=True)
    visuals = VoiceVisuals(width, height)
    clock = pygame.time.Clock()
    from phone_server import PhoneServer
    phones = None
    phone_error = ""
    try:
        phones = PhoneServer()
    except Exception as exc:
        phone_error = f"Phone server unavailable: {exc}. Install requirements.txt and restart."
    mic = None  # PC microphone is opt-in with L.
    lobby = True
    show_codes = False
    phone_processed = [None, None]
    phone_history = [deque(maxlen=2), deque(maxlen=2)]
    phone_pitch = [None, None]
    qr_surfaces = []
    if phones:
        for matrix in phones.qrs:
            unit = max(2, int(min(width * .33, height * .37) / len(matrix)))
            surface = pygame.Surface((len(matrix) * unit, len(matrix) * unit))
            surface.fill((255, 255, 255))
            for row, cells in enumerate(matrix):
                for col, dark in enumerate(cells):
                    if dark:
                        surface.fill((0, 0, 0), (col * unit, row * unit, unit, unit))
            qr_surfaces.append(surface)
    cyan, pink = (74, 224, 226), (255, 123, 164)
    white, muted = (235, 241, 255), (135, 150, 176)
    # The HUD overlays the court so gameplay uses the full screen height.
    top, bottom = 0, height
    paddle_w = max(12, width * .012)
    paddle_percent = 100
    speed_percent = 100
    points_to_win = 7
    left_x, right_x = width * .02, width * .98
    radius = max(7, int(10 * scale))
    paddles = [(top + bottom) / 2] * 2
    scores = [0, 0]
    ball = pygame.Vector2(width / 2, (top + bottom) / 2)
    direction = pygame.Vector2(1, .35).normalize()
    serve_ramp_seconds = 3.0
    speed = width * .29 * .08
    active = 1
    serve_at = time.monotonic() + 3
    paused = False
    menu_open = False
    menu_index = 0
    paused_before_menu = False
    match_over = False
    match_winner = 0
    running = True
    neutral = 440.0
    pitch = None
    rms = 0
    processed = None
    history = deque(maxlen=2)
    calibrating_until = 0
    calibration = []
    notice = ""

    def text(message, x, y, color=white, face=font, alpha=100):
        surface = face.render(message, True, color)
        surface.set_alpha(alpha)
        screen.blit(surface, surface.get_rect(center=(int(x), int(y))))

    def paddle_height():
        return max(1, round((bottom - top) * .23 * paddle_percent / 100))

    def cruise_speed():
        return width * .29 * speed_percent / 100

    def serve_start_speed():
        return cruise_speed() * .08

    def serve(receiver):
        nonlocal active, speed, direction, serve_at
        active = receiver
        speed = serve_start_speed()
        ball.update(width / 2, (top + bottom) / 2)
        direction = pygame.Vector2(1 if receiver else -1, random.uniform(-.45, .45)).normalize()
        serve_at = time.monotonic() + 2

    def serve_avoiding(receiver):
        """End-screen serve aimed off the paddle, so the receiver has to move to catch it."""
        nonlocal active, speed, direction, serve_at
        active = receiver
        speed = cruise_speed() * .65
        ball.update(width / 2, height / 2)
        paddle_x = left_x if receiver == 0 else right_x
        dx = paddle_x - ball.x
        max_dy = abs(dx) * .42
        above = max(radius, ball.y - max_dy)
        below = min(height - radius, ball.y + max_dy)
        miss_y = above if abs(above - paddles[receiver]) >= abs(below - paddles[receiver]) else below
        direction = pygame.Vector2(dx, miss_y - ball.y)
        if direction.length_squared() == 0:
            direction.update(-1 if receiver == 0 else 1, 0)
        direction = direction.normalize()
        serve_at = time.monotonic() + .45

    def finish_match(winner):
        nonlocal match_over, match_winner
        match_over = True
        match_winner = winner
        serve_avoiding(1 - winner)

    def reset_match():
        nonlocal match_over, paused
        scores[:] = [0, 0]
        match_over = False
        half = (top + bottom) / 2
        paddles[:] = [half, half]
        paused = False
        serve(1)

    def adjust_setting(index, delta):
        nonlocal paddle_percent, speed_percent, points_to_win
        if index == 0:
            paddle_percent = min(200, max(50, paddle_percent + delta * 10))
            half = paddle_height() / 2
            for player in range(2):
                paddles[player] = max(top + half, min(bottom - half, paddles[player]))
        elif index == 1:
            speed_percent = min(200, max(50, speed_percent + delta * 10))
        else:
            points_to_win = min(21, max(1, points_to_win + delta))
            if match_over:
                return
            leader = 0 if scores[0] > scores[1] else 1
            if scores[leader] > scores[1 - leader] and scores[leader] >= points_to_win:
                finish_match(leader)

    def open_menu():
        nonlocal menu_open, paused, paused_before_menu
        if menu_open:
            return
        paused_before_menu = paused
        paused = True
        menu_open = True

    def close_menu():
        nonlocal menu_open, paused
        menu_open = False
        paused = paused_before_menu

    try:
        while running:
            dt = min(clock.tick(240) / 1000, .04)
            now = time.monotonic()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if menu_open:
                        if event.key in (pygame.K_ESCAPE, pygame.K_TAB, pygame.K_RETURN):
                            close_menu()
                        elif event.key in (pygame.K_UP, pygame.K_w):
                            menu_index = (menu_index - 1) % 3
                        elif event.key in (pygame.K_DOWN, pygame.K_s):
                            menu_index = (menu_index + 1) % 3
                        elif event.key in (pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d):
                            adjust_setting(menu_index, -1 if event.key in (pygame.K_LEFT, pygame.K_a) else 1)
                        continue
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_TAB:
                        open_menu()
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        reset_match()
                    elif event.key == pygame.K_m:
                        if mic:
                            mic.start()
                    elif event.key == pygame.K_l:
                        if mic is None:
                            mic = Microphone()
                        lobby = False
                        serve(1)
                    elif event.key == pygame.K_q:
                        show_codes = not show_codes
                        lobby = show_codes
                    elif event.key == pygame.K_c:
                        calibration = []
                        calibrating_until = now + 3
                        notice = ""

            ready = phones is not None and all(phones.connected(i) for i in range(2))
            if lobby and ready and not show_codes:
                lobby = False
                serve(1)
            waiting = mic is None and not ready
            if phones:
                phones.active = active
                phones.paused = paused or lobby or waiting or menu_open
                for player in range(2):
                    packet = phones.latest[player]
                    if packet is not None and packet[1] != phone_processed[player]:
                        phone_processed[player] = packet[1]
                        visuals.feed(packet[0], player, neutral)
                        detected, rms = detect_pitch(packet[0])
                        if detected is None:
                            phone_history[player].clear()
                            phone_pitch[player] = None
                        else:
                            phone_history[player].append(detected)
                            phone_pitch[player] = float(np.median(phone_history[player]))
                    if packet is None or now - packet[1] > .3:
                        phone_pitch[player] = None
            latest = mic.latest if mic else None
            if latest is not None and latest[1] != processed:
                processed = latest[1]
                visuals.feed(latest[0], active, neutral)
                detected, rms = detect_pitch(latest[0])
                if detected is None:
                    history.clear()
                    pitch = None
                else:
                    history.append(detected)
                    pitch = float(np.median(history))
                    if calibrating_until > now:
                        calibration.append(detected)
            if latest is None or now - latest[1] > .3:
                pitch = None
            if calibrating_until and now >= calibrating_until:
                calibrating_until = 0
                if calibration:
                    neutral = float(np.median(calibration))
                    notice = f"Neutral tone set to {neutral:.0f} Hz"
                else:
                    notice = "No clear tone heard. Press C and hum again."

            if mic is None:
                pitch = phone_pitch[active]
                if pitch and calibrating_until > now:
                    calibration.append(pitch)
            keys = pygame.key.get_pressed()
            if not paused and not calibrating_until and not lobby and not waiting:
                movement = 0.0
                paddle_h = paddle_height()
                if pitch:
                    target = pitch_position(pitch, neutral, height, paddle_h)
                    follow = 1 - math.exp(-dt / .016)
                    paddles[active] += (target - paddles[active]) * follow
                keyboard = (keys[pygame.K_s] - keys[pygame.K_w]) if active == 0 else (keys[pygame.K_DOWN] - keys[pygame.K_UP])
                if keyboard:
                    movement = keyboard
                paddles[active] = max(top + paddle_h / 2, min(bottom - paddle_h / 2, paddles[active] + movement * height * .65 * dt))
                if now >= serve_at:
                    # Ease into each serve so the receiver can read its direction.
                    base_speed = cruise_speed()
                    progress = min(1.0, (now - serve_at) / serve_ramp_seconds)
                    if match_over:
                        speed = base_speed * .65
                    elif progress < 1.0:
                        ease = progress * progress * (3 - 2 * progress)
                        speed = serve_start_speed() + (base_speed - serve_start_speed()) * ease
                    else:
                        # Continue the normal rally acceleration after the serve.
                        speed = min(width * .95 * speed_percent / 100, max(base_speed, speed) + width * .006 * speed_percent / 100 * dt)
                    # Substeps prevent a fast ball from skipping a paddle.
                    steps = max(1, math.ceil(speed * dt / radius))
                    for _ in range(steps):
                        ball += direction * speed * dt / steps
                        if ball.y - radius <= top:
                            ball.y = top + radius
                            direction.y = abs(direction.y)
                        elif ball.y + radius >= bottom:
                            ball.y = bottom - radius
                            direction.y = -abs(direction.y)
                        for player, x in enumerate((left_x, right_x)):
                            approaching = direction.x < 0 if player == 0 else direction.x > 0
                            if approaching and abs(ball.x - x) <= paddle_w / 2 + radius and abs(ball.y - paddles[player]) <= paddle_h / 2 + radius:
                                if match_over:
                                    reset_match()
                                    break
                                offset = max(-1, min(1, (ball.y - paddles[player]) / (paddle_h / 2)))
                                angle = offset * math.radians(60)
                                direction.update(math.cos(angle) * (1 if player == 0 else -1), math.sin(angle))
                                ball.x = x + (1 if player == 0 else -1) * (paddle_w / 2 + radius + 1)
                                active = 1 - player
                        else:
                            if ball.x < -radius or ball.x > width + radius:
                                loser = 0 if ball.x < 0 else 1
                                if match_over:
                                    serve_avoiding(1 - loser)
                                else:
                                    scores[1 - loser] += 1
                                    if scores[1 - loser] >= points_to_win:
                                        finish_match(1 - loser)
                                    else:
                                        serve(loser)
                                break
                            continue
                        break
            else:
                serve_at += dt

            screen.fill(COURT)
            visuals.draw(screen, dt, active, paddles, now, pitch)
            text("V O I C E   /   P O N G", width / 2, height * .045, muted, label)
            text(f"{scores[0]}     :     {scores[1]}", width / 2, height * .125, white, large)
            color = cyan if active == 0 else pink
            if not match_over:
                text(f"{'LEFT' if active == 0 else 'RIGHT'} PLAYER — YOUR MIC TURN", width / 2, height * .195, color)
                for player, center in enumerate((width * .23, width * .77)):
                    tint = (cyan, pink)[player]
                    text(f"0{player + 1}  /  {'LISTENING' if active == player else 'STANDBY'}", center, height * .59,
                         tint if active == player else muted, label)
                    text("+7 / TOP  ·  NEUTRAL / CENTER  ·  -7 / BOTTOM", center, height * .625, muted, small)
            pygame.draw.line(screen, (23, 29, 42), (left_x, top), (right_x, top), 2)
            pygame.draw.line(screen, (23, 29, 42), (left_x, bottom - 1), (right_x, bottom - 1), 2)
            for y in range(int(top + 10), int(bottom), max(12, int(28 * scale))):
                pygame.draw.line(screen, (21, 27, 39), (width / 2, y), (width / 2, y + 10), 2)
            for player, x in enumerate((left_x, right_x)):
                rect = pygame.Rect(0, 0, paddle_w, paddle_height())
                rect.centerx = round(x)
                rect.top = round(paddles[player] - rect.height / 2)
                pygame.draw.rect(screen, (cyan, pink)[player] if active == player else (65, 76, 98), rect, border_radius=6)
            pygame.draw.circle(screen, white, (round(ball.x), round(ball.y)), radius)
            if calibrating_until:
                text("Hum a comfortable steady note…", width / 2, height * .44, white, large)
            elif waiting and not lobby:
                text("Phone disconnected — press Q for QR codes", width / 2, height * .44, white, font)
            elif match_over:
                side = "LEFT" if match_winner == 0 else "RIGHT"
                text(f"{side} WINS", width / 2, height * .40, (cyan, pink)[match_winner], large, 230)
                text("Hum onto the ball to play again", width / 2, height * .50, white, font, 230)
                text("R: reset", width / 2, height * .56, muted, small, 230)
            elif paused and not menu_open:
                text("PAUSED", width / 2, height * .44, white, large)
            elif now < serve_at and not menu_open:
                text(str(max(1, math.ceil(serve_at - now))), width / 2, height * .44, white, large)
            tone = f"{pitch:.0f} Hz" if pitch else "Hum or sing a clear tone"
            text(f"LOW = DOWN     •     {tone}     •     HIGH = UP", width / 2, height * .885, color)
            message = (mic.error if mic else "") or notice or f"Neutral: {neutral:.0f} Hz  |  Phone microphones connected"
            # Fit long device errors on screen.
            if len(message) > 110:
                message = message[:107] + "…"
            text(message, width / 2, height * .93, muted, small)
            text("Tab: menu   •   C: calibrate   •   Space: pause   •   R: reset   •   Q: QR   •   L: PC mic   •   Esc: quit", width / 2, height * .975, muted, small)
            if lobby:
                screen.fill(COURT)
                text("VOICE PONG / CONNECT PHONES", width / 2, height * .08, white, font)
                text("Same Wi-Fi as this computer. Scan one code per player, then enable the microphone.", width / 2, height * .16, muted, small)
                if phones:
                    for player, center in enumerate((width * .26, width * .74)):
                        text(f"PLAYER {player + 1}", center, height * .24, (cyan, pink)[player], font)
                        image = qr_surfaces[player]
                        screen.blit(image, image.get_rect(midtop=(int(center), int(height * .29))))
                        text("CONNECTED" if phones.connected(player) else "Scan and allow microphone", center, height * .73, (cyan, pink)[player], font)
                        # Keep the full manual URL readable even on smaller displays.
                        url_font = pygame.font.SysFont("Consolas", max(9, int(12 * scale)))
                        text(phones.urls[player], center, height * .79, muted, url_font)
                    text("Both microphones ready = automatic start. Reconnect either phone using its same code.", width / 2, height * .86, white, small)
                    text("HTTP over local Wi-Fi. Allow Python through Private network firewall.", width / 2, height * .91, muted, small)
                else:
                    text(phone_error[:130], width / 2, height * .5, pink, small)
                text("L: play with PC microphone instead   |   Tab: menu   |   Esc: quit", width / 2, height * .97, muted, small)
            if menu_open:
                panel = pygame.Surface((int(width * .72), int(height * .62)), pygame.SRCALPHA)
                panel.fill((0, 0, 0, 230))
                panel_rect = panel.get_rect(center=(width / 2, height / 2))
                screen.blit(panel, panel_rect)
                pygame.draw.rect(screen, white, panel_rect, 2, border_radius=16)
                text("MENU", width / 2, panel_rect.top + height * .08, white, font, 255)
                rows = (
                    ("PADDLE SIZE", f"{paddle_percent}%"),
                    ("BALL SPEED", f"{speed_percent}%"),
                    ("FIRST TO", str(points_to_win)),
                )
                for index, (row_label, value) in enumerate(rows):
                    text(f"{row_label}      {value}", width / 2, panel_rect.top + height * (.20 + index * .12),
                         cyan if index == menu_index else white, font, 255)
                text("Up/Down: choose     Left/Right: change     Enter: done", width / 2, panel_rect.bottom - height * .07, muted, small, 255)
            pygame.display.flip()
    finally:
        if mic:
            mic.close()
        if phones:
            phones.close()
        pygame.mouse.set_visible(True)
        pygame.quit()


def watch_and_run():
    """Restart the full-screen game after edits, until the player quits."""
    source = Path(__file__).resolve()

    def revision():
        try:
            return source.read_bytes()
        except OSError:
            return None

    child = None
    try:
        while True:
            current = revision()
            child = subprocess.Popen([sys.executable, str(source), "--play"])
            while child.poll() is None:
                time.sleep(.2)
                updated = revision()
                if updated is None or updated == current:
                    continue
                # Wait for the editor to finish saving before restarting.
                time.sleep(.3)
                if revision() != updated:
                    continue
                try:
                    compile(updated, str(source), "exec")
                except SyntaxError:
                    continue
                child.terminate()
                child.wait()
                break
            else:
                return child.returncode
    except KeyboardInterrupt:
        return 0
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()


if __name__ == "__main__":
    if "--play" in sys.argv:
        main()
    else:
        sys.exit(watch_and_run())
