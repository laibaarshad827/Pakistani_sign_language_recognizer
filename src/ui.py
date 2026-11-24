"""Screen drawing for the live trainer (OpenCV only, no camera code)."""
import os
import textwrap

import cv2
import numpy as np

from learner import practice_level

REFERENCE_DIR = "data/reference"
PANEL_WIDTH = 330
HEADER = 46
FONT = cv2.FONT_HERSHEY_DUPLEX

# colours are (blue, green, red)
BG = (52, 34, 26)
CARD = (84, 58, 44)
WHITE, GREY = (255, 255, 255), (185, 175, 165)
TEAL, PURPLE, PINK = (180, 205, 0), (220, 80, 150), (170, 110, 255)
YELLOW, ORANGE, GREEN, RED = (0, 215, 255), (0, 150, 255), (110, 210, 70), (80, 80, 235)
LEVEL_COLORS = {"Practise a lot": RED, "Practise more": ORANGE, "Almost there": YELLOW, "Well done": GREEN,
                "Not tried yet": GREY}


def load_reference(sign, size=220):
    """Load the reference picture of a sign; merged names like 'haa+hamza' use the first part."""
    for name in (sign, sign.split("+")[0]):
        path = os.path.join(REFERENCE_DIR, f"{name}.png")
        if os.path.exists(path):
            image = cv2.imread(path)
            if image is not None:
                return cv2.resize(image, (size, size))
    return None


def put(canvas, text, position, scale=0.6, color=WHITE, thickness=1):
    """Write text on the canvas."""
    cv2.putText(canvas, text, position, FONT, scale, color, thickness, cv2.LINE_AA)


def put_centered(canvas, text, centre_x, y, scale, color, thickness=2):
    """Write text centred horizontally around centre_x."""
    width = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
    put(canvas, text, (int(centre_x - width / 2), y), scale, color, thickness)


def gradient(width, height, left, right):
    """A picture that fades smoothly from one colour (left) to another (right)."""
    ramp = np.linspace(0.0, 1.0, width)[None, :, None]
    row = np.array(left)[None, None, :] * (1 - ramp) + np.array(right)[None, None, :] * ramp
    return np.repeat(row, height, axis=0).astype(np.uint8)


def bar(canvas, x, y, width, fraction, color, height=14):
    """Draw a horizontal progress bar."""
    fraction = max(0.0, min(1.0, fraction))
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (95, 75, 62), -1)
    if fraction > 0:
        cv2.rectangle(canvas, (x, y), (x + int(width * fraction), y + height), color, -1)


def score_color(score):
    """Red for a low score, yellow in the middle, green for a high one."""
    return GREEN if score >= 60 else YELLOW if score >= 30 else RED


def draw_header(canvas, info):
    """Colourful title bar with one dot per round (green = passed, red = missed)."""
    canvas[:HEADER, :] = gradient(canvas.shape[1], HEADER, TEAL, PURPLE)
    put(canvas, "PSL TRAINER", (15, 31), 0.85, WHITE, 2)
    put(canvas, "learn Pakistan Sign Language", (190, 31), 0.5, (240, 240, 240))
    if info is None:
        return
    rounds, results = info["rounds"], info["results"]
    for i in range(min(rounds, 20)):
        x = canvas.shape[1] - 18 - (min(rounds, 20) - 1 - i) * 24
        if i < len(results):
            cv2.circle(canvas, (x, 23), 8, GREEN if results[i] else RED, -1)
        elif i == len(results) and info["state"] != "finished":
            cv2.circle(canvas, (x, 23), 8, YELLOW, 2)
        else:
            cv2.circle(canvas, (x, 23), 6, (225, 225, 225), 1)


def banner(canvas, text, color, width):
    """A coloured banner with big text across the top of the camera picture."""
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, HEADER + 18), (width, HEADER + 92), color, -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    put_centered(canvas, text, width / 2, HEADER + 72, 1.5, WHITE, 3)


def render(frame, info, reference):
    """Return the full screen: camera picture on the left, trainer panel on the right."""
    height, width = frame.shape[:2]
    canvas = np.full((HEADER + height, width + PANEL_WIDTH, 3), BG, dtype=np.uint8)
    if info["state"] == "finished":
        return render_summary(canvas, info)
    canvas[HEADER:, :width] = frame
    draw_header(canvas, info)

    # coloured frame around the camera picture shows how it is going
    border = GREY
    if info["state"] == "playing":
        border = YELLOW if info["hold_fraction"] > 0 else TEAL
    elif info["state"] == "result":
        border = GREEN if info["last_success"] else ORANGE
    cv2.rectangle(canvas, (0, HEADER), (width - 1, HEADER + height - 1), border, 5)
    if info["state"] == "result":
        if info["last_success"]:
            banner(canvas, "CORRECT!", GREEN, width)
        else:
            banner(canvas, "KEEP PRACTISING", ORANGE, width)

    x = width + 15
    put(canvas, "TARGET SIGN", (x, HEADER + 22), 0.45, GREY)
    put(canvas, str(info["target"]), (x, HEADER + 58), 1.1, YELLOW, 2)
    top = HEADER + 68
    if reference is not None:
        canvas[top:top + reference.shape[0], x:x + reference.shape[1]] = reference
    else:
        cv2.rectangle(canvas, (x, top), (x + 220, top + 220), CARD, -1)
        put(canvas, "no reference picture", (x + 22, top + 115), 0.5, GREY)
    cv2.rectangle(canvas, (x, top), (x + 220, top + 220), TEAL, 2)

    y = top + 250
    shown = info["predicted"] if info["predicted"] is not None else "-"
    matches = info["predicted"] == info["target"]
    put(canvas, f"You show: {shown}", (x, y), 0.6, GREEN if matches else WHITE)
    put(canvas, "Score", (x, y + 28), 0.5, GREY)
    bar(canvas, x + 70, y + 15, 220, info["score"] / 100, score_color(info["score"]))
    put(canvas, "Hold", (x, y + 52), 0.5, GREY)
    bar(canvas, x + 70, y + 39, 220, info["hold_fraction"], TEAL)
    put(canvas, "Time", (x, y + 76), 0.5, GREY)
    bar(canvas, x + 70, y + 63, 220, info["time_fraction"], ORANGE if info["time_fraction"] < 0.25 else PINK)

    colour = WHITE
    if info["state"] == "result":
        colour = GREEN if info["last_success"] else ORANGE
    for i, line in enumerate(textwrap.wrap(info["message"], 32)[:3]):
        put(canvas, line, (x, y + 106 + 22 * i), 0.55, colour)
    return canvas


def render_summary(canvas, info):
    """End-of-session screen: how it went and how much practice each sign needs."""
    summary = info["summary"]
    draw_header(canvas, info)
    total = max(1, summary["rounds"])
    accuracy = summary["correct"] / total

    put(canvas, "Session complete!", (30, 95), 1.2, YELLOW, 2)
    put(canvas, f"{summary['correct']} of {summary['rounds']} rounds correct", (30, 128), 0.7, WHITE)
    for label, x in (("SIGN", 30), ("RESULT", 170), ("MASTERY", 270), ("PRACTICE NEEDED", 430)):
        put(canvas, label, (x, 168), 0.45, GREY)
    shown = summary["rows"][:10]
    for i, (sign, wins, tries, mastery) in enumerate(shown):
        y = 200 + 30 * i
        level = practice_level(mastery)
        put(canvas, sign, (30, y), 0.6, WHITE)
        mark = GREEN if wins == tries else ORANGE if wins else RED
        cv2.circle(canvas, (180, y - 6), 8, mark, -1)
        put(canvas, f"{wins}/{tries}", (198, y), 0.5, WHITE)
        bar(canvas, 270, y - 14, 140, mastery, LEVEL_COLORS[level])
        put(canvas, level, (430, y), 0.5, LEVEL_COLORS[level])
    if len(summary["rows"]) > len(shown):
        put(canvas, f"+ {len(summary['rows']) - len(shown)} more signs", (30, 200 + 30 * len(shown)), 0.45, GREY)

    # right column: big result ring, verdict and the weakest signs overall
    centre = (800, 120)
    cv2.circle(canvas, centre, 62, CARD, 12)
    if accuracy > 0:
        cv2.ellipse(canvas, centre, (62, 62), 0, -90, -90 + 360 * accuracy, GREEN if accuracy >= 0.5 else ORANGE, 12)
    put_centered(canvas, f"{accuracy * 100:.0f}%", centre[0], centre[1] + 12, 1.0, WHITE, 2)
    if accuracy >= 0.8:
        verdict, colour = "Excellent! Keep it up.", GREEN
    elif accuracy >= 0.5:
        verdict, colour = "Good progress. Repeat the red and orange signs.", YELLOW
    else:
        verdict, colour = "Keep practising - the weak signs will come back.", ORANGE
    for i, line in enumerate(textwrap.wrap(verdict, 30)):
        put(canvas, line, (645, 225 + 24 * i), 0.55, colour)
    put(canvas, "PRACTISE THESE NEXT", (645, 300), 0.5, TEAL)
    for i, (sign, mastery, attempts) in enumerate(summary["weakest"]):
        y = 332 + 30 * i
        level = practice_level(mastery) if attempts else "Not tried yet"
        put(canvas, sign, (645, y), 0.55, WHITE)
        bar(canvas, 750, y - 13, 90, mastery, LEVEL_COLORS[level])
        put(canvas, level, (850, y), 0.42, LEVEL_COLORS[level])
    put(canvas, "Press q to close", (30, canvas.shape[0] - 14), 0.45, GREY)
    return canvas


class Celebration:
    """A confetti shower with sparkles, shown whenever a sign is passed."""

    COLORS = [YELLOW, PINK, TEAL, GREEN, ORANGE, WHITE, PURPLE]

    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.particles = []
        self.previous_state = None
        self.last_time = None

    def burst(self, x, y):
        """Throw a fan of confetti upwards from (x, y)."""
        for _ in range(150):
            angle = self.rng.uniform(-np.pi * 0.95, -np.pi * 0.05)
            speed = self.rng.uniform(160, 560)
            self.particles.append({
                "x": x, "y": y, "vx": speed * np.cos(angle), "vy": speed * np.sin(angle),
                "color": self.COLORS[int(self.rng.integers(len(self.COLORS)))],
                "size": int(self.rng.integers(3, 8)), "life": float(self.rng.uniform(1.4, 2.6)),
                "shape": int(self.rng.integers(3))})

    def draw(self, canvas, info, now):
        """Start a burst when a round has just been passed, then move and draw the confetti."""
        dt = 0.0 if self.last_time is None else min(0.1, now - self.last_time)
        self.last_time = now
        if self.previous_state == "playing" and info["state"] == "result" and info["last_success"]:
            self.burst((canvas.shape[1] - PANEL_WIDTH) / 2, HEADER + 250)
        self.previous_state = info["state"]
        alive = []
        for p in self.particles:
            p["vy"] += 650 * dt
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["life"] -= dt
            if p["life"] <= 0 or p["y"] > canvas.shape[0] + 10:
                continue
            x, y, s = int(p["x"]), int(p["y"]), p["size"]
            if p["shape"] == 0:
                cv2.rectangle(canvas, (x, y), (x + s, y + s + 2), p["color"], -1)
            elif p["shape"] == 1:
                cv2.circle(canvas, (x, y), max(2, s // 2 + 1), p["color"], -1)
            else:                                          # sparkle
                cv2.line(canvas, (x - s, y), (x + s, y), p["color"], 1)
                cv2.line(canvas, (x, y - s), (x, y + s), p["color"], 1)
            alive.append(p)
        self.particles = alive