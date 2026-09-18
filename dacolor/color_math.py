from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def srgb_channel_to_linear(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def rgb_to_xyz(rgb: Sequence[int]) -> tuple[float, float, float]:
    r, g, b = [srgb_channel_to_linear(v / 255.0) for v in rgb]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return x, y, z


def _xyz_f(value: float) -> float:
    delta = 6 / 29
    if value > delta**3:
        return value ** (1 / 3)
    return value / (3 * delta**2) + 4 / 29


def rgb_to_lab(rgb: Sequence[int]) -> tuple[float, float, float]:
    x, y, z = rgb_to_xyz(rgb)
    xn, yn, zn = 0.95047, 1.0, 1.08883
    fx = _xyz_f(x / xn)
    fy = _xyz_f(y / yn)
    fz = _xyz_f(z / zn)
    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return l, a, b


def ciede2000(lab1: Sequence[float], lab2: Sequence[float]) -> float:
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2

    avg_lp = (l1 + l2) / 2.0
    c1 = math.sqrt(a1**2 + b1**2)
    c2 = math.sqrt(a2**2 + b2**2)
    avg_c = (c1 + c2) / 2.0

    g = 0.5 * (1 - math.sqrt((avg_c**7) / (avg_c**7 + 25**7))) if avg_c else 0.0
    a1p = (1 + g) * a1
    a2p = (1 + g) * a2
    c1p = math.sqrt(a1p**2 + b1**2)
    c2p = math.sqrt(a2p**2 + b2**2)
    avg_cp = (c1p + c2p) / 2.0

    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    delta_lp = l2 - l1
    delta_cp = c2p - c1p

    if c1p * c2p == 0:
        delta_hp = 0.0
    else:
        dh = h2p - h1p
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        delta_hp = dh
    delta_hp_term = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(delta_hp / 2))

    if c1p * c2p == 0:
        avg_hp = h1p + h2p
    else:
        dh = abs(h1p - h2p)
        if dh > 180:
            avg_hp = (h1p + h2p + 360) / 2 if (h1p + h2p) < 360 else (h1p + h2p - 360) / 2
        else:
            avg_hp = (h1p + h2p) / 2

    t = (
        1
        - 0.17 * math.cos(math.radians(avg_hp - 30))
        + 0.24 * math.cos(math.radians(2 * avg_hp))
        + 0.32 * math.cos(math.radians(3 * avg_hp + 6))
        - 0.20 * math.cos(math.radians(4 * avg_hp - 63))
    )

    delta_ro = 30 * math.exp(-(((avg_hp - 275) / 25) ** 2))
    rc = 2 * math.sqrt((avg_cp**7) / (avg_cp**7 + 25**7)) if avg_cp else 0.0
    sl = 1 + ((0.015 * ((avg_lp - 50) ** 2)) / math.sqrt(20 + ((avg_lp - 50) ** 2)))
    sc = 1 + 0.045 * avg_cp
    sh = 1 + 0.015 * avg_cp * t
    rt = -math.sin(math.radians(2 * delta_ro)) * rc

    kl = kc = kh = 1.0
    return math.sqrt(
        (delta_lp / (kl * sl)) ** 2
        + (delta_cp / (kc * sc)) ** 2
        + (delta_hp_term / (kh * sh)) ** 2
        + rt * (delta_cp / (kc * sc)) * (delta_hp_term / (kh * sh))
    )


def build_ciede2000_matrix(labs: Iterable[Sequence[float]]) -> np.ndarray:
    labs = list(labs)
    size = len(labs)
    matrix = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(i, size):
            value = ciede2000(labs[i], labs[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix
