"""Synthetic sheets and surfaces for tests and demos (no network)."""
from __future__ import annotations

import numpy as np


def parallel_sheets(shape=(80, 140, 140), spacing=10.0, z0=0.0, sigma=1.5, noise=0.0, seed=0,
                    base=20.0, amp=200.0):
    """Volume (z, y, x) with bright sheets at z = z0 + k * spacing (planes)."""
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    ph = (z - z0) / spacing
    d = (ph - np.round(ph)) * spacing  # signed distance to nearest sheet
    vol = base + amp * np.exp(-0.5 * (d / sigma) ** 2)
    vol = np.broadcast_to(vol, shape).astype(np.float32).copy()
    if noise:
        vol += np.random.default_rng(seed).normal(0, noise, shape).astype(np.float32)
    return vol


def concentric_sheets(shape=(40, 200, 200), spacing=10.0, center=(100, 100), sigma=1.5, base=20.0, amp=200.0):
    """Volume with cylindrical sheets around the z axis (radius = k * spacing)."""
    y = np.arange(shape[1], dtype=np.float32)[:, None] - center[0]
    x = np.arange(shape[2], dtype=np.float32)[None, :] - center[1]
    r = np.sqrt(y * y + x * x)
    ph = r / spacing
    d = (ph - np.round(ph)) * spacing
    sl = base + amp * np.exp(-0.5 * (d / sigma) ** 2)
    return np.broadcast_to(sl[None], shape).astype(np.float32).copy()


def plane_surface(h=50, w=50, z=40.0, y0=10.0, x0=10.0, node=2.0):
    """tifxyz-style grid (h, w, 3) zyx: rows -> y, cols -> x, constant z."""
    r, c = np.mgrid[0:h, 0:w].astype(np.float32)
    zyx = np.stack([np.full_like(r, z), y0 + node * r, x0 + node * c], -1)
    return zyx


def smooth_bump(h, w, r0, r1, c0, c1, ramp):
    """1 inside [r0:r1, c0:c1], 0 outside, smooth cosine ramps of ``ramp`` nodes."""
    r = np.arange(h)[:, None].astype(np.float32)
    c = np.arange(w)[None, :].astype(np.float32)

    def edge(x, a, b):
        up = np.clip((x - (a - ramp)) / ramp, 0, 1)
        dn = np.clip(((b + ramp) - x) / ramp, 0, 1)
        f = np.minimum(up, dn)
        return 0.5 - 0.5 * np.cos(np.pi * f)
    return edge(r, r0, r1) * edge(c, c0, c1)


def cylinder_surface(radius=50.0, center=(100, 100), z_range=(5, 35), h=30, w=60, theta0=0.3, theta1=1.6):
    """Surface on the cylinder r = radius: rows -> z, cols -> angle."""
    zz = np.linspace(z_range[0], z_range[1], h, dtype=np.float32)
    th = np.linspace(theta0, theta1, w, dtype=np.float32)
    Z, T = np.meshgrid(zz, th, indexing="ij")
    return np.stack([Z, center[0] + radius * np.sin(T), center[1] + radius * np.cos(T)], -1)


def parallel_cos(shape=(80, 140, 140), period=10.0, z0=0.0, noise=0.0, seed=0):
    """Lasagna-style cos field: cos(2 pi (z - z0) / period), +1 on the sheets."""
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    vol = np.cos(2 * np.pi * (z - z0) / period)
    vol = np.broadcast_to(vol, shape).astype(np.float32).copy()
    if noise:
        vol += np.random.default_rng(seed).normal(0, noise, shape).astype(np.float32)
    return vol
