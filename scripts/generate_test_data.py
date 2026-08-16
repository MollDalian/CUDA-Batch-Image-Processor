#!/usr/bin/env python3
"""Generate test PPM images for the CUDA image processing pipeline.

Pure Python -- no external dependencies beyond the standard library.
Creates a variety of patterns (gradients, checkerboards, circles, stripes,
sine waves, and random noise) so the edge-detection pipeline has interesting
inputs to work with.

Usage:
    python3 generate_test_data.py [output_dir] [count] [size]

Defaults: 100 images, 128x128 pixels, written to data/
"""

import math
import os
import random
import struct
import sys


def write_ppm(filename, width, height, pixels):
    """Write an RGB pixel list as a binary PPM (P6) file."""
    with open(filename, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        buf = bytearray()
        for r, g, b in pixels:
            buf.append(max(0, min(255, int(r))))
            buf.append(max(0, min(255, int(g))))
            buf.append(max(0, min(255, int(b))))
        f.write(bytes(buf))


# ---------- pattern generators ----------

def gradient(w, h):
    return [
        (int(255 * x / w), int(255 * y / h), int(255 * (x + y) / (w + h)))
        for y in range(h) for x in range(w)
    ]


def checkerboard(w, h, cell=16):
    return [
        (240, 200, 50) if ((x // cell) + (y // cell)) % 2 == 0 else (50, 50, 200)
        for y in range(h) for x in range(w)
    ]


def circles(w, h):
    cx, cy = w / 2, h / 2
    pixels = []
    for y in range(h):
        for x in range(w):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            r = int(127.5 * (1 + math.sin(d * 0.15)))
            g = int(127.5 * (1 + math.sin(d * 0.15 + 2.094)))
            b = int(127.5 * (1 + math.sin(d * 0.15 + 4.189)))
            pixels.append((r, g, b))
    return pixels


def stripes(w, h, horizontal=True, stripe_w=12):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    return [
        colors[((y if horizontal else x) // stripe_w) % len(colors)]
        for y in range(h) for x in range(w)
    ]


def sine_wave(w, h, fx=0.12, fy=0.08):
    pixels = []
    for y in range(h):
        for x in range(w):
            r = int(127.5 * (1 + math.sin(x * fx)))
            g = int(127.5 * (1 + math.sin(y * fy)))
            b = int(127.5 * (1 + math.sin((x + y) * 0.09)))
            pixels.append((r, g, b))
    return pixels


def noise(w, h, seed):
    random.seed(seed)
    return [(random.randint(0, 255), random.randint(0, 255),
             random.randint(0, 255)) for _ in range(w * h)]


# ---------- main ----------

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    os.makedirs(output_dir, exist_ok=True)

    named = [
        ("gradient",    lambda w, h: gradient(w, h)),
        ("checker",     lambda w, h: checkerboard(w, h)),
        ("circles",     lambda w, h: circles(w, h)),
        ("stripes_h",   lambda w, h: stripes(w, h, True)),
        ("stripes_v",   lambda w, h: stripes(w, h, False)),
        ("sine_wave",   lambda w, h: sine_wave(w, h)),
    ]

    print(f"Generating {count} test images ({size}x{size}) in {output_dir}/")

    for i in range(count):
        if i < len(named):
            name, gen = named[i]
            pixels = gen(size, size)
        else:
            name = "noise"
            pixels = noise(size, size, seed=i)

        filename = os.path.join(output_dir, f"{name}_{i:04d}.ppm")
        write_ppm(filename, size, size, pixels)

        if (i + 1) % 25 == 0 or i + 1 == count:
            print(f"  {i + 1}/{count}")

    print("Done.")


if __name__ == "__main__":
    main()
