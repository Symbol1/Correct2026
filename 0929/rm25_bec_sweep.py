#!/usr/bin/env python3
"""Monte Carlo sweep for the randomized RM(2, 5) erasure decoder."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def span_mask(basis: tuple[int, int, int]) -> int:
    mask = 0
    for coefficients in range(8):
        vector = 0
        for index, basis_vector in enumerate(basis):
            if coefficients & (1 << index):
                vector ^= basis_vector
        mask |= 1 << vector
    return mask


def affine_three_flats() -> np.ndarray:
    linear_flats: set[int] = set()
    for first in range(1, 32):
        for second in range(1, 32):
            if second == first:
                continue
            span_two = {0, first, second, first ^ second}
            for third in range(1, 32):
                if third not in span_two:
                    linear_flats.add(span_mask((first, second, third)))

    affine_flats: set[int] = set()
    for flat in linear_flats:
        points = [point for point in range(32) if flat & (1 << point)]
        for translation in range(32):
            translated = 0
            for point in points:
                translated |= 1 << (point ^ translation)
            affine_flats.add(translated)

    if len(linear_flats) != 155 or len(affine_flats) != 620:
        raise RuntimeError("unexpected number of 3-flats")
    return np.array(sorted(affine_flats), dtype=np.uint32)


def simulate(
    p: float,
    trials: int,
    attempts: int,
    checks: np.ndarray,
    rng: np.random.Generator,
) -> int:
    if p == 0.0:
        return 0
    if p == 1.0:
        return trials

    powers = np.left_shift(np.uint32(1), np.arange(32, dtype=np.uint32))
    erased = rng.random((trials, 32)) < p
    states = np.sum(erased * powers, axis=1, dtype=np.uint32)

    for _ in range(attempts):
        selected = checks[rng.integers(0, len(checks), size=trials)]
        intersections = states & selected
        singleton = (intersections != 0) & (
            (intersections & (intersections - np.uint32(1))) == 0
        )
        states[singleton] ^= intersections[singleton]
        if not np.any(states):
            break

    return int(np.count_nonzero(states))


def wilson_interval(errors: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    estimate = errors / trials
    denominator = 1.0 + z * z / trials
    center = (estimate + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        estimate * (1.0 - estimate) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_svg(
    path: Path,
    rows: list[dict[str, float | int]],
    attempts: int,
    seed: int,
) -> None:
    width, height = 1400, 900
    left, right, top, bottom = 140, 70, 120, 120
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_position(value: float) -> float:
        return left + value * plot_width

    def y_position(value: float) -> float:
        return top + (1.0 - value) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: DejaVu Sans, sans-serif; fill: #17202a; }</style>',
        '<text x="700" y="48" text-anchor="middle" font-size="30" font-weight="bold">RM(2, 5) randomized erasure decoder on BEC(p)</text>',
        f'<text x="700" y="84" text-anchor="middle" font-size="19">{attempts:,} random weight-8 checks per block; {int(rows[0]["trials"]):,} trials per p; seed {seed}</text>',
    ]

    for tick in range(11):
        value = tick / 10
        x = x_position(value)
        y = y_position(value)
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#d5d8dc" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 18}" y="{y + 7:.2f}" text-anchor="end" '
            f'font-size="18">{value:.1f}</text>'
        )
        elements.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" '
            'stroke="#edf0f2" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{height - bottom + 35}" text-anchor="middle" '
            f'font-size="18">{value:.1f}</text>'
        )

    elements.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#17202a" stroke-width="2"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#17202a" stroke-width="2"/>',
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 45}" text-anchor="middle" font-size="23">BEC erasure probability p</text>',
        f'<text x="42" y="{top + plot_height / 2:.2f}" text-anchor="middle" font-size="23" transform="rotate(-90 42 {top + plot_height / 2:.2f})">Block error probability</text>',
    ])

    points = " ".join(
        f'{x_position(float(row["p"])):.2f},{y_position(float(row["estimate"])):.2f}'
        for row in rows
    )
    elements.append(
        f'<polyline points="{points}" fill="none" stroke="#1769aa" '
        'stroke-width="5" stroke-linejoin="round"/>'
    )

    for row in rows:
        x = x_position(float(row["p"]))
        y = y_position(float(row["estimate"]))
        low = y_position(float(row["ci_lower"]))
        high = y_position(float(row["ci_upper"]))
        elements.extend([
            f'<line x1="{x:.2f}" y1="{high:.2f}" x2="{x:.2f}" y2="{low:.2f}" stroke="#1769aa" stroke-width="2"/>',
            f'<line x1="{x - 7:.2f}" y1="{high:.2f}" x2="{x + 7:.2f}" y2="{high:.2f}" stroke="#1769aa" stroke-width="2"/>',
            f'<line x1="{x - 7:.2f}" y1="{low:.2f}" x2="{x + 7:.2f}" y2="{low:.2f}" stroke="#1769aa" stroke-width="2"/>',
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#1769aa" stroke="white" stroke-width="2"/>',
        ])

    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--attempts", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("0929/rm25_bec_sweep"),
    )
    args = parser.parse_args()

    checks = affine_three_flats()
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int]] = []
    for step in range(11):
        p = step / 10
        errors = simulate(p, args.trials, args.attempts, checks, rng)
        estimate = errors / args.trials
        if p in (0.0, 1.0):
            lower = upper = estimate
        else:
            lower, upper = wilson_interval(errors, args.trials)
        row: dict[str, float | int] = {
            "p": p,
            "errors": errors,
            "trials": args.trials,
            "estimate": estimate,
            "ci_lower": lower,
            "ci_upper": upper,
        }
        rows.append(row)
        print(
            f"p={p:.1f}  errors={errors:>{len(str(args.trials))},}/{args.trials:,}"
            f"  estimate={estimate:.6f}  95% CI=[{lower:.6f}, {upper:.6f}]",
            flush=True,
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix.with_suffix(".csv"), rows)
    write_svg(args.output_prefix.with_suffix(".svg"), rows, args.attempts, args.seed)


if __name__ == "__main__":
    main()
