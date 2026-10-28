#!/usr/bin/env python3
"""Compare two global-bias updates for RM(2, 5) over the BSC."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from rm25_bec_sweep import affine_three_flats


def check_supports() -> np.ndarray:
    supports = [
        [point for point in range(32) if int(mask) & (1 << point)]
        for mask in affine_three_flats()
    ]
    result = np.asarray(supports, dtype=np.intp)
    if result.shape != (620, 8):
        raise RuntimeError("unexpected check-support array")
    return result


def extrinsic_bias(values: np.ndarray) -> np.ndarray:
    """For each entry, multiply the other seven entries in its row."""
    left = np.ones_like(values)
    right = np.ones_like(values)
    left[:, 1:] = np.cumprod(values[:, :-1], axis=1)
    right[:, :-1] = np.cumprod(values[:, :0:-1], axis=1)[:, ::-1]
    return left * right


def decisions(beta: np.ndarray, tie_bits: np.ndarray) -> np.ndarray:
    return (beta < 0) | ((beta == 0) & tie_bits)


def simulate_p(
    p: float,
    trials: int,
    attempts: int,
    supports: np.ndarray,
    rng: np.random.Generator,
    clip_epsilon: float,
) -> dict[str, float | int]:
    channel_errors = rng.random((trials, 32)) < p
    initial = (1.0 - 2.0 * p) * (1.0 - 2.0 * channel_errors)
    heuristic = initial.copy()
    bp_style = initial.copy()
    rows = np.arange(trials)[:, None]

    for _ in range(attempts):
        selected = supports[rng.integers(0, len(supports), size=trials)]

        old_heuristic = heuristic[rows, selected]
        gamma_heuristic = extrinsic_bias(old_heuristic)
        new_heuristic = (
            old_heuristic
            + (1.0 - np.abs(old_heuristic)) * gamma_heuristic
        )
        heuristic[rows, selected] = np.clip(new_heuristic, -1.0, 1.0)

        old_bp_style = bp_style[rows, selected]
        gamma_bp_style = extrinsic_bias(old_bp_style)
        denominator = 1.0 + old_bp_style * gamma_bp_style
        new_bp_style = (old_bp_style + gamma_bp_style) / denominator
        bp_style[rows, selected] = np.clip(
            new_bp_style,
            -1.0 + clip_epsilon,
            1.0 - clip_epsilon,
        )

    tie_bits = rng.integers(0, 2, size=(trials, 32), dtype=np.int8).astype(bool)
    heuristic_errors = decisions(heuristic, tie_bits)
    bp_style_errors = decisions(bp_style, tie_bits)
    return {
        "p": p,
        "trials": trials,
        "attempts": attempts,
        "heuristic_bit_error": float(np.mean(heuristic_errors)),
        "heuristic_block_error": float(np.mean(np.any(heuristic_errors, axis=1))),
        "bp_style_bit_error": float(np.mean(bp_style_errors)),
        "bp_style_block_error": float(np.mean(np.any(bp_style_errors, axis=1))),
    }


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_svg(
    path: Path,
    rows: list[dict[str, float | int]],
    seed: int,
) -> None:
    width, height = 1500, 760
    panel_top, panel_bottom = 180, 100
    panel_width, panel_height = 570, height - panel_top - panel_bottom
    panel_lefts = (130, 850)
    colors = {"heuristic": "#c0392b", "bp_style": "#1769aa"}
    labels = {"heuristic": "Heuristic", "bp_style": "BP-style combination"}

    def x_position(p: float, left: int) -> float:
        return left + (p / 0.5) * panel_width

    def y_position(value: float) -> float:
        return panel_top + (1.0 - value) * panel_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: DejaVu Sans, sans-serif; fill: #17202a; }</style>',
        '<text x="750" y="46" text-anchor="middle" font-size="30" font-weight="bold">RM(2, 5) randomized soft decoder on BSC(p)</text>',
        f'<text x="750" y="82" text-anchor="middle" font-size="19">1,000 checks per block; 10,000 blocks per p; paired simulation; seed {seed}</text>',
    ]

    metrics = (
        ("bit_error", "Bit error probability"),
        ("block_error", "Block error probability"),
    )
    for left, (metric, title) in zip(panel_lefts, metrics):
        for tick in range(11):
            value = tick / 10
            y = y_position(value)
            elements.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + panel_width}" y2="{y:.2f}" '
                'stroke="#d5d8dc" stroke-width="1"/>'
            )
            elements.append(
                f'<text x="{left - 16}" y="{y + 6:.2f}" text-anchor="end" '
                f'font-size="16">{value:.1f}</text>'
            )
        for tick in range(11):
            p = tick * 0.05
            x = x_position(p, left)
            if tick % 2 == 0:
                elements.append(
                    f'<line x1="{x:.2f}" y1="{panel_top}" x2="{x:.2f}" '
                    f'y2="{panel_top + panel_height}" stroke="#edf0f2" stroke-width="1"/>'
                )
                elements.append(
                    f'<text x="{x:.2f}" y="{panel_top + panel_height + 32}" '
                    f'text-anchor="middle" font-size="16">{p:.1f}</text>'
                )
        elements.extend([
            f'<line x1="{left}" y1="{panel_top}" x2="{left}" y2="{panel_top + panel_height}" stroke="#17202a" stroke-width="2"/>',
            f'<line x1="{left}" y1="{panel_top + panel_height}" x2="{left + panel_width}" y2="{panel_top + panel_height}" stroke="#17202a" stroke-width="2"/>',
            f'<text x="{left + panel_width / 2:.2f}" y="{panel_top - 24}" text-anchor="middle" font-size="23" font-weight="bold">{title}</text>',
            f'<text x="{left + panel_width / 2:.2f}" y="{height - 35}" text-anchor="middle" font-size="21">BSC crossover probability p</text>',
        ])

        for method in ("heuristic", "bp_style"):
            key = f"{method}_{metric}"
            points = " ".join(
                f'{x_position(float(row["p"]), left):.2f},{y_position(float(row[key])):.2f}'
                for row in rows
            )
            elements.append(
                f'<polyline points="{points}" fill="none" stroke="{colors[method]}" '
                'stroke-width="4" stroke-linejoin="round"/>'
            )
            for row in rows:
                x = x_position(float(row["p"]), left)
                y = y_position(float(row[key]))
                elements.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" '
                    f'fill="{colors[method]}" stroke="white" stroke-width="1.5"/>'
                )

    legend_y = 112
    for index, method in enumerate(("heuristic", "bp_style")):
        legend_x = 560 + index * 300
        elements.extend([
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 45}" y2="{legend_y}" stroke="{colors[method]}" stroke-width="5"/>',
            f'<text x="{legend_x + 58}" y="{legend_y + 7}" font-size="18">{labels[method]}</text>',
        ])

    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--attempts", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--clip-epsilon", type=float, default=1e-12)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("0929/rm25_bsc_compare"),
    )
    args = parser.parse_args()

    supports = check_supports()
    rng = np.random.default_rng(args.seed)
    rows = []
    for step in range(11):
        p = step * 0.05
        row = simulate_p(
            p,
            args.trials,
            args.attempts,
            supports,
            rng,
            args.clip_epsilon,
        )
        rows.append(row)
        print(
            f'p={p:.2f}  '
            f'heuristic bit/block={row["heuristic_bit_error"]:.6f}/'
            f'{row["heuristic_block_error"]:.6f}  '
            f'BP-style bit/block={row["bp_style_bit_error"]:.6f}/'
            f'{row["bp_style_block_error"]:.6f}',
            flush=True,
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix.with_suffix(".csv"), rows)
    write_svg(args.output_prefix.with_suffix(".svg"), rows, args.seed)


if __name__ == "__main__":
    main()
