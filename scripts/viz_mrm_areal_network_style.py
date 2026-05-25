"""
Network-style visualizations for MRM and the Areal Decoder.

These figures use feature-map cuboids, convolution plates, routing masks, and
multi-scale context panels instead of rectangular flowchart blocks.

Outputs:
  paper_figures/mrm_network_style.{png,pdf,svg}
  paper_figures/areal_decoder_network_style.{png,pdf,svg}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon


OUT_DIR = Path(__file__).parent.parent / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORTS = (("pdf", None), ("png", 260), ("svg", None))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

COLORS = {
    "c1": "#D9ECFF",
    "c2": "#BFDDF6",
    "c3": "#9FC9E8",
    "c4": "#7BAED6",
    "linear": "#D9F99D",
    "areal": "#FFD8A8",
    "mask": "#B2F2BB",
    "mask_inv": "#FFE3C2",
    "conv": "#E9D8FD",
    "ppm": "#FFE8A3",
    "out": "#F9C9C9",
    "edge": "#2B2B2B",
    "muted": "#6B6B6B",
}


def cuboid(ax, x, y, w, h, d=0.20, color="#DDDDDD", alpha=1.0,
           label=None, label_size=8, lw=1.0, edge=None):
    edge = edge or COLORS["edge"]
    front = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    side = np.array([[x + w, y], [x + w + d, y + d], [x + w + d, y + h + d], [x + w, y + h]])
    top = np.array([[x, y + h], [x + d, y + h + d], [x + w + d, y + h + d], [x + w, y + h]])
    rgb = np.array(plt.matplotlib.colors.to_rgb(color))
    for pts, shade in ((side, 0.82), (top, 1.08), (front, 1.0)):
        ax.add_patch(Polygon(
            pts, closed=True, facecolor=np.clip(rgb * shade, 0, 1),
            edgecolor=edge, linewidth=lw, alpha=alpha, joinstyle="round",
        ))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=label_size, color="#111111")


def feature_stack(ax, x, y, w, h, depth=5, color="#DDDDDD", label=None,
                  label_size=8, spread=0.045):
    for i in range(depth - 1, -1, -1):
        cuboid(ax, x + i * spread, y + i * spread, w, h, d=0.15,
               color=color, alpha=0.96, lw=0.9)
    if label:
        ax.text(x + w / 2 + 0.15, y - 0.16, label, ha="center", va="top",
                fontsize=label_size, color=COLORS["edge"])


def conv_plates(ax, x, y, n=3, h=0.68, color=None, label="", label_y=-0.12):
    color = color or COLORS["conv"]
    for i in range(n):
        cuboid(ax, x + i * 0.07, y + i * 0.05, 0.22, h, d=0.055,
               color=color, lw=0.85)
    if label:
        ax.text(x + 0.17, y + label_y, label, ha="center", va="top",
                fontsize=7.5, color=COLORS["muted"])


def arrow(ax, p0, p1, rad=0.0, lw=1.25, color=None, label=None,
          label_offset=(0.0, 0.0), style="-|>", alpha=1.0):
    color = color or COLORS["edge"]
    arr = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=13, linewidth=lw,
        color=color, alpha=alpha, connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)
    if label:
        ax.text((p0[0] + p1[0]) / 2 + label_offset[0],
                (p0[1] + p1[1]) / 2 + label_offset[1],
                label, fontsize=7.5, color=COLORS["muted"],
                ha="center", va="center")


def draw_mask(ax, x, y, w=1.25, h=0.92, inverse=False, label="alpha"):
    yy, xx = np.mgrid[0:1:80j, 0:1:120j]
    mask = (
        0.30
        + 0.45 * np.exp(-((yy - 0.50) ** 2) / 0.018)
        + 0.22 * np.exp(-((xx - yy - 0.08) ** 2) / 0.014)
        + 0.08 * np.sin(10 * xx) * np.cos(7 * yy)
    )
    mask = np.clip(mask, 0, 1)
    if inverse:
        mask = 1.0 - mask
    cmap = "YlGn" if not inverse else "Oranges"
    ax.imshow(mask, extent=(x, x + w, y, y + h), origin="lower",
              cmap=cmap, interpolation="bilinear", zorder=1)
    ax.add_patch(Polygon([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                         closed=True, fill=False, edgecolor=COLORS["edge"],
                         linewidth=1.0, zorder=2))
    ax.text(x + w / 2, y - 0.12, label, ha="center", va="top",
            fontsize=7.6, color=COLORS["edge"])


def draw_ppm(ax, x, y):
    base_w, base_h = 1.05, 0.85
    scales = [("1x1", 0.00, 0.54), ("2x2", 0.46, 0.34), ("4x4", 0.88, 0.18), ("8x8", 1.27, 0.04)]
    for label, dx, dy in scales:
        cuboid(ax, x + dx, y + dy, base_w - dx * 0.30, base_h - dy * 0.35,
               d=0.10, color=COLORS["ppm"], lw=0.85)
        ax.text(x + dx + 0.32, y + dy + 0.23, label, fontsize=7,
                ha="center", color=COLORS["edge"])
    ax.text(x + 0.83, y - 0.12, "PPM scales\n1, 2, 4, 8",
            ha="center", va="top", fontsize=7.5, color=COLORS["edge"])


def save(fig, name):
    for ext, dpi in EXPORTS:
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def panel_mrm():
    fig, ax = plt.subplots(figsize=(15.5, 7.7))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 7.7)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(7.75, 7.35, "MRM Network Visualization",
            ha="center", fontsize=18)
    ax.text(7.75, 7.02,
            "C3 morphology router predicts alpha, then splits every backbone stage into linear and areal streams",
            ha="center", fontsize=9, color=COLORS["muted"])

    # Backbone pyramid.
    stages = [
        ("C1", "96 x H/4 x W/4", 0.90, 0.90, 1.30, 1.10, COLORS["c1"]),
        ("C2", "192 x H/8 x W/8", 1.10, 2.25, 1.08, 0.92, COLORS["c2"]),
        ("C3", "384 x H/16 x W/16", 1.32, 3.43, 0.90, 0.78, COLORS["c3"]),
        ("C4", "768 x H/32 x W/32", 1.55, 4.45, 0.72, 0.64, COLORS["c4"]),
    ]
    for name, shape, x, y, w, h, color in stages:
        feature_stack(ax, x, y, w, h, depth=5, color=color)
        ax.text(0.68, y + h * 0.50, f"{name}\n{shape}", ha="right",
                va="center", fontsize=7.4, color=COLORS["edge"])
    ax.text(1.35, 0.28, "backbone features", ha="center",
            fontsize=8.5, color=COLORS["muted"])

    # MRM branch from C3.
    arrow(ax, (2.45, 3.85), (3.25, 5.35), rad=0.05, label="C3/H16")
    conv_plates(ax, 3.45, 5.25, n=4, h=0.62, color="#D9F99D",
                label="linear branch\n1x15 -> 15x1", label_y=-0.18)
    feature_stack(ax, 4.30, 5.18, 0.78, 0.62, depth=4,
                  color=COLORS["linear"], label="line-sensitive\nmid channels", label_size=7.2)

    arrow(ax, (2.45, 3.85), (3.25, 3.95), rad=-0.02)
    conv_plates(ax, 3.45, 3.72, n=4, h=0.62, color="#FFD8A8",
                label="areal branch\n7x7", label_y=-0.18)
    feature_stack(ax, 4.30, 3.65, 0.78, 0.62, depth=4,
                  color=COLORS["areal"], label="area-sensitive\nmid channels", label_size=7.2)

    arrow(ax, (5.25, 5.48), (6.10, 4.85), rad=-0.10, label="concat")
    arrow(ax, (5.25, 3.95), (6.10, 4.48), rad=0.10)
    conv_plates(ax, 6.25, 4.40, n=3, h=0.68, color="#E9D8FD",
                label="router 1x1\nsigmoid", label_y=-0.20)
    draw_mask(ax, 7.05, 4.34, label="alpha\nlinear weight")

    # Broadcast alpha to all stages.
    draw_mask(ax, 8.65, 5.18, w=1.05, h=0.72, label="alpha up/down")
    draw_mask(ax, 8.65, 3.96, w=1.05, h=0.72, inverse=True, label="1-alpha")
    arrow(ax, (8.30, 4.85), (8.62, 5.45), color="#2F9E44")
    arrow(ax, (8.30, 4.50), (8.62, 4.26), color="#E8590C")

    # Streamed outputs.
    stream_y = [0.95, 2.25, 3.45, 4.55]
    for i, (name, _, _x, _y, w, h, _c) in enumerate(stages):
        yy = stream_y[i]
        feature_stack(ax, 10.25, yy + 0.50, w * 0.72, h * 0.72, depth=4,
                      color=COLORS["linear"], label=f"{name} * alpha", label_size=7.0)
        feature_stack(ax, 12.35, yy + 0.15, w * 0.72, h * 0.72, depth=4,
                      color=COLORS["areal"], label=f"{name} * (1-alpha)", label_size=7.0)
        arrow(ax, (2.55, _y + h * 0.48), (10.20, yy + 0.90),
              rad=-0.06, color="#2F9E44", alpha=0.78)
        arrow(ax, (2.55, _y + h * 0.40), (12.30, yy + 0.55),
              rad=0.06, color="#E8590C", alpha=0.78)
    ax.text(10.72, 6.48, "linear_feats -> DSA Decoder",
            ha="center", fontsize=8.5, color="#2B8A3E")
    ax.text(13.18, 6.12, "areal_feats -> Areal Decoder",
            ha="center", fontsize=8.5, color="#D9480F")

    ax.text(7.75, 0.22,
            "alpha is clamped to [0.1, 0.9] in TMDSSegmentor before routing all feature stages.",
            ha="center", fontsize=8, color=COLORS["muted"], style="italic")
    save(fig, "mrm_network_style")


def panel_areal():
    fig, ax = plt.subplots(figsize=(16.0, 8.0))
    ax.set_xlim(0, 16.0)
    ax.set_ylim(0, 8.0)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(8.0, 7.55, "Areal Decoder Network Visualization",
            ha="center", fontsize=18)
    ax.text(8.0, 7.20,
            "areal-routed feature pyramid -> PPM global context -> top-down FPN -> H/4 concat -> F_A",
            ha="center", fontsize=9, color=COLORS["muted"])

    stages = [
        ("A1", "C1*(1-alpha)\nH/4", 0.82, 0.95, 1.28, 1.08, COLORS["c1"]),
        ("A2", "C2*(1-alpha)\nH/8", 1.05, 2.35, 1.06, 0.90, COLORS["c2"]),
        ("A3", "C3*(1-alpha)\nH/16", 1.28, 3.55, 0.88, 0.76, COLORS["c3"]),
        ("A4", "C4*(1-alpha)\nH/32", 1.52, 4.60, 0.72, 0.62, COLORS["c4"]),
    ]
    for name, label, x, y, w, h, color in stages:
        feature_stack(ax, x, y, w, h, depth=5, color=color)
        ax.text(0.62, y + h * 0.48, f"{name}\n{label}", ha="right",
                va="center", fontsize=7.3, color=COLORS["edge"])

    # PPM on deepest stage.
    arrow(ax, (2.45, 4.95), (3.25, 5.28), rad=0.02, label="deepest")
    draw_ppm(ax, 3.40, 4.75)
    feature_stack(ax, 5.45, 4.80, 0.86, 0.68, depth=5,
                  color=COLORS["ppm"], label="PPM out\nC x H/32 x W/32", label_size=7.4)
    arrow(ax, (5.05, 5.22), (5.42, 5.15))

    # Lateral and top-down FPN for C3/C2/C1.
    lat_y = [1.35, 2.67, 3.82]
    out_y = [1.35, 2.67, 3.82, 4.80]
    for idx, ly in enumerate(lat_y):
        conv_plates(ax, 3.45, ly, n=4, h=0.62, color="#E9D8FD",
                    label="lateral 1x1" if idx == 0 else "")
        arrow(ax, (2.43, stages[idx][3] + stages[idx][5] * 0.50), (3.42, ly + 0.34))
        conv_plates(ax, 5.45, ly, n=4, h=0.62, color="#E9D8FD",
                    label="FPN 3x3" if idx == 0 else "")
        arrow(ax, (3.92, ly + 0.34), (5.42, ly + 0.34), label="lat + top")

    feature_stack(ax, 6.35, 3.75, 0.78, 0.62, depth=4,
                  color=COLORS["areal"], label="P3", label_size=7.2)
    feature_stack(ax, 6.35, 2.62, 0.88, 0.70, depth=4,
                  color=COLORS["areal"], label="P2", label_size=7.2)
    feature_stack(ax, 6.35, 1.32, 1.02, 0.82, depth=4,
                  color=COLORS["areal"], label="P1", label_size=7.2)
    feature_stack(ax, 6.35, 4.85, 0.68, 0.54, depth=4,
                  color=COLORS["areal"], label="P4", label_size=7.2)

    arrow(ax, (6.28, 5.08), (5.42, 4.15), rad=-0.10, label="up")
    arrow(ax, (6.72, 4.02), (5.42, 2.98), rad=-0.10, label="up")
    arrow(ax, (6.72, 2.96), (5.42, 1.68), rad=-0.10, label="up")

    # Upsample all P levels to H/4 and concatenate.
    concat_x, concat_y = 9.15, 2.05
    for i, yy in enumerate(out_y):
        arrow(ax, (7.28, yy + 0.34), (concat_x, concat_y + 0.90 - i * 0.18),
              rad=-0.08 + 0.04 * i, color="#2F9E44",
              label="up to H/4" if i == 3 else None)
    feature_stack(ax, concat_x, concat_y, 1.28, 1.12, depth=8,
                  color="#D8F3DC", label="concat\n4C x H/4 x W/4", label_size=7.7)

    conv_plates(ax, 11.15, 2.30, n=5, h=0.84, color="#FFE8CC",
                label="bottleneck 3x3\nGN + ReLU", label_y=-0.20)
    arrow(ax, (10.65, 2.62), (11.12, 2.72))
    feature_stack(ax, 12.55, 2.12, 1.16, 1.00, depth=6,
                  color=COLORS["out"], label="F_A\nC x H/4 x W/4", label_size=8.4)
    arrow(ax, (11.68, 2.70), (12.52, 2.62))

    ax.text(12.35, 4.75, "Areal stream focuses on region-like defects\nleakage, falling-off, segment damage",
            ha="center", fontsize=8.5, color=COLORS["muted"])
    ax.text(8.0, 0.30,
            "ArealDecoder.forward: PPM(C4), top-down FPN over C3/C2/C1, concat at H/4, then bottleneck.",
            ha="center", fontsize=8, color=COLORS["muted"], style="italic")
    save(fig, "areal_decoder_network_style")


def main():
    panel_mrm()
    panel_areal()
    print("Generated:")
    for name in ("mrm_network_style", "areal_decoder_network_style"):
        for ext, _ in EXPORTS:
            p = OUT_DIR / f"{name}.{ext}"
            print(f"  {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
