"""
Network-style visualization for the DSA decoder.

This figure is intentionally different from the block-diagram schematic in
viz_dsa_architecture.py. It uses feature-map cuboids, convolution plates,
strip-sampling rays, and an attention matrix to resemble a neural-network
architecture visualization.

Outputs:
  paper_figures/dsa_network_style.{png,pdf,svg}
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
    "latent": "#E7D4FF",
    "fused": "#C9E8D0",
    "dsa": "#FFF1B8",
    "attn": "#FFD8A8",
    "out": "#F9C9C9",
    "edge": "#2B2B2B",
    "muted": "#6B6B6B",
}


def cuboid(ax, x, y, w, h, d=0.22, color="#DDDDDD", edge=None, alpha=1.0,
           label=None, label_size=8, label_offset=(0.0, 0.0), lw=1.1):
    """Draw a 2.5D feature-map cuboid."""
    edge = edge or COLORS["edge"]
    front = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    side = np.array([[x + w, y], [x + w + d, y + d], [x + w + d, y + h + d], [x + w, y + h]])
    top = np.array([[x, y + h], [x + d, y + h + d], [x + w + d, y + h + d], [x + w, y + h]])
    for pts, shade in ((side, 0.83), (top, 1.08), (front, 1.0)):
        rgb = np.array(plt.matplotlib.colors.to_rgb(color))
        face = np.clip(rgb * shade, 0, 1)
        ax.add_patch(Polygon(pts, closed=True, facecolor=face, edgecolor=edge,
                             linewidth=lw, alpha=alpha, joinstyle="round"))
    if label:
        ax.text(x + w / 2 + label_offset[0], y + h / 2 + label_offset[1],
                label, ha="center", va="center", fontsize=label_size,
                color="#111111")


def feature_stack(ax, x, y, w, h, depth=5, color="#DDDDDD", label=None,
                  label_size=8, spread=0.045):
    """Draw several shifted cuboids to suggest channel depth."""
    for i in range(depth - 1, -1, -1):
        cuboid(ax, x + i * spread, y + i * spread, w, h, d=0.16,
               color=color, alpha=0.96, lw=0.9)
    if label:
        ax.text(x + w / 2 + 0.16, y - 0.18, label, ha="center",
                va="top", fontsize=label_size, color=COLORS["edge"])


def conv_plates(ax, x, y, n=3, h=0.72, color="#F3E7FF", label="1x1"):
    """Draw thin plates used for conv/projection layers."""
    for i in range(n):
        cuboid(ax, x + i * 0.07, y + i * 0.05, 0.22, h, d=0.06,
               color=color, lw=0.9)
    ax.text(x + 0.18, y - 0.12, label, ha="center", va="top",
            fontsize=7.5, color=COLORS["muted"])


def arrow(ax, p0, p1, rad=0.0, lw=1.25, color=None, label=None,
          label_offset=(0.0, 0.0), style="-|>"):
    color = color or COLORS["edge"]
    arr = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=13, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)
    if label:
        mx = (p0[0] + p1[0]) / 2 + label_offset[0]
        my = (p0[1] + p1[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=7.5, color=COLORS["muted"],
                ha="center", va="center")


def draw_sampling_plane(ax, x, y, w=1.85, h=1.15):
    """Draw a small feature map with learned strip sampling rays."""
    cuboid(ax, x, y, w, h, d=0.12, color=COLORS["dsa"], lw=1.0)
    rng = np.random.default_rng(7)
    cx, cy = x + w * 0.52, y + h * 0.52
    colors = ["#D9480F", "#1971C2", "#2B8A3E", "#862E9C"]
    angles = np.deg2rad([8, 42, -35, 78])
    for i, ang in enumerate(angles):
        dx, dy = np.cos(ang) * 0.68, np.sin(ang) * 0.42
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
                color=colors[i], lw=2.0, solid_capstyle="round")
        pts = np.linspace(-1.0, 1.0, 8)
        jitter = rng.normal(0.0, 0.008, len(pts))
        ax.scatter(cx + dx * pts, cy + dy * pts + jitter, s=10,
                   color=colors[i], zorder=5)
    ax.text(x + w / 2, y - 0.16, "learned strip sampling\nnh=4, ns=4, M=8",
            ha="center", va="top", fontsize=7.5, color=COLORS["edge"])


def draw_attention_matrix(ax, x, y, w=1.28, h=1.0):
    """Draw an attention heatmap tile."""
    mat = np.array([
        [0.85, 0.25, 0.18, 0.08, 0.34, 0.16],
        [0.18, 0.70, 0.20, 0.30, 0.12, 0.08],
        [0.14, 0.26, 0.78, 0.18, 0.10, 0.22],
        [0.08, 0.20, 0.24, 0.82, 0.26, 0.18],
        [0.22, 0.12, 0.10, 0.28, 0.74, 0.20],
        [0.12, 0.08, 0.30, 0.16, 0.24, 0.80],
    ])
    ax.imshow(mat, extent=(x, x + w, y, y + h), cmap="YlOrRd",
              origin="lower", interpolation="nearest", zorder=1)
    ax.add_patch(Polygon([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                         closed=True, fill=False, edgecolor=COLORS["edge"],
                         linewidth=1.1, zorder=2))
    for i in range(1, 6):
        ax.plot([x + w * i / 6] * 2, [y, y + h], color="white", lw=0.5, alpha=0.8)
        ax.plot([x, x + w], [y + h * i / 6] * 2, color="white", lw=0.5, alpha=0.8)
    ax.text(x + w / 2, y - 0.16, "strip attention\nQ x sK^T -> softmax",
            ha="center", va="top", fontsize=7.5, color=COLORS["edge"])


def panel_network_style():
    fig, ax = plt.subplots(figsize=(18, 8.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(9, 7.85, "DSA Decoder Network Visualization",
            ha="center", va="center", fontsize=18)
    ax.text(9, 7.48,
            "feature pyramid -> fused linear stream -> deformable strip attention -> F_L",
            ha="center", va="center", fontsize=9.5, color=COLORS["muted"])

    # Backbone feature pyramid.
    pyramid = [
        ("C1", "96 x H/4 x W/4", 0.70, 0.95, 1.25, 1.18, COLORS["c1"]),
        ("C2", "192 x H/8 x W/8", 0.92, 2.55, 1.04, 0.98, COLORS["c2"]),
        ("C3", "384 x H/16 x W/16", 1.15, 3.90, 0.86, 0.80, COLORS["c3"]),
        ("C4", "768 x H/32 x W/32", 1.38, 5.05, 0.70, 0.66, COLORS["c4"]),
    ]
    for name, shape, x, y, w, h, color in pyramid:
        feature_stack(ax, x, y, w, h, depth=5, color=color)
        ax.text(0.56, y + h * 0.50, f"{name}\n{shape}",
                ha="right", va="center", fontsize=7.4, color=COLORS["edge"])

    ax.text(1.38, 0.38, "DINOv3 ConvNeXt stages", ha="center",
            fontsize=8.5, color=COLORS["muted"])

    # Lateral projections.
    lateral_y = [1.36, 2.86, 4.15, 5.28]
    for (_, _, x, y, w, h, _), ly in zip(pyramid, lateral_y):
        conv_plates(ax, 3.05, ly, label="lateral 1x1")
        arrow(ax, (x + w + 0.32, y + h * 0.55), (3.02, ly + 0.38))

    # Unified feature maps and top-down fusion.
    uni_y = [1.55, 3.18, 4.58, 5.74]
    for ly in uni_y:
        feature_stack(ax, 3.75, ly, 0.84, 0.72, depth=4,
                      color=COLORS["latent"], label="C channels", label_size=6.8)
    for a, b in zip(reversed(uni_y[1:]), reversed(uni_y[:-1])):
        arrow(ax, (4.70, a + 0.28), (4.70, b + 0.72), rad=0.0,
              label="up+add", label_offset=(0.42, 0.0), color="#6F42C1")

    # FPN convs and fused tensor.
    for ly in uni_y:
        conv_plates(ax, 5.65, ly + 0.02, label="FPN 3x3")
        arrow(ax, (4.83, ly + 0.40), (5.62, ly + 0.40))

    feature_stack(ax, 7.15, 1.55, 1.10, 1.08, depth=6,
                  color=COLORS["fused"], label="fused\nC x H/4 x W/4", label_size=8)
    for ly in uni_y:
        arrow(ax, (6.15, ly + 0.38), (7.12, 2.15), rad=-0.16,
              color="#2F9E44")

    # Downsample before DSA.
    feature_stack(ax, 8.95, 1.78, 0.92, 0.84, depth=5,
                  color="#D8F3DC", label="AvgPool x2\nC x H/8 x W/8", label_size=7.6)
    arrow(ax, (8.45, 2.18), (8.92, 2.18), label="downsample", label_offset=(0.0, 0.26))

    # DSA core as neural subgraph.
    x0, y0 = 9.95, 4.75
    feature_stack(ax, x0, y0, 0.86, 0.72, depth=5,
                  color=COLORS["dsa"], label="DSA input X", label_size=7.5)
    arrow(ax, (9.45, 2.58), (10.30, 4.72), rad=-0.20)

    # Q/K/V projections.
    q_pos = (11.25, 5.85)
    k_pos = (11.25, 4.78)
    v_pos = (11.25, 3.72)
    for pos, label in zip((q_pos, k_pos, v_pos), ("Q 1x1", "K 1x1", "V 1x1")):
        conv_plates(ax, pos[0], pos[1], n=4, h=0.58, color="#FFE8CC", label=label)
        arrow(ax, (10.93, 5.05), (pos[0] - 0.05, pos[1] + 0.31), rad=0.06)

    draw_sampling_plane(ax, 12.55, 4.40)
    draw_attention_matrix(ax, 12.86, 2.98)
    arrow(ax, (11.68, 4.98), (12.52, 4.94), label="K,V")
    arrow(ax, (10.62, 5.52), (12.58, 5.40), rad=0.10, label="global directions")
    arrow(ax, (11.65, 6.15), (12.84, 3.92), rad=-0.15, label="Q")
    arrow(ax, (13.50, 4.38), (13.50, 4.02), label="sK")
    arrow(ax, (13.30, 2.98), (12.25, 2.28), rad=0.08, label="attn")
    arrow(ax, (11.62, 3.98), (12.15, 2.22), rad=-0.12, label="sV")

    feature_stack(ax, 11.55, 1.72, 0.92, 0.74, depth=5,
                  color="#FFD8A8", label="attn x sV\nper head", label_size=7.4)
    feature_stack(ax, 10.28, 0.95, 1.02, 0.80, depth=6,
                  color="#FFE3B3", label="concat heads\nC x H/8 x W/8", label_size=7.3)
    arrow(ax, (11.55, 1.95), (11.33, 1.52), rad=0.0)

    conv_plates(ax, 12.35, 0.95, n=4, h=0.58, color="#FFE8CC", label="GroupNorm + out 1x1")
    arrow(ax, (11.44, 1.34), (12.32, 1.28))

    feature_stack(ax, 13.45, 0.93, 0.95, 0.78, depth=5,
                  color=COLORS["out"], label="Y + X\nresidual", label_size=7.5)
    arrow(ax, (12.86, 1.28), (13.42, 1.28))
    arrow(ax, (10.78, 4.78), (13.65, 1.75), rad=-0.45, color="#868E96",
          label="skip X", label_offset=(0.65, 0.42), style="-|>")

    feature_stack(ax, 14.72, 1.35, 1.05, 1.00, depth=6,
                  color=COLORS["out"], label="F_L\nC x H/4 x W/4", label_size=8.3)
    arrow(ax, (14.55, 1.42), (14.70, 1.70), label="upsample x2\nout_norm",
          label_offset=(0.37, 0.20))

    ax.text(12.35, 6.85, "Deformable Strip Attention core",
            fontsize=10.5, ha="center", color=COLORS["edge"])
    ax.text(9.0, 0.20,
            "This is a neural-network style visualization, not an execution graph. "
            "It follows DSADecoder.forward and DeformableStripAttention.forward.",
            ha="center", fontsize=8, color=COLORS["muted"], style="italic")

    for ext, dpi in EXPORTS:
        fig.savefig(OUT_DIR / f"dsa_network_style.{ext}",
                    bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def main():
    panel_network_style()
    print("Generated:")
    for ext, _ in EXPORTS:
        p = OUT_DIR / f"dsa_network_style.{ext}"
        print(f"  {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
