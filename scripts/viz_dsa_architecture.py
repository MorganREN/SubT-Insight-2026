"""
scripts/viz_dsa_architecture.py

DSA Decoder 架构图（论文 Method 章节用）。

产出两张块图：
  paper_figures/dsa_arch_overall.{pdf,png,svg}   —— 整体管线（FPN + AvgPool + DSA + Upsample + Output）
  paper_figures/dsa_arch_dsa_core.{pdf,png,svg}  —— DSA 模块内部机制

实现：matplotlib + FancyBboxPatch（圆角块）+ FancyArrowPatch（箭头），
不依赖 checkpoint、可一键复现。SVG/PDF 可导入 Visio、PPT、Illustrator 或 Inkscape 继续编辑。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 配色（保持中性、印刷友好）
COLORS = {
    "input":  "#E8F4FD",   # 浅蓝
    "fpn":    "#FFF2CC",   # 浅黄
    "op":     "#FFE4B5",   # 浅橙：op
    "dsa":    "#D5E8D4",   # 浅绿（DSA 模块高亮）
    "norm":   "#F0F0F0",   # 灰
    "output": "#FCE5CD",   # 输出
    "edge":   "#333333",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,   # 嵌入 TrueType，便于论文排版
    "ps.fonttype":  42,
    "svg.fonttype": "none",  # 保留文本对象，便于 Visio / Illustrator 二次编辑
})

EXPORTS = (("pdf", None), ("png", 260), ("svg", None))


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def box(ax, xy, w, h, text, color="#FFFFFF", fontsize=9, lw=1.0):
    """绘制圆角块，中心坐标 xy。"""
    bx = FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, facecolor=color, edgecolor=COLORS["edge"],
    )
    ax.add_patch(bx)
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, p0, p1, label=None, fontsize=7, style="-|>", rad=0.0, lw=1.0,
          color=None, label_offset=(0.06, 0.06)):
    """箭头 p0 → p1，可选 label。"""
    cs = f"arc3,rad={rad}" if rad != 0.0 else None
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color or COLORS["edge"],
        connectionstyle=cs,
    )
    ax.add_patch(a)
    if label:
        mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        ax.text(mid[0] + label_offset[0], mid[1] + label_offset[1],
                label, fontsize=fontsize, color="#555")


def save_figure(fig, name: str):
    """保存同一张图的论文版 PDF、预览 PNG 与可编辑 SVG。"""
    for ext, dpi in EXPORTS:
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=dpi)


# ──────────────────────────────────────────────────────────────────────────────
# Panel A — DSA Decoder 整体管线
# ──────────────────────────────────────────────────────────────────────────────

def panel_a():
    fig, ax = plt.subplots(figsize=(14, 6.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.4); ax.axis("off")

    # 四个 backbone stage（深→浅，自上而下）
    stages = [
        ("C4", "768  @H/32", 5.5),
        ("C3", "384  @H/16", 4.2),
        ("C2", "192  @H/8",  2.9),
        ("C1", "96  @H/4",   1.6),
    ]

    # 左列：输入特征
    for name, ch, y in stages:
        box(ax, (1.1, y), 1.6, 0.8, f"{name}\n{ch}", color=COLORS["input"])

    # 第二列：lateral 1×1 conv
    for _, _, y in stages:
        box(ax, (3.5, y), 1.6, 0.8, "lateral\n1×1 conv", color=COLORS["fpn"])
        arrow(ax, (1.9, y), (2.7, y))

    # 自顶向下 add（lateral i 的输出上采样加到 lateral i-1 的输出）
    for i in range(len(stages) - 1):
        y_top = stages[i][2]
        y_bot = stages[i + 1][2]
        # 短斜箭头：在 lateral 列右侧画
        arrow(ax, (4.0, y_top - 0.4), (4.0, y_bot + 0.4),
              label="up×2 + add", label_offset=(0.1, -0.05), fontsize=7)

    # 第三列：FPN 3×3 conv
    for _, _, y in stages:
        box(ax, (6.1, y), 1.6, 0.8, "FPN\n3×3 conv", color=COLORS["fpn"])
        arrow(ax, (4.3, y), (5.3, y))

    # 第四列：upsample 到 H/4 + 求和符号
    sum_x, sum_y = 9.0, 1.6
    box(ax, (sum_x, sum_y), 0.7, 0.7, "⊕", color="#FFFFFF", fontsize=16)
    ax.text(sum_x, sum_y - 0.55, "fused\n@H/4",
            ha="center", va="top", fontsize=7, color="#555")

    for _, _, y in stages:
        rad = 0.0 if y == 1.6 else -0.15
        label = "up→H/4" if y != 1.6 else None
        arrow(ax, (6.9, y), (sum_x - 0.4, sum_y), rad=rad, label=label,
              label_offset=(-0.6, 0.05), fontsize=7)

    # AvgPool ↓2
    box(ax, (10.6, 1.6), 1.5, 0.7, "AvgPool ↓2\n(memory-saving)",
        color=COLORS["norm"], fontsize=8)
    arrow(ax, (sum_x + 0.4, 1.6), (9.85, 1.6))
    ax.text(10.6, 1.0, "[B, C, H/8, W/8]", ha="center", fontsize=7.5, color="#555")

    # DSA Module（高亮）
    box(ax, (12.7, 1.6), 1.8, 1.0,
        "DSA Module\nnh=4, ns=4, M=8",
        color=COLORS["dsa"], fontsize=9, lw=1.5)
    arrow(ax, (11.35, 1.6), (11.95, 1.6))
    ax.text(12.7, 0.9, "strip attention @ H/8", ha="center", fontsize=7.5, color="#555")

    # Upsample ↑2
    box(ax, (12.7, 3.2), 1.5, 0.7, "Upsample ↑2", color=COLORS["norm"])
    arrow(ax, (12.7, 2.1), (12.7, 2.85))

    # out_norm
    box(ax, (12.7, 4.4), 1.5, 0.7, "out_norm\n(GroupNorm)", color=COLORS["norm"], fontsize=8)
    arrow(ax, (12.7, 3.55), (12.7, 4.05))

    # 输出
    box(ax, (12.7, 5.6), 1.7, 0.7, "F_L\n[B, C, H/4, W/4]",
        color=COLORS["output"], fontsize=9, lw=1.2)
    arrow(ax, (12.7, 4.75), (12.7, 5.25))

    ax.set_title("DSA Decoder — Overall Pipeline (FPN + Strip Attention)",
                 fontsize=12, pad=10)
    ax.text(0.25, 0.35,
            "FPN lateral/fpn convs run in float32; DSA is applied after AvgPool to reduce token cost.",
            fontsize=7.5, color="#666", style="italic")

    save_figure(fig, "dsa_arch_overall")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Panel B — DSA 模块内部机制
# ──────────────────────────────────────────────────────────────────────────────

def panel_b():
    fig, ax = plt.subplots(figsize=(12.5, 10))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 10); ax.axis("off")

    # Input X
    box(ax, (6.25, 9.4), 2.4, 0.6, "Input X    [B, C, H/8, W/8]",
        color=COLORS["input"], fontsize=10)

    # 四条平行分支：direction_pred / Q / K / V
    branches = [
        (1.5,  "direction_pred\nAvgPool → Linear",        COLORS["fpn"]),
        (4.5,  "Q proj  1×1\n[B,C,H,W]",                  COLORS["fpn"]),
        (8.0,  "K proj  1×1\n[B,C,H,W]",                  COLORS["fpn"]),
        (11.0, "V proj  1×1\n[B,C,H,W]",                  COLORS["fpn"]),
    ]
    for x, label, c in branches:
        arrow(ax, (6.25, 9.1), (x, 8.55), rad=(x - 6.25) * 0.03)
        box(ax, (x, 8.2), 1.8, 0.7, label, color=c, fontsize=8)

    # direction_pred 之后：reshape + L2 normalize
    box(ax, (1.5, 6.9), 1.8, 0.7,
        "reshape → L2-norm\nd_hat: [B, nh, ns, 2]", color=COLORS["norm"], fontsize=7)
    arrow(ax, (1.5, 7.85), (1.5, 7.25))

    # base grid + t·offset → sampling grid
    box(ax, (3.5, 6.9), 1.6, 0.7,
        "base grid +\nt·offset", color=COLORS["norm"], fontsize=7)

    box(ax, (2.5, 5.4), 2.6, 0.8,
        "sampling grid sg\n[B, nh, ns, HW, M, 2]",
        color=COLORS["op"], fontsize=8)
    arrow(ax, (1.7, 6.55), (2.3, 5.85), rad=-0.1)
    arrow(ax, (3.3, 6.55), (2.7, 5.85), rad=0.1)

    # grid_sample (K) and grid_sample (V)
    box(ax, (7.0, 5.4), 1.8, 0.7, "grid_sample\n→ sK", color=COLORS["op"], fontsize=8)
    box(ax, (10.0, 5.4), 1.8, 0.7, "grid_sample\n→ sV", color=COLORS["op"], fontsize=8)
    ax.text(8.5, 4.75, "sK/sV: [B, HW, ns*M, hd]", ha="center", fontsize=7.5, color="#555")
    # sg → grid_sample(K) 与 grid_sample(V)
    arrow(ax, (3.8, 5.4), (6.1, 5.4), label="sg", label_offset=(-0.05, 0.15), fontsize=7)
    arrow(ax, (3.8, 5.2), (9.1, 5.0), rad=-0.25, label="sg", fontsize=7,
          label_offset=(0.0, -0.25))
    # K → grid_sample(K)
    arrow(ax, (8.0, 7.85), (7.0, 5.8))
    # V → grid_sample(V)
    arrow(ax, (11.0, 7.85), (10.0, 5.8))

    # 注意力打分：Q · sK^T / √d_h → softmax → attn
    box(ax, (4.8, 3.7), 2.6, 0.8,
        "Q · sK^T / √d_h\n→ softmax → attn",
        color=COLORS["op"], fontsize=9)
    ax.text(4.8, 3.05, "attn: [B, HW, 1, ns*M]", ha="center", fontsize=7.5, color="#555")
    arrow(ax, (4.5, 7.85), (4.5, 4.15), rad=0.1, label="Q", label_offset=(-0.25, 0.1))
    arrow(ax, (6.6, 5.0), (5.4, 4.15), label="sK", label_offset=(0.05, 0.1))

    # attn · sV → out per head
    box(ax, (8.0, 2.4), 2.6, 0.8,
        "attn · sV\n→ out per head\n[B, HW, hd]",
        color=COLORS["op"], fontsize=8)
    arrow(ax, (5.2, 3.25), (7.5, 2.85), label="attn", label_offset=(0.0, 0.1))
    arrow(ax, (10.0, 5.0), (8.4, 2.85), label="sV", label_offset=(0.0, -0.2))

    # 拼接各头
    box(ax, (5.5, 1.2), 2.4, 0.7,
        "concat heads\n→ [B, C, H/8, W/8]",
        color=COLORS["fpn"], fontsize=8)
    arrow(ax, (7.5, 1.95), (5.9, 1.55))

    # GroupNorm
    box(ax, (4.0, 0.4), 1.6, 0.5, "GroupNorm",
        color=COLORS["norm"], fontsize=8)
    arrow(ax, (5.0, 0.85), (4.4, 0.65), rad=-0.1)

    # out_proj 1×1
    box(ax, (6.5, 0.4), 1.6, 0.5, "out_proj 1×1",
        color=COLORS["fpn"], fontsize=8)
    arrow(ax, (4.8, 0.4), (5.7, 0.4))

    # ⊕ 残差
    box(ax, (8.7, 0.4), 0.6, 0.5, "⊕", color="#FFFFFF", fontsize=14)
    arrow(ax, (7.3, 0.4), (8.4, 0.4))

    # 残差 skip：从 X 顶端走到 ⊕（用虚线突出 skip 性质）
    skip_color = "#888888"
    # X 右侧 → 向右下沿外缘走 → 进入 ⊕ 右侧
    skip_arr = FancyArrowPatch(
        (7.45, 9.4), (8.7, 0.7),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.0,
        color=skip_color, linestyle="--",
        connectionstyle="arc3,rad=-0.55",
    )
    ax.add_patch(skip_arr)
    ax.text(11.7, 5.5, "residual (X)", fontsize=8, color=skip_color, rotation=-78)

    # 输出 Y
    box(ax, (10.5, 0.4), 1.6, 0.5, "Output Y",
        color=COLORS["output"], fontsize=9, lw=1.2)
    arrow(ax, (9.0, 0.4), (9.7, 0.4))

    ax.set_title("DSA Module — Internal Mechanism",
                 fontsize=12, pad=10)

    # 说明：方向为全局共享（不随空间位置变化）
    ax.text(0.1, -0.4,
            "Note: direction_pred outputs per-image global directions, not per-pixel directions. Default: nh=4, ns=4, M=8.",
            fontsize=7.5, color="#666", style="italic")

    save_figure(fig, "dsa_arch_dsa_core")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────

def main():
    panel_a()
    panel_b()
    print("已输出：")
    for name in ("dsa_arch_overall", "dsa_arch_dsa_core"):
        for ext, _ in EXPORTS:
            p = OUT_DIR / f"{name}.{ext}"
            print(f"  {p} ({p.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
