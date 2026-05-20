"""
scripts/viz_dsa_dynamics.py

从训练好的 TMDS checkpoint 跑真实样本，输出 DSA decoder 的数据驱动可视化：

  1. dirs_arrows       学到的方向向量箭头（叠加到原图上）
  2. strip_sampling    选定 query 位置的 strip 采样轨迹
  3. attn_heatmap      注意力权重热力图 + top-k 采样点高亮
  4. feature_pca       DSA 前/后特征 PCA 对比

实现要点：
  - 主代码 models/heads/dsa_decoder.py 不动；
  - 在脚本内对 DeformableStripAttention.forward 做镜像复算（使用 checkpoint 同一份权重），
    一边算一边保留中间张量；
  - 通过 forward pre-hook 抓取 DSA 模块的输入张量（fused_half）。

用法：
    python scripts/viz_dsa_dynamics.py
    python scripts/viz_dsa_dynamics.py --checkpoint outputs_final/final_tmds/best.pth
    python scripts/viz_dsa_dynamics.py --image_path dataset/.../some_crack_image.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from loguru import logger

from utils.constants import IMAGENET_MEAN, IMAGENET_STD
from utils.segmentor_loader import build_segmentor_from_checkpoint


# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────────────

def load_image_tensor(image_path: str, input_size: int = 512):
    """返回 (img_tensor [1,3,H,W] 已归一化, img_np_uint8 [H,W,3])。"""
    img = Image.open(image_path).convert("RGB").resize((input_size, input_size),
                                                       Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    img_np = (arr * 255).astype(np.uint8)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    norm = (arr - mean) / std
    t = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t, img_np


def find_default_crack_image(input_size: int = 512) -> Path | None:
    """从 tongji 验证集中挑一张含足够多 crack 像素的图。"""
    val_img = Path("dataset/tongji_data_awesome/img_dir/valid")
    val_ann = Path("dataset/tongji_data_awesome/ann_dir/valid")
    if not val_img.exists() or not val_ann.exists():
        return None
    for ann in sorted(val_ann.glob("*.png"))[:200]:
        m = np.array(Image.open(ann))
        if (m == 1).sum() > 800:   # 一张图至少 800 个 crack 像素
            cand = val_img / (ann.stem + ".jpg")
            if cand.exists():
                return cand
    return None


# ──────────────────────────────────────────────────────────────────────────────
# DSA forward 镜像复算（捕获中间张量）
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def instrumented_dsa_forward(dsa, x):
    """镜像 DeformableStripAttention.forward，返回所有中间量。"""
    B, C, H, W = x.shape
    device = x.device
    nh, ns, M = dsa.num_heads, dsa.num_strips, dsa.M
    hd = dsa.head_dim

    x_f = x.float()
    dirs_raw = dsa.direction_pred(x_f)                       # [B, nh*ns*2]
    Q = dsa.q_proj(x_f); K = dsa.k_proj(x_f); V = dsa.v_proj(x_f)

    dirs = dirs_raw.view(B, nh, ns, 2)
    dirs = F.normalize(dirs, dim=-1, eps=1e-6)               # [B, nh, ns, 2]

    t = torch.linspace(-dsa.max_offset, dsa.max_offset, M, device=device)
    strip_offsets = dirs.unsqueeze(-2) * t.view(1, 1, 1, -1, 1)   # [B,nh,ns,M,2]

    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([grid_x, grid_y], dim=-1)
    base_flat = base.view(1, H * W, 1, 2)

    head_outs, head_attns, head_sgs = [], [], []
    for h in range(nh):
        K_h = K[:, h * hd:(h + 1) * hd]
        V_h = V[:, h * hd:(h + 1) * hd]
        Q_h = Q[:, h * hd:(h + 1) * hd]

        off_h = strip_offsets[:, h].unsqueeze(2)             # [B, ns, 1, M, 2]
        sg = (base_flat + off_h).clamp(-1, 1)                # [B, ns, HW, M, 2]
        head_sgs.append(sg)
        sg_flat = sg.reshape(B * ns, H * W, M, 2)

        K_h_flat = K_h.unsqueeze(1).expand(-1, ns, -1, -1, -1) \
                      .reshape(B * ns, hd, H, W)
        V_h_flat = V_h.unsqueeze(1).expand(-1, ns, -1, -1, -1) \
                      .reshape(B * ns, hd, H, W)

        sK = F.grid_sample(K_h_flat, sg_flat, mode='bilinear',
                           align_corners=True, padding_mode='border')
        sV = F.grid_sample(V_h_flat, sg_flat, mode='bilinear',
                           align_corners=True, padding_mode='border')
        sK = sK.reshape(B, ns, hd, H * W, M).permute(0, 3, 1, 4, 2) \
               .reshape(B, H * W, ns * M, hd)
        sV = sV.reshape(B, ns, hd, H * W, M).permute(0, 3, 1, 4, 2) \
               .reshape(B, H * W, ns * M, hd)

        Q_flat = Q_h.view(B, hd, H * W).permute(0, 2, 1).unsqueeze(2)
        attn = torch.matmul(Q_flat, sK.transpose(-1, -2)) * dsa.scale
        attn = F.softmax(attn, dim=-1)                        # [B, HW, 1, ns*M]
        head_attns.append(attn)

        out_h = torch.matmul(attn, sV).squeeze(2).permute(0, 2, 1).view(B, hd, H, W)
        head_outs.append(out_h)

    out = torch.cat(head_outs, dim=1)
    out = F.group_norm(out, dsa.norm.num_groups,
                       dsa.norm.weight.float(),
                       dsa.norm.bias.float() if dsa.norm.bias is not None else None,
                       dsa.norm.eps)
    out_proj = dsa.out_proj(out.to(x.dtype))
    attended = out_proj + x                                    # 残差

    return {
        "dirs":        dirs.detach().cpu(),                            # [B,nh,ns,2]
        "sg":          torch.stack(head_sgs, dim=1).detach().cpu(),    # [B,nh,ns,HW,M,2]
        "attn":        torch.stack(head_attns, dim=1).detach().cpu(),  # [B,nh,HW,1,ns*M]
        "fused_half":  x.detach().cpu(),
        "attended":    attended.detach().cpu(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 可视化函数
# ──────────────────────────────────────────────────────────────────────────────

def viz_dirs_arrows(dirs, img_np, save_path):
    """方向箭头叠加：dirs[1, nh, ns, 2] → 16 个箭头按头分色。"""
    Himg, Wimg = img_np.shape[:2]
    nh, ns = dirs.shape[1], dirs.shape[2]
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img_np)
    ax.axis('off')
    ax.set_title(f"Learned strip directions  ({nh} heads × {ns} strips = {nh*ns} vectors, globally shared)",
                 fontsize=10)

    arrow_len = min(Himg, Wimg) * 0.18
    cx, cy = Wimg / 2, Himg / 2
    grid_step = min(Himg, Wimg) * 0.06   # head/strip 间的轻微位移
    for h in range(nh):
        for s in range(ns):
            dx, dy = dirs[0, h, s].numpy()
            sx = cx + (h - (nh - 1) / 2) * grid_step
            sy = cy + (s - (ns - 1) / 2) * grid_step
            ax.arrow(sx, sy, dx * arrow_len, dy * arrow_len,
                     head_width=10, head_length=14,
                     color=colors[h % len(colors)],
                     linewidth=1.8, alpha=0.9, length_includes_head=True)
    # 图例
    for h in range(nh):
        ax.plot([], [], color=colors[h % len(colors)],
                linewidth=2.2, label=f"head {h}")
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85)

    for ext, dpi in [("pdf", None), ("png", 200)]:
        plt.savefig(save_path.with_suffix(f".{ext}"), bbox_inches='tight', dpi=dpi)
    plt.close()


def _img_to_pixel(qx_feat, qy_feat, H_feat, W_feat, Himg, Wimg):
    """特征坐标 → 像素坐标。"""
    return qx_feat * Wimg / W_feat, qy_feat * Himg / H_feat


def viz_strip_sampling(dirs, max_offset, M, queries_feat, H_feat, W_feat,
                       img_np, save_path):
    """画选定 query 处每条 strip 的 M 个采样点。queries_feat: list of (qy, qx) in feature coords."""
    Himg, Wimg = img_np.shape[:2]
    nh, ns = dirs.shape[1], dirs.shape[2]
    colors = plt.cm.tab10.colors
    t_vals = np.linspace(-max_offset, max_offset, M)

    nq = len(queries_feat)
    fig, axes = plt.subplots(1, nq, figsize=(6 * nq, 6))
    if nq == 1:
        axes = [axes]

    for qi, (qyf, qxf) in enumerate(queries_feat):
        ax = axes[qi]
        ax.imshow(img_np)
        ax.axis('off')
        qx_img, qy_img = _img_to_pixel(qxf, qyf, H_feat, W_feat, Himg, Wimg)
        ax.plot(qx_img, qy_img, 'wo', markersize=11, markerfacecolor='red',
                markeredgecolor='white', markeredgewidth=1.6, zorder=10)
        ax.set_title(f"query @ feat ({qxf},{qyf}) → img ({int(qx_img)},{int(qy_img)})",
                     fontsize=9)
        # 每个 head/strip 的采样点
        for h in range(nh):
            for s in range(ns):
                dx, dy = dirs[0, h, s].numpy()
                # feature normalized coords 中 t * dir 偏移：在 H_feat=W_feat 网格上
                # 对应像素位移 = t * dir * (Wimg/2)（因 normalized [-1,1] 范围 = feature 全宽）
                sxs = qx_img + t_vals * dx * (Wimg / 2)
                sys = qy_img + t_vals * dy * (Himg / 2)
                ax.plot(sxs, sys, '-', color=colors[h % len(colors)],
                        alpha=0.35, linewidth=0.9)
                ax.plot(sxs, sys, '.', color=colors[h % len(colors)],
                        markersize=4, alpha=0.9)
    plt.suptitle(f"Strip sampling points ({nh}×{ns} strips × {M} pts per strip)",
                 fontsize=11)
    for ext, dpi in [("pdf", None), ("png", 200)]:
        plt.savefig(save_path.with_suffix(f".{ext}"), bbox_inches='tight', dpi=dpi)
    plt.close()


def viz_attention_heatmap(attn, dirs, max_offset, M, query_feat, H_feat, W_feat,
                          img_np, save_path):
    """选定 query 的 attention[ns × M] 热力图 + 采样点按权重高亮。"""
    Himg, Wimg = img_np.shape[:2]
    nh, ns = dirs.shape[1], dirs.shape[2]
    qyf, qxf = query_feat
    q_idx = qyf * W_feat + qxf
    colors = plt.cm.tab10.colors
    t_vals = np.linspace(-max_offset, max_offset, M)
    qx_img, qy_img = _img_to_pixel(qxf, qyf, H_feat, W_feat, Himg, Wimg)

    fig, axes = plt.subplots(nh, 2, figsize=(11, 3.2 * nh))
    if nh == 1:
        axes = axes.reshape(1, 2)

    for h in range(nh):
        w = attn[0, h, q_idx, 0].numpy().reshape(ns, M)     # [ns, M]

        ax_h = axes[h, 0]
        im = ax_h.imshow(w, aspect='auto', cmap='viridis')
        ax_h.set_xticks(range(M))
        ax_h.set_xticklabels([f"{v:+.2f}" for v in t_vals], fontsize=7)
        ax_h.set_yticks(range(ns))
        ax_h.set_yticklabels([f"strip {s}" for s in range(ns)], fontsize=8)
        ax_h.set_xlabel("sample offset t", fontsize=8)
        ax_h.set_title(f"head {h} — attention weights", fontsize=9)
        plt.colorbar(im, ax=ax_h, fraction=0.04, pad=0.02)

        ax_i = axes[h, 1]
        ax_i.imshow(img_np); ax_i.axis('off')
        ax_i.plot(qx_img, qy_img, 'wo', markersize=11, markerfacecolor='red',
                  markeredgecolor='white', markeredgewidth=1.6, zorder=10)
        for s in range(ns):
            dx, dy = dirs[0, h, s].numpy()
            sxs = qx_img + t_vals * dx * (Wimg / 2)
            sys = qy_img + t_vals * dy * (Himg / 2)
            sizes = 6 + 130 * (w[s] / max(w[s].max(), 1e-6))
            ax_i.scatter(sxs, sys, s=sizes, c=[colors[h % len(colors)]] * M,
                         alpha=0.55, edgecolors='white', linewidths=0.4)
            ax_i.plot(sxs, sys, '-', color=colors[h % len(colors)],
                      alpha=0.25, linewidth=0.6)
        ax_i.set_title(f"head {h} — sample points (size ∝ attention)", fontsize=9)

    plt.suptitle(f"Attention at query feat ({qxf},{qyf})", fontsize=11)
    plt.tight_layout()
    for ext, dpi in [("pdf", None), ("png", 200)]:
        plt.savefig(save_path.with_suffix(f".{ext}"), bbox_inches='tight', dpi=dpi)
    plt.close()


def _pca_rgb(feat):
    """[1, C, H, W] tensor → [H, W, 3] RGB via PCA。"""
    _, C, H, W = feat.shape
    flat = feat[0].permute(1, 2, 0).reshape(-1, C).numpy()
    rgb = PCA(n_components=3).fit_transform(flat)
    rgb = (rgb - rgb.min(axis=0)) / (rgb.max(axis=0) - rgb.min(axis=0) + 1e-8)
    return rgb.reshape(H, W, 3)


def viz_feature_pca(fused_half, attended, img_np, save_path):
    """DSA 前/后特征 PCA 着色对比。"""
    rgb_in = _pca_rgb(fused_half)
    rgb_out = _pca_rgb(attended)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(img_np); axes[0].axis('off'); axes[0].set_title("input image", fontsize=10)
    axes[1].imshow(rgb_in); axes[1].axis('off')
    axes[1].set_title("fused_half (DSA input)  PCA-RGB", fontsize=10)
    axes[2].imshow(rgb_out); axes[2].axis('off')
    axes[2].set_title("attended (DSA output)  PCA-RGB", fontsize=10)
    plt.tight_layout()
    for ext, dpi in [("pdf", None), ("png", 200)]:
        plt.savefig(save_path.with_suffix(f".{ext}"), bbox_inches='tight', dpi=dpi)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate DSA decoder dynamic visualizations from a trained checkpoint."
    )
    parser.add_argument("--checkpoint", default="outputs_final/final_tmds/best.pth")
    parser.add_argument("--image_path", default=None,
                        help="样本图路径；未提供时自动从 tongji valid 中挑一张含裂缝图")
    parser.add_argument("--output_dir", default="paper_figures/dynamics")
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query_feat", type=int, nargs=2, default=None,
                        metavar=("QX", "QY"),
                        help="主 query 在 H/8 特征图上的 (x, y) 坐标；默认自动找裂缝中心")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 找样本图 ─────────────────────────────────────────────────────
    if args.image_path is None:
        cand = find_default_crack_image(args.input_size)
        if cand is None:
            raise FileNotFoundError(
                "未在 dataset/tongji_data_awesome/{img,ann}_dir/valid 中找到含裂缝样本。"
                "请用 --image_path 显式指定。"
            )
        args.image_path = str(cand)
    logger.info(f"样本图: {args.image_path}")

    img_tensor, img_np = load_image_tensor(args.image_path, args.input_size)
    img_tensor = img_tensor.to(device)

    # ── 2. 加载模型 ─────────────────────────────────────────────────────
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, cfg = build_segmentor_from_checkpoint(ckpt, device)
    model.eval()
    logger.info(f"模型加载完成: {type(model).__name__}")

    # 定位 DSA 模块
    if not hasattr(model, "linear_decoder") or not hasattr(model.linear_decoder, "dsa"):
        raise AttributeError(
            "模型不是 TMDSSegmentor 或缺少 linear_decoder.dsa；本脚本只支持 TMDS checkpoint。"
        )
    dsa = model.linear_decoder.dsa

    # ── 3. 用 forward pre-hook 捕获 fused_half（DSA 输入），并运行完整模型一次 ──
    captured = {}

    def pre_hook(_module, inputs):
        captured["fused_half"] = inputs[0].detach()

    handle = dsa.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            _ = model(img_tensor)
    finally:
        handle.remove()

    fused_half = captured["fused_half"]
    logger.info(f"fused_half shape: {tuple(fused_half.shape)}")
    H_feat, W_feat = fused_half.shape[-2:]

    # ── 4. 镜像复算 DSA（捕获中间张量）──────────────────────────────────
    interm = instrumented_dsa_forward(dsa, fused_half)
    # 一致性 sanity check（与生产 forward 比对）
    with torch.no_grad():
        prod_out = dsa(fused_half).detach().cpu()
    max_diff = (interm["attended"] - prod_out).abs().max().item()
    logger.info(f"镜像 vs 生产 forward 最大差异: {max_diff:.2e} "
                f"({'OK' if max_diff < 1e-3 else '⚠ 偏大'})")

    # ── 5. 选定 query 位置 ──────────────────────────────────────────────
    if args.query_feat is not None:
        primary_query = (args.query_feat[1], args.query_feat[0])  # (qy, qx)
    else:
        # 自动找：根据 attended 在 H/8 上 crack-likely 区域，取最强响应位置
        # 简化：取 fused_half 第一通道在 GT crack 区域（如有 ann）的最强；否则取整图最强
        ann_p = Path(str(args.image_path).replace("img_dir", "ann_dir")
                                          .replace(".jpg", ".png"))
        if ann_p.exists():
            ann = np.array(Image.open(ann_p).resize((W_feat, H_feat), Image.NEAREST))
            crack_mask = (ann == 1)
            if crack_mask.sum() >= 10:
                # 取 crack mask 内 attended 范数最大的位置
                feat_norm = interm["attended"][0].norm(dim=0).numpy()
                feat_norm[~crack_mask] = -np.inf
                qy, qx = np.unravel_index(np.argmax(feat_norm), feat_norm.shape)
                primary_query = (int(qy), int(qx))
            else:
                primary_query = (H_feat // 2, W_feat // 2)
        else:
            primary_query = (H_feat // 2, W_feat // 2)
    logger.info(f"primary query (feat 坐标 qy,qx) = {primary_query}")

    # 其他 query：在 primary 附近偏移 + 中心
    secondary_queries = [
        primary_query,
        (max(0, primary_query[0] - H_feat // 6), primary_query[1]),
        (primary_query[0], min(W_feat - 1, primary_query[1] + W_feat // 6)),
    ]

    # ── 6. 生成 4 张图 ─────────────────────────────────────────────────
    logger.info("生成可视化...")
    viz_dirs_arrows(interm["dirs"], img_np, out_dir / "dirs_arrows")
    viz_strip_sampling(interm["dirs"], dsa.max_offset, dsa.M,
                       secondary_queries, H_feat, W_feat,
                       img_np, out_dir / "strip_sampling")
    viz_attention_heatmap(interm["attn"], interm["dirs"],
                          dsa.max_offset, dsa.M,
                          primary_query, H_feat, W_feat,
                          img_np, out_dir / "attn_heatmap")
    viz_feature_pca(interm["fused_half"], interm["attended"], img_np,
                    out_dir / "feature_pca")

    logger.success(f"完成。输出目录: {out_dir.resolve()}")
    for name in ("dirs_arrows", "strip_sampling", "attn_heatmap", "feature_pca"):
        for ext in ("pdf", "png"):
            p = out_dir / f"{name}.{ext}"
            if p.exists():
                logger.info(f"  {p}  ({p.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
