"""
utils/feishu.py
飞书自定义机器人通知工具。

配置方式：在项目根目录的 .env 文件中写入：
    FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxx
    FEISHU_SIGN_SECRET=xxxxxxxx   # 可选，开启安全设置后需要

未配置 FEISHU_WEBHOOK_URL 时，所有调用静默跳过。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.request
from pathlib import Path

from loguru import logger

# ── 自动加载项目根目录下的 .env ──────────────────────────────────────────────
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    with _ENV_PATH.open(encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


def _sign(secret: str, timestamp: str) -> str:
    """飞书自定义机器人签名（安全设置开启时使用）。"""
    msg = f"{timestamp}\n{secret}"
    mac = hmac.new(msg.encode("utf-8"), digestmod=hashlib.sha256)
    return base64.b64encode(mac.digest()).decode("utf-8")


def _post(payload: dict) -> bool:
    """向 Webhook URL 发送 JSON payload，返回是否成功。"""
    webhook = os.environ.get("FEISHU_WEBHOOK_URL", "").strip()
    if not webhook:
        return False

    secret = os.environ.get("FEISHU_SIGN_SECRET", "").strip()
    if secret:
        ts = str(int(time.time()))
        payload["timestamp"] = ts
        payload["sign"] = _sign(secret, ts)

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("code", 0) != 0:
                logger.warning(f"[飞书] 发送失败: {result}")
                return False
        return True
    except Exception as exc:
        logger.warning(f"[飞书] 请求异常: {exc}")
        return False


def send_eval_result(
    epoch: int,
    total_epochs: int,
    metrics: dict,
    val_loss: float,
    is_best: bool,
    class_names: tuple[str, ...],
    run_name: str = "",
) -> None:
    """
    在每次 eval 后调用，向飞书群发送结构化评估结果。

    Parameters
    ----------
    epoch        : 当前 epoch
    total_epochs : 总 epoch 数
    metrics      : SegEvaluator.compute() 返回的字典，含 mIoU / aAcc / mDice / IoU
    val_loss     : 验证集平均 loss
    is_best      : 是否刷新了最优 mIoU
    class_names  : 类别名称列表
    run_name     : 训练配置名称（用于消息标题区分不同实验）
    """
    if not os.environ.get("FEISHU_WEBHOOK_URL", "").strip():
        return  # 未配置，静默跳过

    miou  = metrics["mIoU"]  * 100
    aacc  = metrics["aAcc"]  * 100
    mdice = metrics["mDice"] * 100
    iou   = metrics["IoU"]   # numpy array, shape (num_classes,)

    # ── 构造消息文本 ──────────────────────────────────────────────────────────
    title_tag  = f"✅ 新最优  " if is_best else ""
    run_tag    = f"[{run_name}]  " if run_name else ""
    header     = f"【SubT 训练 Eval】{run_tag}Epoch {epoch}/{total_epochs}"

    summary = (
        f"mIoU : {miou:.2f}%\n"
        f"aAcc : {aacc:.2f}%\n"
        f"mDice: {mdice:.2f}%\n"
        f"Loss : {val_loss:.4f}"
    )

    per_class_lines = "\n".join(
        f"  {name:<14s}: {float(iou[i]) * 100:5.1f}%"
        for i, name in enumerate(class_names)
    )

    best_line = "\n✅ 新最优模型！best.pth 已保存" if is_best else ""

    full_text = (
        f"{title_tag}{header}\n"
        f"{'─' * 36}\n"
        f"{summary}\n"
        f"{'─' * 36}\n"
        f"各类别 IoU:\n{per_class_lines}"
        f"{best_line}"
    )

    _post({"msg_type": "text", "content": {"text": full_text}})


def send_training_done(
    best_miou: float,
    output_dir: str,
    run_name: str = "",
) -> None:
    """训练完成时发送汇总通知。"""
    if not os.environ.get("FEISHU_WEBHOOK_URL", "").strip():
        return

    run_tag = f"[{run_name}]  " if run_name else ""
    text = (
        f"【SubT 训练完成】{run_tag}\n"
        f"最优 mIoU : {best_miou * 100:.2f}%\n"
        f"输出目录  : {output_dir}"
    )
    _post({"msg_type": "text", "content": {"text": text}})
