"""将 valid 合并进 train，test 重命名为 valid。

执行后结果：
  - train = 原 train + 原 valid
  - valid = 原 test
  - test  目录不再存在
"""

from __future__ import annotations

import shutil
from pathlib import Path

DATASET_DIR = Path("dataset/tongji_data")

# 需要处理的子目录（skel_dir 只有 train，跳过）
SPLIT_DIRS = ["img_dir", "ann_dir", "visualization"]


def merge_valid_into_train(split_dir: Path) -> None:
    src = split_dir / "valid"
    dst = split_dir / "train"
    if not src.exists():
        print(f"[跳过] 不存在: {src}")
        return

    files = list(src.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())

    for src_file in files:
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

    shutil.rmtree(src)
    print(f"[合并] {src} -> {dst}  ({file_count} 个文件)")


def rename_test_to_valid(split_dir: Path) -> None:
    src = split_dir / "test"
    dst = split_dir / "valid"
    if not src.exists():
        print(f"[跳过] 不存在: {src}")
        return
    if dst.exists():
        print(f"[警告] {dst} 已存在，无法重命名，请先检查")
        return

    src.rename(dst)
    print(f"[重命名] {src} -> {dst}")


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"数据集目录不存在: {DATASET_DIR}")

    for name in SPLIT_DIRS:
        split_dir = DATASET_DIR / name
        if not split_dir.exists():
            print(f"[跳过] 目录不存在: {split_dir}")
            continue
        merge_valid_into_train(split_dir)
        rename_test_to_valid(split_dir)

    print("\n完成。当前分割：train + valid（原 test）")


if __name__ == "__main__":
    main()
