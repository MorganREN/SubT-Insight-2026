"""将 dataset/tongji 的 LabelMe 标注转换为 dataset/tongji_data 的结构。"""

from __future__ import annotations

import base64
import io
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


SOURCE_DIR = Path("dataset/tongji")
TARGET_DIR = Path("dataset/tongji_data")
TARGET_SIZE = (640, 640)
SPLIT_RATIO = (0.70, 0.15, 0.15)
RANDOM_SEED = 42
OVERLAY_INTENSITY = 115
CLEAR_OUTPUT = True
DRAW_OVERLAY_LEGEND = True


CLASS_DEFECTS = {
	1: "crack",
	2: "leakage_b",
	3: "leakage_w",
	4: "leakage_g",
	5: "lining_falling_off",
	6: "segment_damage",
}


LABEL_TO_CLASS = {
	# tongji 主标签
	"crack": 1,
	"leakageb": 2,
	"leakagew": 3,
	"leakageg": 4,
	"liningfallingoff": 5,
	"segmentdamage": 6,

	# 旧标签兼容映射（统一到 tongji 分类）
	"cracka": 1,
	"crackb": 1,
	"cracksmkjm": 1,

	"leakage": 3,
	"lf": 3,

	"spalling": 5,
	"ss": 5,

	"repair": 6,
	"repairs": 6,
	"other": 6,
}


CLASS_COLORS_RGB = {
	1: (255, 0, 0),
	2: (255, 128, 0),
	3: (0, 0, 255),
	4: (0, 255, 255),
	5: (255, 255, 0),
	6: (255, 0, 255),
}


@dataclass
class Sample:
	stem: str
	json_path: Path
	image_path: Path | None
	image_data_b64: str | None


def normalize_label(label: str) -> str:
	norm = label.strip().lower()
	for token in (" ", "_", "-", "/"):
		norm = norm.replace(token, "")
	return norm


def map_label_to_class(label_raw: str) -> int | None:
	label = normalize_label(label_raw)
	if label in LABEL_TO_CLASS:
		return LABEL_TO_CLASS[label]

	# 宽松兜底，减少漏标
	if "crack" in label:
		return 1
	if label.startswith("leakage"):
		if label.endswith("b"):
			return 2
		if label.endswith("g"):
			return 4
		return 3
	if "lining" in label and "off" in label:
		return 5
	if "segment" in label and "damage" in label:
		return 6
	return None


def decode_image_data(image_data_b64: str) -> Image.Image:
	image_bytes = base64.b64decode(image_data_b64)
	img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
	return ImageOps.exif_transpose(img)


def find_image_path(stem: str, data: dict) -> Path | None:
	for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
		path = SOURCE_DIR / f"{stem}{ext}"
		if path.exists():
			return path

	image_path = data.get("imagePath", "")
	if image_path:
		cand = SOURCE_DIR / Path(image_path).name
		if cand.exists():
			return cand
	return None


def collect_samples() -> list[Sample]:
	samples: list[Sample] = []
	for json_path in sorted(SOURCE_DIR.glob("*.json")):
		data = json.loads(json_path.read_text(encoding="utf-8"))
		image_path = find_image_path(json_path.stem, data)
		has_image_data = bool(data.get("imageData"))

		if image_path is None and not has_image_data:
			print(f"[跳过] 无对应图片: {json_path.name}")
			continue

		samples.append(
			Sample(
				stem=json_path.stem,
				json_path=json_path,
				image_path=image_path,
				image_data_b64=None,
			)
		)
	return samples


def split_samples(samples: list[Sample]) -> dict[str, list[Sample]]:
	random.seed(RANDOM_SEED)
	random.shuffle(samples)

	n = len(samples)
	n_train = int(n * SPLIT_RATIO[0])
	n_valid = int(n * SPLIT_RATIO[1])

	return {
		"train": samples[:n_train],
		"valid": samples[n_train:n_train + n_valid],
		"test": samples[n_train + n_valid:],
	}


def load_image(sample: Sample, data: dict) -> Image.Image:
	image_data_b64 = data.get("imageData")
	if image_data_b64:
		return decode_image_data(image_data_b64)

	if sample.image_path is not None:
		img = Image.open(sample.image_path).convert("RGB")
		return ImageOps.exif_transpose(img)
	raise FileNotFoundError(f"样本无可用图像数据: {sample.json_path.name}")


def draw_shape(draw: ImageDraw.ImageDraw, shape: dict, cls_id: int) -> None:
	points = shape.get("points", [])
	shape_type = str(shape.get("shape_type", "polygon")).lower()
	if not points:
		return

	if shape_type == "rectangle" and len(points) >= 2:
		(x1, y1), (x2, y2) = points[:2]
		draw.rectangle([x1, y1, x2, y2], fill=cls_id)
		return

	if shape_type == "circle" and len(points) >= 2:
		(cx, cy), (px, py) = points[:2]
		r = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
		draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cls_id)
		return

	if len(points) >= 3:
		draw.polygon([(p[0], p[1]) for p in points], fill=cls_id)
	elif len(points) == 2:
		draw.line([(points[0][0], points[0][1]), (points[1][0], points[1][1])], fill=cls_id, width=3)


def create_mask(shapes: list[dict], width: int, height: int) -> np.ndarray:
	mask_pil = Image.new("L", (width, height), 0)
	draw = ImageDraw.Draw(mask_pil)
	unknown_labels: set[str] = set()

	for shape in shapes:
		label_raw = str(shape.get("label", ""))
		cls_id = map_label_to_class(label_raw)
		if cls_id is None:
			unknown_labels.add(label_raw)
			continue
		draw_shape(draw, shape, cls_id)

	if unknown_labels:
		print(f"[提示] 存在未映射标签(已忽略): {sorted(unknown_labels)}")

	return np.array(mask_pil, dtype=np.uint8)


def resize_image(image: Image.Image) -> Image.Image:
	return image.resize(TARGET_SIZE, Image.LANCZOS)


def resize_mask(mask: np.ndarray) -> np.ndarray:
	return np.array(Image.fromarray(mask, mode="L").resize(TARGET_SIZE, Image.NEAREST), dtype=np.uint8)


def generate_mask_overlay(image_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
	color_mask = np.zeros_like(image_arr, dtype=np.uint8)
	for cls_id, color in CLASS_COLORS_RGB.items():
		color_mask[mask_arr == cls_id] = color

	overlay = image_arr.astype(np.int16).copy()
	hit = mask_arr > 0
	overlay[hit] = np.clip(
		image_arr[hit].astype(np.int16) - OVERLAY_INTENSITY + color_mask[hit].astype(np.int16),
		0,
		255,
	)
	return overlay.astype(np.uint8)


def add_overlay_legend(overlay_arr: np.ndarray) -> np.ndarray:
	"""在 overlay 底部追加颜色图例：颜色块 + 病害类别名。"""
	items = [(cls_id, CLASS_DEFECTS[cls_id], CLASS_COLORS_RGB[cls_id]) for cls_id in sorted(CLASS_DEFECTS.keys())]
	if not items:
		return overlay_arr

	overlay_img = Image.fromarray(overlay_arr)
	w, h = overlay_img.size

	cols = 3
	rows = math.ceil(len(items) / cols)
	pad = 10
	row_h = 26
	swatch = 16
	legend_h = pad * 2 + rows * row_h

	canvas = Image.new("RGB", (w, h + legend_h), (245, 245, 245))
	canvas.paste(overlay_img, (0, 0))

	draw = ImageDraw.Draw(canvas)
	draw.line([(0, h), (w, h)], fill=(200, 200, 200), width=1)

	col_w = max(1, w // cols)
	for idx, (cls_id, class_name, color) in enumerate(items):
		row = idx // cols
		col = idx % cols

		x0 = col * col_w + pad
		y0 = h + pad + row * row_h
		draw.rectangle([x0, y0, x0 + swatch, y0 + swatch], fill=color, outline=(0, 0, 0), width=1)
		draw.text((x0 + swatch + 6, y0 - 1), f"{cls_id}: {class_name}", fill=(0, 0, 0))

	return np.array(canvas, dtype=np.uint8)


def generate_segmented(image_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
	segmented = np.zeros_like(image_arr, dtype=np.uint8)
	hit = mask_arr > 0
	segmented[hit] = image_arr[hit]
	return segmented


def ensure_dirs() -> dict[str, dict[str, Path]]:
	if CLEAR_OUTPUT:
		for name in ("img_dir", "ann_dir", "visualization"):
			path = TARGET_DIR / name
			if path.exists():
				shutil.rmtree(path)

	dirs = {
		"img": {},
		"ann": {},
		"overlay": {},
		"segmented": {},
	}
	for split in ("train", "valid", "test"):
		dirs["img"][split] = TARGET_DIR / "img_dir" / split
		dirs["ann"][split] = TARGET_DIR / "ann_dir" / split
		dirs["overlay"][split] = TARGET_DIR / "visualization" / split / "mask_overlay"
		dirs["segmented"][split] = TARGET_DIR / "visualization" / split / "segmented"
		for key in dirs:
			dirs[key][split].mkdir(parents=True, exist_ok=True)
	return dirs


def process_split(samples: list[Sample], split: str, dirs: dict[str, dict[str, Path]]) -> int:
	processed = 0
	for idx, sample in enumerate(samples, start=1):
		data = json.loads(sample.json_path.read_text(encoding="utf-8"))
		image_orig = load_image(sample, data)

		ann_width = int(data.get("imageWidth") or image_orig.size[0])
		ann_height = int(data.get("imageHeight") or image_orig.size[1])

		mask = create_mask(data.get("shapes", []), ann_width, ann_height)

		if image_orig.size != (ann_width, ann_height):
			image_aligned = image_orig.resize((ann_width, ann_height), Image.LANCZOS)
		else:
			image_aligned = image_orig

		image = resize_image(image_aligned)
		mask = resize_mask(mask)

		image_arr = np.array(image, dtype=np.uint8)
		overlay = generate_mask_overlay(image_arr, mask)
		if DRAW_OVERLAY_LEGEND:
			overlay = add_overlay_legend(overlay)
		segmented = generate_segmented(image_arr, mask)

		stem = sample.stem
		Image.fromarray(image_arr).save(dirs["img"][split] / f"{stem}.jpg", quality=95)
		Image.fromarray(mask, mode="L").save(dirs["ann"][split] / f"{stem}.png")
		Image.fromarray(overlay).save(dirs["overlay"][split] / f"{stem}.png")
		Image.fromarray(segmented).save(dirs["segmented"][split] / f"{stem}.png")

		processed += 1
		if idx % 100 == 0 or idx == len(samples):
			print(f"[{split}] {idx}/{len(samples)}")

	return processed


def main() -> None:
	if not SOURCE_DIR.exists():
		raise FileNotFoundError(f"源目录不存在: {SOURCE_DIR}")

	TARGET_DIR.mkdir(parents=True, exist_ok=True)

	samples = collect_samples()
	if not samples:
		raise RuntimeError("未找到可转换样本。")

	print(f"收集到样本: {len(samples)}")
	splits = split_samples(samples)
	dirs = ensure_dirs()

	counts = {}
	for split in ("train", "valid", "test"):
		counts[split] = process_split(splits[split], split, dirs)

	print("\n转换完成")
	print(f"输出目录: {TARGET_DIR.resolve()}")
	print(f"train={counts['train']}, valid={counts['valid']}, test={counts['test']}")
	print("类别定义: 0=background, " + ", ".join(f"{k}={v}" for k, v in CLASS_DEFECTS.items()))


if __name__ == "__main__":
	main()
