"""
Mini‑CLEVR Dataset Generator
===========================
Generates a synthetic VQA dataset of simple 2-D "CLEVR-like" scenes with
multiple colored shapes.  For each image it automatically creates a set of
questions (property / counting / spatial‑relation) together with answers,
compatible with common VQA loaders.

Usage (from CLI) -----------------------------------------------------------
python generate_mini_clevr.py --n_images 8000 --output_dir ./mini_clevr \
       --train_ratio 0.8 --val_ratio 0.1 --img_size 224
---------------------------------------------------------------------------
Output structure -----------------------------------------------------------
mini_clevr/
 ├─ images/
 │   ├─ train/000001.png
 │   ├─ val/...
 │   └─ test/...
 ├─ train.jsonl   (CLEVR‑style QA records)
 ├─ val.jsonl
 ├─ test.jsonl
 └─ answer2idx.json
---------------------------------------------------------------------------
Each *.jsonl line contains::
  {
    "image": "images/train/000001.png",
    "question": "What color is the sphere?",
    "answer": "red",
    "type": "property"
  }
---------------------------------------------------------------------------
The script is **stand-alone** (depends only on Pillow & NumPy) and avoids
object overlap to keep questions valid.
"""

import argparse, json, math, os, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

# GLOBALS CONFIGS #
SHAPES = ["circle", "square", "triangle", "pentagon"]
COLORS = {
    "red": (220, 20, 60),
    "blue": (65, 105, 225),
    "green": (60, 179, 113),
    "yellow": (250, 215, 0),
    "purple": (138, 43, 226),
    "cyan": (0, 206, 209),
}
SIZE_RANGE = {"small": (14, 18), "large": (24, 30)}  # radius / half‑side px

# Spatial relation threshold (px) when deciding left/right/above/below
REL_MARGIN = 4

def parse_args():
    p = argparse.ArgumentParser(description="Generate a mini CLEVR‑style VQA dataset.")
    p.add_argument("--n_images", type=int, default=8000)
    p.add_argument("--output_dir", type=str, default="./mini_clevr")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    return p.parse_args()

# GEOMETRY UTILS.
def overlaps(cx: int, cy: int, r: int, others: List[Tuple[int, int, int]]) -> bool:
    """Return True if circle (cx,cy,r) overlaps any in *others* (list of (x,y,r))."""
    for ox, oy, orad in others:
        dist = math.hypot(cx - ox, cy - oy)
        if dist < r + orad + 2:  # 2‑px buffer
            return True
    return False

# DRAW PRIMITIVES
def draw_shape(draw: ImageDraw.ImageDraw, shape: str, cx: int, cy: int, rad: int, color: Tuple[int, int, int]):
    if shape == "circle":
        bbox = [cx - rad, cy - rad, cx + rad, cy + rad]
        draw.ellipse(bbox, fill=color)
    elif shape == "square":
        bbox = [cx - rad, cy - rad, cx + rad, cy + rad]
        draw.rectangle(bbox, fill=color)
    elif shape == "triangle":
        pts = [
            (cx, cy - rad),
            (cx - rad, cy + rad),
            (cx + rad, cy + rad),
        ]
        draw.polygon(pts, fill=color)
    elif shape == "pentagon":
        pts = [
            (cx + rad * math.sin(2 * math.pi * i / 5), cy - rad * math.cos(2 * math.pi * i / 5))
            for i in range(5)
        ]
        draw.polygon(pts, fill=color)
    else:
        raise ValueError(f"Unknown shape {shape}")

# QUESTION TEXT GENERATION
def make_property_q(obj):
    q = f"What color is the {obj['shape']}?"
    return {"question": q, "answer": obj["color"], "type": "property"}


def make_count_q(color, objects):
    cnt = sum(1 for o in objects if o["color"] == color)
    q = f"How many {color} objects are there?"
    return {"question": q, "answer": str(cnt), "type": "count"}


def make_relation_q(o1, o2):
    # Ask either 'left of' or 'right of' with 50 % probability.
    ask_left = random.choice([True, False])      
    rel_word = "left" if ask_left else "right"

    q = f"Is the {o1['color']} {o1['shape']} to the {rel_word} " \
        f"of the {o2['color']} {o2['shape']}?"

    cond_left = o1["cx"] < o2["cx"] - REL_MARGIN
    ans = "yes" if (cond_left and ask_left) or (not cond_left and not ask_left) else "no"
    return {"question": q, "answer": ans, "type": "relation"}

# MAIN
def generate_dataset(args):
    random.seed(42)
    np.random.seed(42)
    out_dir = Path(args.output_dir)
    img_root = out_dir / "images"
    img_root.mkdir(parents=True, exist_ok=True)

    # Prepare splits
    n_train = int(args.n_images * args.train_ratio)
    n_val = int(args.n_images * args.val_ratio)
    splits = (["train"] * n_train) + (["val"] * n_val) + (["test"] * (args.n_images - n_train - n_val))
    random.shuffle(splits)

    # Answer vocab (collect on the fly)
    answer_vocab = set()

    # JSONL writers
    writers = {s: open(out_dir / f"{s}.jsonl", "w", encoding="utf8") for s in ["train", "val", "test"]}

    for idx, split in enumerate(splits, 1):
        W, H = args.img_size, args.img_size
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        objects = []  # store dicts with attributes + position
        n_obj = random.randint(3, 6)
        attempts = 0
        while len(objects) < n_obj and attempts < 100:
            attempts += 1
            shape = random.choice(SHAPES)
            color_name, color_rgb = random.choice(list(COLORS.items()))
            size_key = random.choice(list(SIZE_RANGE.keys()))
            rad = random.randint(*SIZE_RANGE[size_key])
            cx = random.randint(rad + 1, W - rad - 1)
            cy = random.randint(rad + 1, H - rad - 1)
            if overlaps(cx, cy, rad, [(o["cx"], o["cy"], o["rad"]) for o in objects]):
                continue
            draw_shape(draw, shape, cx, cy, rad, color_rgb)
            objects.append({
                "shape": shape,
                "color": color_name,
                "size": size_key,
                "cx": cx,
                "cy": cy,
                "rad": rad,
            })

        # Save image
        split_dir = img_root / split
        split_dir.mkdir(exist_ok=True)
        img_filename = f"{idx:06d}.png"
        img.save(split_dir / img_filename)

        # Generate QAs
        qas = []
        # a) property (1 per image)
        qas.append(make_property_q(random.choice(objects)))
        # b) count questions (2 colors)
        colors_in_scene = list({o["color"] for o in objects})
        for col in random.sample(colors_in_scene, k=min(2, len(colors_in_scene))):
            qas.append(make_count_q(col, objects))
        # c) relation (1 per image)
        if len(objects) >= 2:
            o1, o2 = random.sample(objects, 2)
            qas.append(make_relation_q(o1, o2))

        # Write JSONL lines
        for qa in qas:
            record = {
                "image": str(Path("images") / split / img_filename),
                **qa,
            }
            writers[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            answer_vocab.add(qa["answer"])

    # Close writers
    for w in writers.values():
        w.close()

    # Save answer vocabulary
    vocab = {ans: i for i, ans in enumerate(sorted(answer_vocab))}
    with open(out_dir / "answer2idx.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"DONE! Generated {args.n_images} images in '{out_dir}'.")


if __name__ == "__main__":
    generate_dataset(parse_args())
