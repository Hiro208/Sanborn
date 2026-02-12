import os
import cv2
import json
import numpy as np
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
v1 
- Purpose: generate a basic synthetic training set (buildings + line/text occlusion + light noise)
"""

# Load config
CONFIG_PATH = "config.json"
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

PATCH_SIZE = config['train']['patch_size']
NUM_SAMPLES = 2000

BG_COLOR = tuple(config['color']['standard_bg'])
COLOR_BRICK = tuple(config['color']['fallback']['brick'])
COLOR_WOOD = tuple(config['color']['fallback']['wood'])

OUTPUT_BASE = os.path.join(config['data']['output_dir'], 'synthetic_pretrain')
IMG_OUT_DIR = os.path.join(OUTPUT_BASE, 'images')
MASK_OUT_DIR = os.path.join(OUTPUT_BASE, 'masks')
os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(MASK_OUT_DIR, exist_ok=True)


# Color jitter
def random_color_jitter(base_color, max_shift=30):
    shifted = [
        max(0, min(255, c + random.randint(-max_shift, max_shift)))
        for c in base_color
    ]
    return tuple(shifted)


# Draw one building (rectangle or L-shape)
def draw_building(img, mask, class_id, base_color):
    color = random_color_jitter(base_color)
    shape_type = random.choice(['rect', 'rect', 'L_shape'])

    x = random.randint(50, PATCH_SIZE - 150)
    y = random.randint(50, PATCH_SIZE - 150)
    w = random.randint(40, 120)
    h = random.randint(40, 120)

    if shape_type == 'rect':
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(mask, (x, y), (x + w, y + h), class_id, -1)
    else:
        w2 = random.randint(20, w)
        h2 = random.randint(h, h + 60)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w2, y + h2), color, -1)
        cv2.rectangle(mask, (x, y), (x + w, y + h), class_id, -1)
        cv2.rectangle(mask, (x, y), (x + w2, y + h2), class_id, -1)


# Add lines, text, and noise
def add_noise_and_occlusions(img):
    h, w, _ = img.shape
    text_color = (15, 15, 15)

    for _ in range(random.randint(2, 6)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        cv2.line(img, (x1, y1), (x2, y2), text_color, random.randint(1, 2))

    for _ in range(random.randint(5, 15)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        cv2.line(img, (cx, cy), (cx + random.randint(-20, 20), cy + random.randint(-20, 20)), text_color, 1)

    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]
    texts = ['D', 'S', 'F', '187', '2', 'AUTO', 'STORE', 'ALABAMA', 'x', 'o']

    for _ in range(random.randint(10, 25)):
        txt = random.choice(texts)
        tx = random.randint(10, w - 40)
        ty = random.randint(20, h - 20)
        font = random.choice(fonts)
        scale = random.uniform(0.5, 1.2)
        thickness = random.randint(1, 2)
        cv2.putText(img, txt, (tx, ty), font, scale, text_color, thickness, cv2.LINE_AA)

    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img_noisy = cv2.add(img.astype(np.float32), noise)
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

    if random.random() > 0.5:
        img_noisy = cv2.GaussianBlur(img_noisy, (3, 3), 0)

    return img_noisy


# Generate one sample
def generate_single_sample(i):
    # Per-process seed to avoid duplicate samples
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    bg_jittered = random_color_jitter(BG_COLOR, max_shift=15)
    img = np.full((PATCH_SIZE, PATCH_SIZE, 3), bg_jittered, dtype=np.uint8)
    mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

    num_buildings = random.randint(4, 12)
    for _ in range(num_buildings):
        building_type = random.choice(['brick', 'wood'])
        if building_type == 'brick':
            draw_building(img, mask, class_id=1, base_color=COLOR_BRICK)
        else:
            draw_building(img, mask, class_id=2, base_color=COLOR_WOOD)

    img_dirty = add_noise_and_occlusions(img)
    img_bgr = cv2.cvtColor(img_dirty, cv2.COLOR_RGB2BGR)

    img_filename = os.path.join(IMG_OUT_DIR, f"synth_{i:04d}.png")
    mask_filename = os.path.join(MASK_OUT_DIR, f"synth_{i:04d}.png")

    cv2.imwrite(img_filename, img_bgr)
    cv2.imwrite(mask_filename, mask)

    return i


# Generate dataset in parallel
def generate_dataset_parallel():
    max_workers = min(16, os.cpu_count() or 4)
    print(f"Start generating {NUM_SAMPLES} synthetic samples")
    print(f"Workers: {max_workers}")
    print(f"Output: {OUTPUT_BASE}")

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single_sample, i) for i in range(NUM_SAMPLES)]

        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 200 == 0:
                print(f"Generated {completed} / {NUM_SAMPLES}")

    print("Done")


if __name__ == "__main__":
    generate_dataset_parallel()