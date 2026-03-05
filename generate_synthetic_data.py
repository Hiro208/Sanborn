import os
import cv2
import json
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
v4
- Purpose: stable v2/v3 variant for harder historical-style synthetic samples
- Upgrade: hue-locked jitter + very dark samples + huge-building/micro-shed mix
"""

# Load config
CONFIG_PATH = "config.json"
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

PATCH_H = config['train']['patch_size']
PATCH_W = int(PATCH_H * (7/8))

NUM_SAMPLES = 2000

# Base colors
COLOR_BASE_BG_RGB = (246, 241, 228)
COLOR_BASE_YELLOW_RGB = (241, 231, 142)
COLOR_BASE_PINK_RGB = (233, 150, 170)

config['color']['standard_bg'] = list(COLOR_BASE_BG_RGB)
config['color']['fallback']['brick'] = list(COLOR_BASE_PINK_RGB)
config['color']['fallback']['wood'] = list(COLOR_BASE_YELLOW_RGB)

OUTPUT_BASE = os.path.join(config['data']['output_dir'], 'synthetic_pretrain_v2')
IMG_OUT_DIR = os.path.join(OUTPUT_BASE, 'images')
MASK_OUT_DIR = os.path.join(OUTPUT_BASE, 'masks')
os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(MASK_OUT_DIR, exist_ok=True)

# Color jitter: lock hue and control brightness by mode

def random_color_jitter(base_color, mode='building'):
    """Avoid hue drift and keep brick/wood semantics stable."""
    if mode == 'bg':
        brightness = random.uniform(0.6, 1.05)
    else:
        brightness = random.uniform(0.35, 1.05)

    shifted = [max(0, min(255, int(c * brightness))) for c in base_color]
    shifted = [max(0, min(255, c + random.randint(-5, 5))) for c in shifted]
    return tuple(shifted)

# Building polygon
def generate_shape_poly(bounding_box):
    x, y, w, h = bounding_box
    shape_type = random.choice(['rect', 'L_shape', 'T_shape', 'H_shape'])

    margin = random.randint(1, 4)
    ix, iy, iw, ih = x + margin, y + margin, w - 2*margin, h - 2*margin
    if iw < 5 or ih < 5: return None

    poly = []
    if shape_type == 'rect':
        poly = [(ix, iy), (ix+iw, iy), (ix+iw, iy+ih), (ix, iy+ih)]
    elif shape_type == 'L_shape':
        cut_w, cut_h = int(iw * random.uniform(0.3, 0.7)), int(ih * random.uniform(0.3, 0.7))
        corner = random.choice(['tl', 'tr', 'bl', 'br'])
        if corner == 'tl': poly = [(ix+cut_w, iy), (ix+iw, iy), (ix+iw, iy+ih), (ix, iy+ih), (ix, iy+cut_h), (ix+cut_w, iy+cut_h)]
        elif corner == 'tr': poly = [(ix, iy), (ix+iw-cut_w, iy), (ix+iw-cut_w, iy+cut_h), (ix+iw, iy+cut_h), (ix+iw, iy+ih), (ix, iy+ih)]
        else: poly = [(ix+cut_w, iy), (ix+iw, iy), (ix+iw, iy+ih), (ix, iy+ih), (ix, iy+cut_h), (ix+cut_w, iy+cut_h)]
    elif shape_type == 'T_shape':
        cut_w = int(iw * random.uniform(0.15, 0.3))
        cut_h = int(ih * random.uniform(0.4, 0.6))
        poly = [(ix+cut_w, iy), (ix+iw-cut_w, iy), (ix+iw-cut_w, iy+cut_h), (ix+iw, iy+cut_h), (ix+iw, iy+ih), (ix, iy+ih), (ix, iy+cut_h), (ix+cut_w, iy+cut_h)]
    elif shape_type == 'H_shape':
        cut_w = int(iw * random.uniform(0.3, 0.4))
        cut_h = int(ih * random.uniform(0.3, 0.5))
        poly = [(ix, iy), (ix+cut_w, iy), (ix+cut_w, iy+cut_h), (ix+iw-cut_w, iy+cut_h), (ix+iw-cut_w, iy), (ix+iw, iy), (ix+iw, iy+ih), (ix, iy+ih)]

    return poly if len(poly) > 3 else None

def add_historic_aging(cv_img):
    h, w, _ = cv_img.shape

    # 1) Noise and stains
    noise = np.random.normal(0, 5, cv_img.shape).astype(np.float32)
    img_aged = cv2.add(cv_img.astype(np.float32), noise)

    s_vs_p = 0.5
    amount = 0.005
    num_salt = np.ceil(amount * (h * w) * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in cv_img.shape[:2]]
    img_aged[tuple(coords)] = [10, 10, 10]

    for _ in range(random.randint(0, 3)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(3, 10)
        cv2.circle(img_aged, (cx, cy), radius, (30,30,30), -1)

    # 2) Vignette
    Y = np.linspace(-1, 1, h)
    X = np.linspace(-1, 1, w)
    x, y = np.meshgrid(X, Y)
    vignette = np.sqrt(x*x + y*y)
    vignette = 1 - (vignette / vignette.max() * 0.3)
    vignette = np.expand_dims(vignette, axis=2)
    vignette = np.concatenate((vignette, vignette, vignette), axis=2)
    img_aged = (img_aged * vignette)

    # 3) Random global darkening
    if random.random() < 0.3:
        dark_scale = random.uniform(0.3, 0.6)
        img_aged = img_aged * dark_scale

    img_aged = np.clip(img_aged, 0, 255).astype(np.uint8)

    if random.random() > 0.5:
        img_aged = cv2.GaussianBlur(img_aged, (1, 1), 0)

    return img_aged

def place_buildings_on_grid(patch_w, patch_h):
    """Place buildings on a grid with overlap and size control."""
    density_type = random.choice(['sparse', 'medium', 'dense', 'industrial'])
    buildings = []
    occupied_boxes = []

    def check_overlap(box):
        x1, y1, w1, h1 = box
        for (ox, oy, ow, oh) in occupied_boxes:
            if not (x1+w1 < ox or x1 > ox+ow or y1+h1 < oy or y1 > oy+oh): return True
        return False

    # 1) Place extra-large buildings first
    if density_type in ['industrial', 'medium'] or random.random() < 0.3:
        num_huge = random.randint(1, 2)
        for _ in range(num_huge):
            hw = random.randint(int(patch_w * 0.4), int(patch_w * 0.95))
            hh = random.randint(int(patch_h * 0.4), int(patch_h * 0.95))
            hx = random.randint(5, max(10, patch_w - hw - 5))
            hy = random.randint(5, max(10, patch_h - hh - 5))

            box = (hx, hy, hw, hh)
            if not check_overlap(box):
                occupied_boxes.append(box)
                poly = generate_shape_poly(box)
                if poly: buildings.append((poly, 1 if random.random() < 0.7 else 2))

    # 2) Fill grid with medium/small buildings
    if density_type == 'sparse':
        grid_w, grid_h = random.randint(120, 180), random.randint(150, 200)
        build_prob = 0.4
    elif density_type == 'industrial':
        grid_w, grid_h = random.randint(100, 150), random.randint(120, 160)
        build_prob = 0.3
    else:
        grid_w, grid_h = random.randint(40, 90), random.randint(50, 110)
        build_prob = 0.85

    rows = patch_h // grid_h + 1
    cols = patch_w // grid_w + 1
    margin_p = 10

    for r in range(rows):
        for c in range(cols):
            lot_x = margin_p + c * grid_w + random.randint(-15, 15)
            lot_y = margin_p + r * grid_h + random.randint(-15, 15)
            lot_w = min(grid_w, patch_w - lot_x)
            lot_h = min(grid_h, patch_h - lot_y)

            if lot_w < 20 or lot_h < 20: continue

            if random.random() < build_prob:
                fill_ratio_w = random.uniform(0.3, 0.95)
                fill_ratio_h = random.uniform(0.3, 0.95)
                bw, bh = int(lot_w * fill_ratio_w), int(lot_h * fill_ratio_h)

                if bw < 15 or bh < 15: continue
                bx = lot_x + random.randint(2, max(3, lot_w - bw))
                by = lot_y + random.randint(2, max(3, lot_h - bh))

                box = (bx, by, bw, bh)
                if not check_overlap(box):
                    occupied_boxes.append(box)
                    poly = generate_shape_poly(box)
                    if poly: buildings.append((poly, 2 if random.random() < 0.7 else 1))

                    # 3) Optional micro sheds
                    if random.random() < 0.5:
                        shed_w, shed_h = random.randint(5, 15), random.randint(5, 15)
                        sx = bx + random.randint(-10, bw)
                        sy = by + bh + random.randint(2, 10)

                        shed_box = (sx, sy, shed_w, shed_h)
                        if (not check_overlap(shed_box)) and sx > 0 and sy > 0 and (sx+shed_w) < patch_w and (sy+shed_h) < patch_h:
                            occupied_boxes.append(shed_box)
                            shed_poly = [(sx, sy), (sx+shed_w, sy), (sx+shed_w, sy+shed_h), (sx, sy+shed_h)]
                            buildings.append((shed_poly, 2))

    return buildings, density_type

def add_noise_labels_and_streets(cv_img, patch_w, patch_h, density_type):
    text_color = (15, 15, 15)

    num_grid_lines = random.randint(3, 7) if density_type != 'dense' else random.randint(6, 12)
    for _ in range(num_grid_lines):
        x1, y1 = random.randint(0, patch_w), random.randint(0, patch_h)
        x2, y2 = random.randint(0, patch_w), random.randint(0, patch_h)
        if random.random() > 0.4: cv2.line(cv_img, (x1, y1), (x2, y2), (40, 40, 40), 1)
        else:
            cv2.line(cv_img, (x1, 0), (x1, patch_h), (40, 40, 40), 1)
            cv2.line(cv_img, (0, y1), (patch_w, y1), (40, 40, 40), 1)

    for _ in range(random.randint(5, 15)):
        cx, cy = random.randint(0, patch_w), random.randint(0, patch_h)
        cv2.line(cv_img, (cx, cy), (cx + random.randint(-20, 20), cy + random.randint(-20, 20)), text_color, 1)

    texts = ['D', 'S', 'F', 'C.', 'AUTO', 'STORE', 'ALABAMA', 'GEORGIA', '187', 'x', 'o']
    for _ in range(random.randint(15, 35)):
        txt = random.choice(texts)
        tx = random.randint(10, patch_w - 40)
        ty = random.randint(20, patch_h - 20)
        scale = random.uniform(0.5, 1.2)
        thickness = random.randint(1, 2)
        font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX])

        if random.random() > 0.8:
             cv2.putText(cv_img, txt, (tx, ty), font, scale, (30,30,30), thickness + 2, cv2.LINE_AA)
             cv2.putText(cv_img, txt, (tx, ty), font, scale, (255,255,255), thickness, cv2.LINE_AA)
        else:
            cv2.putText(cv_img, txt, (tx, ty), font, scale, text_color, thickness, cv2.LINE_AA)

    return cv_img

def generate_single_sample_poly(i):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    bg_jittered = random_color_jitter(COLOR_BASE_BG_RGB, mode='bg')
    pil_img = Image.new("RGB", (PATCH_W, PATCH_H), bg_jittered)
    pil_mask = Image.new("L", (PATCH_W, PATCH_H), 0)

    draw_img = ImageDraw.Draw(pil_img)
    draw_mask = ImageDraw.Draw(pil_mask)

    buildings, density_type = place_buildings_on_grid(PATCH_W, PATCH_H)

    for poly_points, class_id in buildings:
        base_color = COLOR_BASE_PINK_RGB if class_id == 1 else COLOR_BASE_YELLOW_RGB
        jitter_color = random_color_jitter(base_color, mode='building')

        try: pil_poly_points = [tuple(p) for p in poly_points]
        except TypeError: continue

        draw_img.polygon(pil_poly_points, fill=jitter_color)
        draw_img.polygon(pil_poly_points, outline=(40, 40, 40), width=1)
        draw_mask.polygon(pil_poly_points, fill=class_id)

    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_dirty = add_noise_labels_and_streets(cv_img, PATCH_W, PATCH_H, density_type)
    img_aged = add_historic_aging(img_dirty)
    mask_arr = np.array(pil_mask)

    img_filename = os.path.join(IMG_OUT_DIR, f"synth_{i:04d}.png")
    mask_filename = os.path.join(MASK_OUT_DIR, f"synth_{i:04d}.png")
    cv2.imwrite(img_filename, img_aged)
    cv2.imwrite(mask_filename, mask_arr)

    return i

def generate_dataset_parallel():
    max_workers = min(16, os.cpu_count() or 4)
    print(f"Start generating {NUM_SAMPLES} samples...")
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single_sample_poly, i) for i in range(NUM_SAMPLES)]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 200 == 0:
                print(f"Generated {completed} / {NUM_SAMPLES}")
    print("Done.")

if __name__ == "__main__":
    generate_dataset_parallel()