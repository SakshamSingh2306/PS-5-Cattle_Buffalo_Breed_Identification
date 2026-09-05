"""
yolo_breed_detector.py

Adds multi-animal detection on top of the existing single-image breed
classifier in app.py.

Pipeline for one uploaded photo:
  1. YOLO26 (Ultralytics, COCO-pretrained) scans the image and finds every
     "cow"-class box. COCO has no separate "buffalo" class, so this step
     only LOCALIZES animals — it does not yet know species or breed.
  2. Each cropped box is fed individually into the already-loaded breed
     classifier (predict_breed_topk from app.py) to get that animal's
     breed — and from the breed, its species (Cow or Buffalo).
  3. YOLO26's boxes are drawn back onto a copy of the original image, each
     one labeled and colored according to its own predicted breed, and a
     list of per-animal prediction dicts is returned for display.

Drop this file next to app.py and import from it (see the integration
points marked "NEW:" in app.py).
"""

import colorsys
import hashlib

import streamlit as st
from PIL import ImageDraw, ImageFont

# COCO detection class YOLO26 uses to localize cattle/buffalo in a photo.
COCO_COW_CLASS_NAME = "cow"

# Buffalo breeds the classifier can output (lowercase, spaces as
# underscores). Anything not in this set is treated as "Cow". Extend this
# if your classifier's classes.json includes more buffalo breeds.
BUFFALO_BREEDS = {
    "murrah", "nili_ravi", "nili-ravi", "jaffarabadi", "bhadawari", "surti",
    "mehsana", "nagpuri", "banni", "toda", "pandharpuri", "kalahandi",
    "chilika", "godavari", "marathwadi", "south_kanara",
}


@st.cache_resource
def load_yolo_model():
    """Loads YOLO26 (auto-downloads COCO-pretrained weights on first use,
    requires internet). Swap 'yolo26s.pt' for 'yolo26n.pt' (lighter/faster)
    or 'yolo26l.pt' / 'yolo26x.pt' (higher accuracy, needs a GPU)."""
    from ultralytics import YOLO
    return YOLO("yolo26s.pt")


def species_for_breed(breed_label):
    key = breed_label.lower().strip().replace(" ", "_")
    return "Buffalo" if key in BUFFALO_BREEDS else "Cow"


def color_for_breed(breed_label):
    """Deterministic color per breed name, so the same breed always gets
    the same color across boxes, the legend, and the stats table."""
    digest = hashlib.md5(breed_label.encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.48, 0.72)
    return (int(r * 255), int(g * 255), int(b * 255))


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def detect_boxes(image, yolo_model):
    """Returns a list of {"box": (x1,y1,x2,y2), "det_conf": float} for
    every cow/buffalo YOLO26 finds. Falls back to treating the whole image
    as a single box if nothing is detected or YOLO26 can't run (e.g. no
    internet to download weights on first use)."""
    width, height = image.size
    try:
        results = yolo_model.predict(source=image, verbose=False)[0]
        names = results.names
        boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            if cls_name.lower() != COCO_COW_CLASS_NAME:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            boxes.append({"box": (x1, y1, x2, y2), "det_conf": float(box.conf[0].item())})
        if not boxes:
            boxes = [{"box": (0, 0, width, height), "det_conf": None, "fallback": True}]
        return boxes
    except Exception:
        return [{"box": (0, 0, width, height), "det_conf": None, "fallback": True}]


def detect_and_classify(image, yolo_model, predict_breed_topk_fn, k=3):
    """
    Runs YOLO26 to find each animal, classifies each crop with the
    existing breed classifier, draws a labeled/colored box for each one on
    a copy of the image, and returns (annotated_image, detections).

    predict_breed_topk_fn: pass app.py's existing predict_breed_topk
    function in directly, so this module reuses the already-loaded
    classifier / transform / breed_labels instead of loading its own copy.
    """
    width, height = image.size
    boxes = detect_boxes(image, yolo_model)

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            max(14, width // 60),
        )
    except Exception:
        font = ImageFont.load_default()

    detections = []
    for idx, b in enumerate(boxes):
        x1, y1, x2, y2 = b["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop = image.crop((x1, y1, x2, y2))
        if crop.width < 2 or crop.height < 2:
            continue

        top_k = predict_breed_topk_fn(crop, k=k)
        if not top_k:
            continue
        breed, confidence = top_k[0]
        species = species_for_breed(breed)
        color = color_for_breed(breed)
        hex_color = rgb_to_hex(color)

        box_w = max(3, width // 400)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_w)
        label_text = f"{species}: {breed} ({confidence:.1f}%)"
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        pad = 4
        label_y = y1 - text_h - 2 * pad if y1 - text_h - 2 * pad > 0 else y1
        draw.rectangle([x1, label_y, x1 + text_w + 2 * pad, label_y + text_h + 2 * pad], fill=color)
        draw.text((x1 + pad, label_y + pad), label_text, fill="white", font=font)

        detections.append({
            "id": idx + 1,
            "breed": breed,
            "species": species,
            "confidence": confidence,
            "top_k": top_k,
            "detection_confidence": b["det_conf"] * 100 if b.get("det_conf") is not None else None,
            "box": (round(x1), round(y1), round(x2), round(y2)),
            "box_size": (round(x2 - x1), round(y2 - y1)),
            "color_hex": hex_color,
            "fallback_whole_image": bool(b.get("fallback")),
        })

    return annotated, detections
