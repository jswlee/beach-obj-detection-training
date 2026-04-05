"""
Preprocessing utilities for the beach object-detection pipeline.

Responsibilities:
  - ROI masking (blacking-out pool / boardwalk areas on qualifying images)
  - Image tiling (slicing full-resolution frames into 640×640 patches)
  - Full YOLO-dataset processing (mask → tile → remap labels → write data.yaml)

All polygon / overlap constants are imported from config.py so they stay in
one place.
"""

import cv2
import numpy as np
import shutil
import yaml
from pathlib import Path
from functools import lru_cache
from tqdm import tqdm

# Project-level constants (edit in config.py, not here)
from config import (
    EXCLUSION_POLYGON,
    TILE_SIZE,
    OVERLAP_FULL,
    OVERLAP_YOUTUBE,
    MIN_OVERLAP_RATIO,
)


# ---------------------------------------------------------------------------
# Filename classification helpers
# ---------------------------------------------------------------------------

def is_youtube_snapshot(img_filename: str) -> bool:
    """Return True if the image filename starts with 'youtube_snapshot'."""
    return img_filename.lower().startswith("youtube_snapshot")


def should_apply_mask(img_filename: str) -> bool:
    """Decide whether this image should receive the ROI exclusion mask.

    Rules (only youtube_snapshot images are candidates):
      - youtube_snapshot_<DIGITS>…  → mask (timestamp-style name)
      - youtube_snapshot_Kaanapali… → mask
      - youtube_snapshot_<OTHER>…   → NO mask (other beach locations)
    """
    name = img_filename.lower()
    prefix = "youtube_snapshot_"
    if not name.startswith(prefix):
        return False

    suffix = name[len(prefix):]
    if not suffix:
        return False

    # Timestamp-style (starts with a digit)
    if suffix[0].isdigit():
        return True

    # Kaanapali beach
    if suffix.startswith("kaanapali"):
        return True

    return False


# ---------------------------------------------------------------------------
# ROI mask creation
# ---------------------------------------------------------------------------

def create_roi_mask(image_shape: tuple) -> np.ndarray:
    """Create a binary mask (0 / 1 uint8) for the region of interest.

    The exclusion polygon is filled with 0; everything else is 1.

    Args:
        image_shape: (height, width) or (height, width, channels).

    Returns:
        2-D uint8 array with 1 inside the ROI, 0 outside.
    """
    height, width = image_shape[:2]
    mask = np.ones((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [EXCLUSION_POLYGON], 0)
    return mask


@lru_cache(maxsize=16)
def get_roi_mask_u8(img_h: int, img_w: int) -> np.ndarray:
    """Cached ROI mask scaled to 0/255 for use with cv2.bitwise_and."""
    return (create_roi_mask((img_h, img_w)) * 255).astype(np.uint8)


def is_box_in_exclusion_zone(
    x_norm: float, y_norm: float,
    width_px: int, height_px: int,
    mask: np.ndarray,
) -> bool:
    """Return True if the normalised box centre falls in the masked-out area."""
    cx = max(0, min(int(x_norm * width_px), width_px - 1))
    cy = max(0, min(int(y_norm * height_px), height_px - 1))
    return mask[cy, cx] == 0


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def get_tiles(img_w: int, img_h: int, tile_size: int, overlap: float) -> list[tuple[int, int]]:
    """Generate (x, y) tile offsets covering the full image with overlap.

    Each tuple is the top-left corner of a tile_size × tile_size crop.
    Tiles at the right / bottom edge are clamped so they don't exceed the
    image boundary.
    """
    stride = int(tile_size * (1 - overlap))
    tiles: list[tuple[int, int]] = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x_eff = min(x, img_w - tile_size)
            y_eff = min(y, img_h - tile_size)
            tiles.append((x_eff, y_eff))
            if x + tile_size >= img_w:
                break
            x += stride
        if y + tile_size >= img_h:
            break
        y += stride
    return tiles


def get_bottom_three_tiles(img_w: int, img_h: int, tile_size: int) -> list[tuple[int, int]]:
    """Return exactly three horizontally-spaced tiles aligned to the image bottom.

    Layout: left edge, centre, right edge – all at the same y offset.
    Used for youtube_snapshot frames where only the bottom band contains
    the beach.
    """
    if img_w <= tile_size:
        x_left = x_mid = x_right = 0
    else:
        x_left = 0
        x_right = img_w - tile_size
        x_mid = int(round((x_left + x_right) / 2.0))

    y0 = max(0, img_h - tile_size)
    return [(x_left, y0), (x_mid, y0), (x_right, y0)]


def get_tiles_for_image(
    img_filename: str, img_w: int, img_h: int, tile_size: int,
) -> list[tuple[int, int]]:
    """Return the correct tile layout for a given image.

    YouTube snapshots → 3 bottom tiles.
    Everything else   → full-resolution overlapping grid.
    """
    if is_youtube_snapshot(img_filename):
        return get_bottom_three_tiles(img_w, img_h, tile_size)
    return get_tiles(img_w, img_h, tile_size, OVERLAP_FULL)


def crop_and_mask_tiles(
    img_bgr: np.ndarray,
    tile_size: int,
    img_filename: str,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Crop tiles from the image, applying the ROI mask where appropriate.

    Returns:
        tile_imgs: list of BGR tile arrays (each tile_size × tile_size)
        offsets:   corresponding (x, y) offsets in the original image
    """
    h, w = img_bgr.shape[:2]
    tiles = get_tiles_for_image(img_filename, w, h, tile_size)
    apply_mask = should_apply_mask(img_filename)

    mask_u8 = get_roi_mask_u8(h, w) if apply_mask else None

    tile_imgs: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    for tx, ty in tiles:
        tile = img_bgr[ty : ty + tile_size, tx : tx + tile_size]
        if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
            continue
        if mask_u8 is not None:
            tile_mask = mask_u8[ty : ty + tile_size, tx : tx + tile_size]
            tile = cv2.bitwise_and(tile, tile, mask=tile_mask)
        tile_imgs.append(tile)
        offsets.append((tx, ty))
    return tile_imgs, offsets


# ---------------------------------------------------------------------------
# Full YOLO dataset processing (preprocess → optional tile → write)
# ---------------------------------------------------------------------------

def process_yolo_dataset(
    input_dir: str,
    output_dir: str,
    enable_slicing: bool = False,
    person_only: bool = False,
    person_class_id: int = 3,
    img_size: int = TILE_SIZE,
    min_pixel_size: int = 0,
) -> str:
    """Process a Roboflow YOLO export into a training-ready dataset.

    Steps for each image:
      1. Optionally apply the ROI exclusion mask (youtube_snapshot images).
      2. Filter labels (person-only, min-size, exclusion-zone).
      3. Either copy the whole image or slice it into tiles, remapping labels.
      4. Write a new data.yaml pointing at the output directory.

    Args:
        input_dir:       Path to raw Roboflow dataset (train/valid/test splits).
        output_dir:      Where to write the processed dataset.
        enable_slicing:  If True, tile images into img_size × img_size patches.
        person_only:     If True, keep only person-class annotations.
        person_class_id: Class ID for "person" in the raw dataset.
        img_size:        Tile size in pixels (only used when slicing).
        min_pixel_size:  Drop annotations whose sqrt(w*h) is below this.

    Returns:
        Absolute path to the generated data.yaml.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Start fresh
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    print(f"\nProcessing mode: {'SLICING' if enable_slicing else 'NO-SLICE'}")
    print(f"Person-only filtering: {'ENABLED' if person_only else 'DISABLED'}")
    if min_pixel_size > 0:
        print(f"Min pixel size filter: removing annotations < {min_pixel_size}px")
    print(f"Selective masking: youtube_snapshot_ images only\n")

    # Counters for the summary printed at the end
    tiles_written = 0
    label_files_written = 0
    boxes_kept = 0
    boxes_dropped = 0
    boxes_filtered_small = 0

    for split in ["train", "valid", "test"]:
        img_src = input_path / split / "images"
        lbl_src = input_path / split / "labels"
        img_dst = output_path / split / "images"
        lbl_dst = output_path / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        if not img_src.exists():
            continue

        print(f"Processing {split}...")
        image_files = list(img_src.glob("*.jpg")) + list(img_src.glob("*.png"))

        for img_file in tqdm(image_files):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            h, w = img.shape[:2]

            # --- Selective masking ---
            apply_mask = should_apply_mask(img_file.name)
            if apply_mask:
                mask = create_roi_mask((h, w))
                img_processed = img.copy()
                img_processed[mask == 0] = 0
            else:
                mask = None
                img_processed = img.copy()

            # --- Load and filter labels ---
            boxes: list[tuple[int, float, float, float, float]] = []
            lbl_file = lbl_src / f"{img_file.stem}.txt"
            if lbl_file.exists():
                with open(lbl_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            cls = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                        except (ValueError, IndexError):
                            continue

                        if person_only and cls != person_class_id:
                            continue

                        if apply_mask and mask is not None:
                            if is_box_in_exclusion_zone(cx, cy, w, h, mask):
                                continue

                        if min_pixel_size > 0:
                            size_px = (bw * w * bh * h) ** 0.5
                            if size_px < min_pixel_size:
                                boxes_filtered_small += 1
                                continue

                        boxes.append((cls, cx, cy, bw, bh))

            # --- Write outputs ---
            if enable_slicing:
                tiles = get_tiles_for_image(img_file.name, w, h, img_size)
                for tile_idx, (tx, ty) in enumerate(tiles):
                    tile_img = img_processed[ty : ty + img_size, tx : tx + img_size]
                    tile_boxes: list[str] = []

                    for cls, bx, by, bw_n, bh_n in boxes:
                        abs_cx, abs_cy = bx * w, by * h
                        abs_w, abs_h = bw_n * w, bh_n * h
                        box_x1 = abs_cx - abs_w / 2.0
                        box_y1 = abs_cy - abs_h / 2.0
                        box_x2 = abs_cx + abs_w / 2.0
                        box_y2 = abs_cy + abs_h / 2.0

                        # Intersect with tile bounds
                        ix1 = max(box_x1, tx)
                        iy1 = max(box_y1, ty)
                        ix2 = min(box_x2, tx + img_size)
                        iy2 = min(box_y2, ty + img_size)
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue

                        original_area = abs_w * abs_h
                        if original_area <= 0:
                            continue
                        if (ix2 - ix1) * (iy2 - iy1) / original_area < MIN_OVERLAP_RATIO:
                            continue

                        # Convert to tile-local normalised coords
                        new_cx = ((ix1 - tx + ix2 - tx) / 2.0) / img_size
                        new_cy = ((iy1 - ty + iy2 - ty) / 2.0) / img_size
                        new_w = (ix2 - ix1) / img_size
                        new_h = (iy2 - iy1) / img_size

                        # Clamp
                        new_cx = max(0.0, min(1.0, new_cx))
                        new_cy = max(0.0, min(1.0, new_cy))
                        new_w = max(0.0, min(1.0, new_w))
                        new_h = max(0.0, min(1.0, new_h))

                        if new_w > 0 and new_h > 0:
                            out_cls = 0 if person_only else cls
                            tile_boxes.append(
                                f"{out_cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"
                            )
                            boxes_kept += 1
                        else:
                            boxes_dropped += 1

                    tile_name = f"{img_file.stem}_tile_{tile_idx}.jpg"
                    cv2.imwrite(str(img_dst / tile_name), tile_img)
                    tiles_written += 1
                    if tile_boxes:
                        (lbl_dst / f"{img_file.stem}_tile_{tile_idx}.txt").write_text(
                            "\n".join(tile_boxes)
                        )
                        label_files_written += 1
            else:
                # NO-SLICE: copy image and filtered labels directly
                cv2.imwrite(str(img_dst / img_file.name), img_processed)
                if boxes:
                    out_lines = []
                    for cls, cx, cy, bw_n, bh_n in boxes:
                        out_cls = 0 if person_only else cls
                        out_lines.append(
                            f"{out_cls} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}"
                        )
                    (lbl_dst / f"{img_file.stem}.txt").write_text("\n".join(out_lines))

    # --- Write data.yaml ---
    orig_yaml = input_path / "data.yaml"
    if orig_yaml.exists():
        with open(orig_yaml) as f:
            data = yaml.safe_load(f)
        data["path"] = str(output_path.absolute())
        data["train"] = str(output_path / "train" / "images")
        data["val"] = str(output_path / "valid" / "images")
        data["test"] = str(output_path / "test" / "images")
        if person_only:
            data["nc"] = 1
            data["names"] = ["person"]
        with open(output_path / "data.yaml", "w") as f:
            yaml.dump(data, f, sort_keys=False)

    # --- Summary ---
    mode_str = "sliced" if enable_slicing else "processed"
    print(f"\n✓ {mode_str.capitalize()} dataset created at: {output_path}")
    if min_pixel_size > 0:
        print(f"  Annotations filtered (< {min_pixel_size}px): {boxes_filtered_small}")
    if enable_slicing:
        print(f"  Tiles written: {tiles_written}")
        print(f"  Label files written: {label_files_written}")
        print(f"  Boxes kept: {boxes_kept}")
        print(f"  Boxes dropped: {boxes_dropped}")

    return str(output_path / "data.yaml")
