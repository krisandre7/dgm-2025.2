from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_DIR = Path(
    "/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/"
    "milk10k_dataset_full/unsegmented_milk10k/MM"
)

# This can be the filtered CSV you uploaded/generated, as long as it has:
# isic_id, lesion_id, and either label or diagnosis_full.
METADATA_CSV = Path("/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/milk10k_dataset_full/milk10k_melanoma_nevi_filtered.csv")

OUTPUT_ROOT = Path(
    "/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/data/"
    "milk10k_newly_cropped_melanoma"
)

CROP_SIZE = 256
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def normalize_image_id(value):
    """Accept ISIC_1234567, ISIC_1234567.jpg, or a path; return ISIC_1234567."""
    return Path(str(value).strip()).stem


def load_melanoma_id_to_lesion_id(metadata_csv):
    df = pd.read_csv(metadata_csv)

    required_cols = {"isic_id", "lesion_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{metadata_csv} is not enough for this crop layout. "
            f"Missing columns: {sorted(missing_cols)}"
        )

    # The filtered CSV already has label='melanoma'/'nevi'. If label is absent,
    # fall back to diagnosis_full from the merged metadata+supplement table.
    if "label" in df.columns:
        is_melanoma = df["label"].astype(str).str.lower().str.contains("melanoma", na=False)
    elif "diagnosis_full" in df.columns:
        is_melanoma = df["diagnosis_full"].astype(str).str.lower().str.contains("melanoma", na=False)
    else:
        raise ValueError(
            f"{metadata_csv} has isic_id and lesion_id, but no label or diagnosis_full. "
            "I cannot safely select only melanoma rows."
        )

    df = df.loc[is_melanoma, ["isic_id", "lesion_id"]].copy()
    df["isic_id"] = df["isic_id"].map(normalize_image_id)
    df["lesion_id"] = df["lesion_id"].astype(str).str.strip()
    df = df[(df["isic_id"] != "") & (df["lesion_id"] != "")]

    conflicting = df.groupby("isic_id")["lesion_id"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        examples = ", ".join(conflicting.index[:10])
        raise ValueError(f"Some isic_id values map to multiple lesion_id values: {examples}")

    df = df.drop_duplicates("isic_id")
    return dict(zip(df["isic_id"], df["lesion_id"]))


def build_image_index(image_dir):
    image_paths = {}

    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_id = normalize_image_id(path.name)
        if image_id in image_paths:
            print(f"[WARNING] duplicate image id in folder, keeping first: {image_id}")
            continue

        image_paths[image_id] = path

    return image_paths


def largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return np.zeros(mask.shape, dtype=bool)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def remove_dark_dermoscopy_border(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Removes the black circular dermoscopy border without hurting clinical images.
    valid = gray > 10
    valid = largest_component(valid)

    kernel = np.ones((9, 9), np.uint8)
    valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    if valid.sum() < gray.size * 0.25:
        # If the border removal got too aggressive, use the full image instead.
        return np.ones(gray.shape, dtype=bool)

    return valid.astype(bool)


def otsu_on_valid_pixels(channel, valid_mask):
    values = channel[valid_mask]
    if values.size < 100:
        return None

    threshold, _ = cv2.threshold(
        values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return threshold


def select_reasonable_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return None

    h, w = mask.shape
    image_area = h * w
    image_center = np.array([w / 2, h / 2])

    best_label = None
    best_score = -np.inf

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 100 or area > image_area * 0.70:
            continue

        centroid = np.array(centroids[label])
        center_distance = np.linalg.norm((centroid - image_center) / image_center)

        # Prefer large lesion-like components, with a mild preference for central lesions.
        score = area * (1.0 - 0.25 * center_distance)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None

    return labels == best_label


def segment_lesion(rgb):
    # ponytail: classical threshold segmentation. Known ceiling: very low-contrast
    # lesions, severe artifacts, or off-center multi-lesion images may fail. Upgrade
    # path: replace this function with a pretrained lesion segmentation model.
    valid_skin = remove_dark_dermoscopy_border(rgb)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lightness = lab[:, :, 0]
    saturation = hsv[:, :, 1]

    # Pigmented lesions are often darker and/or more saturated than surrounding skin.
    pigment_score = (255 - lightness).astype(np.float32) * 0.75 + saturation.astype(np.float32) * 0.25
    pigment_score = cv2.normalize(pigment_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    score_threshold = otsu_on_valid_pixels(pigment_score, valid_skin)
    lightness_threshold = otsu_on_valid_pixels(lightness, valid_skin)
    saturation_threshold = otsu_on_valid_pixels(saturation, valid_skin)

    if score_threshold is None or lightness_threshold is None or saturation_threshold is None:
        return None

    score_mask = pigment_score > score_threshold
    dark_mask = lightness < lightness_threshold
    saturated_mask = saturation > saturation_threshold

    mask = valid_skin & (score_mask | (dark_mask & saturated_mask))

    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((17, 17), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    mask = select_reasonable_component(mask)
    if mask is None:
        return None

    return mask.astype(np.uint8) * 255


def crop_start_for_axis(min_pos, max_pos, image_size, crop_size):
    if image_size <= crop_size:
        return 0

    centered_start = int(round((min_pos + max_pos + 1) / 2 - crop_size / 2))

    lesion_size = max_pos - min_pos + 1
    if lesion_size <= crop_size:
        # Pick any valid start that contains the full lesion bbox. Use the centered
        # one when possible, otherwise clamp to the interval that still contains it.
        lowest_start_that_contains_bbox = max(0, max_pos - crop_size + 1)
        highest_start_that_contains_bbox = min(min_pos, image_size - crop_size)
        return int(
            np.clip(
                centered_start,
                lowest_start_that_contains_bbox,
                highest_start_that_contains_bbox,
            )
        )

    return int(np.clip(centered_start, 0, image_size - crop_size))


def pad_to_size(array, size, value=0):
    h, w = array.shape[:2]
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)

    if pad_h == 0 and pad_w == 0:
        return array

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if array.ndim == 2:
        padding = ((top, bottom), (left, right))
    else:
        padding = ((top, bottom), (left, right), (0, 0))

    return np.pad(array, padding, mode="constant", constant_values=value)


def best_crop_start_by_mask(mask, crop_size):
    """Return the 256x256 crop start that keeps the most lesion-mask pixels."""
    lesion = (mask > 0).astype(np.uint8)
    total_lesion_pixels = int(lesion.sum())
    if total_lesion_pixels == 0:
        return None

    h, w = lesion.shape
    window_h = min(crop_size, h)
    window_w = min(crop_size, w)

    # Exact sliding-window count using an integral image. This is still cheap for
    # dermoscopy images and avoids a heuristic center crop when the lesion bbox is
    # larger than 256x256.
    integral = cv2.integral(lesion, sdepth=cv2.CV_32S)
    scores = (
        integral[window_h:, window_w:]
        - integral[:-window_h, window_w:]
        - integral[window_h:, :-window_w]
        + integral[:-window_h, :-window_w]
    )

    max_pixels_in_crop = int(scores.max())
    candidates_yx = np.argwhere(scores == max_pixels_in_crop)

    ys, xs = np.where(lesion > 0)
    lesion_center_xy = np.array([xs.mean(), ys.mean()], dtype=np.float32)
    crop_centers_xy = np.column_stack(
        (
            candidates_yx[:, 1] + window_w / 2.0,
            candidates_yx[:, 0] + window_h / 2.0,
        )
    )

    # Tie-breaker: if several windows keep the same number of lesion pixels,
    # choose the one whose crop center is closest to the lesion centroid.
    distances = np.sum((crop_centers_xy - lesion_center_xy) ** 2, axis=1)
    best_idx = int(np.argmin(distances))
    y1, x1 = candidates_yx[best_idx]

    return int(x1), int(y1), max_pixels_in_crop, total_lesion_pixels


def crop_around_mask(image, mask, crop_size=256):
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None, None, False

    h, w = mask.shape[:2]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    full_lesion_fits = (x_max - x_min + 1 <= crop_size) and (y_max - y_min + 1 <= crop_size)

    if full_lesion_fits:
        x1 = crop_start_for_axis(x_min, x_max, w, crop_size)
        y1 = crop_start_for_axis(y_min, y_max, h, crop_size)
    else:
        # The lesion bbox is too large for a 256x256 crop. Instead of using the
        # bbox center, choose the valid crop with the largest number of lesion
        # pixels inside it.
        best_crop = best_crop_start_by_mask(mask, crop_size)
        if best_crop is None:
            return None, None, False
        x1, y1, _, _ = best_crop

    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)

    image_crop = pad_to_size(image[y1:y2, x1:x2], crop_size, value=0)
    mask_crop = pad_to_size(mask[y1:y2, x1:x2], crop_size, value=0)

    return image_crop, mask_crop, full_lesion_fits


def self_check_crop_logic():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[20:50, 250:290] = 255

    image_crop, mask_crop, full_lesion_fits = crop_around_mask(image, mask, crop_size=256)

    assert image_crop.shape == (256, 256, 3)
    assert mask_crop.shape == (256, 256)
    assert full_lesion_fits
    assert mask_crop.sum() == mask.sum()

    # Non-trivial check: when the bbox is larger than 256, the crop must keep the
    # dense lesion region instead of blindly taking the bbox-centered crop.
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[20:276, 20:276] = 255
    mask[145:150, 276:350] = 255
    mask[130:170, 350:390] = 255

    image_crop, mask_crop, full_lesion_fits = crop_around_mask(image, mask, crop_size=256)

    assert image_crop.shape == (256, 256, 3)
    assert mask_crop.shape == (256, 256)
    assert not full_lesion_fits
    assert int((mask_crop > 0).sum()) == 256 * 256


def main():
    self_check_crop_logic()

    id_to_lesion_id = load_melanoma_id_to_lesion_id(METADATA_CSV)
    image_paths = build_image_index(IMAGE_DIR)

    warnings = []
    saved = 0
    missing_images = sorted(set(id_to_lesion_id) - set(image_paths))

    for idx, image_id in enumerate(sorted(set(id_to_lesion_id) & set(image_paths))):
        lesion_id = id_to_lesion_id[image_id]
        image_path = image_paths[image_id]

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            warnings.append((idx, image_id, "could not read image"))
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = segment_lesion(rgb)

        if mask is None:
            warnings.append((idx, image_id, "could not segment lesion"))
            continue

        image_crop, mask_crop, full_lesion_fits = crop_around_mask(rgb, mask, crop_size=CROP_SIZE)
        if image_crop is None or mask_crop is None:
            warnings.append((idx, image_id, "could not crop lesion"))
            continue

        if not full_lesion_fits:
            lesion_pixels = int((mask > 0).sum())
            kept_pixels = int((mask_crop > 0).sum())
            coverage = kept_pixels / max(1, lesion_pixels)
            warnings.append(
                (
                    idx,
                    image_id,
                    f"lesion bbox is larger than 256x256; saved max-lesion crop "
                    f"with {coverage:.1%} of lesion mask pixels",
                )
            )

        image_output_dir = OUTPUT_ROOT / "images" / lesion_id
        mask_output_dir = OUTPUT_ROOT / "masks" / lesion_id
        image_output_dir.mkdir(parents=True, exist_ok=True)
        mask_output_dir.mkdir(parents=True, exist_ok=True)

        image_output_path = image_output_dir / f"{image_id}_crop00.jpg"
        mask_output_path = mask_output_dir / f"{image_id}_crop00_mask.png"

        cv2.imwrite(str(image_output_path), cv2.cvtColor(image_crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_output_path), mask_crop)
        saved += 1

    print(f"Metadata melanoma rows: {len(id_to_lesion_id)}")
    print(f"Images found in folder: {len(image_paths)}")
    print(f"Saved crops: {saved}")
    print(f"Missing metadata images in folder: {len(missing_images)}")
    print(f"Warnings: {len(warnings)}")

    if missing_images:
        print("\nMissing image files:")
        for image_id in missing_images:
            print(f"[WARNING] missing image file for {image_id}")

    if warnings:
        print("\nSegmentation/crop warnings:")
        for idx, image_id, reason in warnings:
            print(f"[WARNING] idx={idx}, isic_id={image_id}: {reason}")


if __name__ == "__main__":
    main()
