from pathlib import Path
import os


SOURCE_DATASETS = [
    Path("/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/data/milk10k_melanoma_cropped_256"),
    Path("/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/data/milk10k_newly_cropped_melanoma"),
]

DEST_DATASET = Path(
    "/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/data/milk10k_melanoma_complete"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def expected_mask_name(image_path):
    """Convert ISIC_XXXXXXX_crop00.jpg -> ISIC_XXXXXXX_crop00_mask.png."""
    return f"{image_path.stem}_mask.png"


def list_images(images_root):
    """List all image files inside images/{lesion_id}/."""
    if not images_root.exists():
        raise FileNotFoundError(f"Missing images folder: {images_root}")

    return sorted(
        path for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def same_symlink_target(link_path, source_path):
    """Return True when link_path is a symlink already pointing to source_path."""
    if not link_path.is_symlink():
        return False

    current_target = Path(os.readlink(link_path))

    if not current_target.is_absolute():
        current_target = link_path.parent / current_target

    return current_target.resolve() == source_path.resolve()


def make_safe_symlink(source_path, link_path):
    """
    Create one symlink without overwriting existing files.

    Returns:
        "created" if a new symlink was created
        "exists" if the correct symlink already existed
        "conflict" if something else already exists there
    """
    source_path = source_path.resolve()
    link_path.parent.mkdir(parents=True, exist_ok=True)

    # os.path.lexists also catches broken symlinks; Path.exists does not.
    if os.path.lexists(link_path):
        if same_symlink_target(link_path, source_path):
            return "exists"

        print(f"[CONFLICT] Destination already exists and points somewhere else:")
        print(f"  destination: {link_path}")
        print(f"  wanted:      {source_path}")
        return "conflict"

    link_path.symlink_to(source_path)
    return "created"


def remove_empty_folders(root):
    """Remove empty lesion_id folders left behind, deepest first."""
    if not root.exists():
        return 0

    removed = 0

    for folder in sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            folder.rmdir()
            removed += 1
        except OSError:
            pass

    return removed


def main():
    assert expected_mask_name(Path("ISIC_1234567_crop00.jpg")) == "ISIC_1234567_crop00_mask.png"

    counts = {
        "image_links_created": 0,
        "mask_links_created": 0,
        "already_existing_valid_links": 0,
        "missing_masks": 0,
        "conflicts": 0,
    }

    for source_dataset in SOURCE_DATASETS:
        source_images_root = source_dataset / "images"
        source_masks_root = source_dataset / "masks"

        print(f"\nProcessing: {source_dataset}")

        for image_path in list_images(source_images_root):
            lesion_id = image_path.parent.name

            dest_image_path = DEST_DATASET / "images" / lesion_id / image_path.name

            image_status = make_safe_symlink(image_path, dest_image_path)

            if image_status == "created":
                counts["image_links_created"] += 1
            elif image_status == "exists":
                counts["already_existing_valid_links"] += 1
            else:
                counts["conflicts"] += 1

            mask_name = expected_mask_name(image_path)
            source_mask_path = source_masks_root / lesion_id / mask_name
            dest_mask_path = DEST_DATASET / "masks" / lesion_id / mask_name

            if not source_mask_path.exists():
                counts["missing_masks"] += 1
                print(f"[WARNING] Missing mask for image:")
                print(f"  image: {image_path}")
                print(f"  mask:  {source_mask_path}")
                continue

            mask_status = make_safe_symlink(source_mask_path, dest_mask_path)

            if mask_status == "created":
                counts["mask_links_created"] += 1
            elif mask_status == "exists":
                counts["already_existing_valid_links"] += 1
            else:
                counts["conflicts"] += 1

    empty_image_folders_removed = remove_empty_folders(DEST_DATASET / "images")
    empty_mask_folders_removed = remove_empty_folders(DEST_DATASET / "masks")

    print("\nDone.")
    print(f"Image symlinks created: {counts['image_links_created']}")
    print(f"Mask symlinks created: {counts['mask_links_created']}")
    print(f"Already-existing valid links: {counts['already_existing_valid_links']}")
    print(f"Missing masks: {counts['missing_masks']}")
    print(f"Conflicts: {counts['conflicts']}")
    print(f"Empty image folders removed: {empty_image_folders_removed}")
    print(f"Empty mask folders removed: {empty_mask_folders_removed}")
    print(f"\nComplete dataset saved at: {DEST_DATASET}")


if __name__ == "__main__":
    main()