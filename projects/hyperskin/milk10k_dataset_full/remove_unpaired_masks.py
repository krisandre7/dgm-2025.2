from pathlib import Path  # Imports Path to handle filesystem paths cleanly.
import pandas as pd  # Imports pandas to save the deletion log as a CSV file.


OUTPUT_ROOT = Path(  # Defines the root folder that contains both images/ and masks/.
    "/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/data/"  # First part of the output root path.
    "milk10k_newly_cropped_melanoma"  # Final folder name of the cropped melanoma dataset.
)

IMAGE_DIR = OUTPUT_ROOT / "images"  # Defines the folder where the manually cleaned cropped images are stored.
MASK_DIR = OUTPUT_ROOT / "masks"  # Defines the folder where the cropped masks are stored.
DELETED_LOG_CSV = OUTPUT_ROOT / "deleted_masks_without_images.csv"  # Defines where the deletion log CSV will be saved.

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}  # Allowed image file extensions.
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}  # Allowed mask file extensions.


def isic_id_from_crop_stem(stem):  # Defines a helper to recover the ISIC id from a crop filename stem.
    return stem.replace("_mask", "").split("_crop")[0]  # Removes mask suffix and crop suffix, leaving only ISIC_XXXXXXX.


def crop_key_from_image_path(path):  # Defines a helper to create a matching key from an image path.
    lesion_id = path.parent.name  # Gets the lesion_id from the parent folder name.
    crop_id = path.stem  # Gets the crop id from the filename without extension, e.g. ISIC_XXXXXXX_crop00.
    return lesion_id, crop_id  # Returns a pair that uniquely identifies this cropped image.


def crop_key_from_mask_path(path):  # Defines a helper to create a matching key from a mask path.
    lesion_id = path.parent.name  # Gets the lesion_id from the parent folder name.
    crop_id = path.stem.replace("_mask", "")  # Removes _mask so ISIC_XXXXXXX_crop00_mask matches ISIC_XXXXXXX_crop00.
    return lesion_id, crop_id  # Returns a pair that uniquely identifies the corresponding cropped image.


def list_files(root, extensions):  # Defines a helper to recursively list files with specific extensions.
    if not root.exists():  # Checks whether the requested folder exists.
        raise FileNotFoundError(f"Folder does not exist: {root}")  # Stops early with a clear error if the folder is missing.

    return sorted(  # Returns the matching files in deterministic sorted order.
        path for path in root.rglob("*")  # Recursively iterates over everything inside the root folder.
        if path.is_file() and path.suffix.lower() in extensions  # Keeps only files with allowed extensions.
    )


def remove_empty_lesion_folders(root):  # Defines a helper to delete empty lesion_id folders.
    deleted_dirs = []  # Stores the paths of folders that were deleted.

    for folder in sorted(  # Iterates over folders in a safe deletion order.
        (p for p in root.rglob("*") if p.is_dir()),  # Recursively finds all folders inside the root folder.
        key=lambda p: len(p.parts),  # Sorts folders by path depth.
        reverse=True,  # Deletes deepest folders first so parent folders can become empty afterward.
    ):
        try:  # Tries to delete the folder.
            folder.rmdir()  # Deletes the folder only if it is empty.
            deleted_dirs.append(str(folder))  # Saves the deleted folder path in the report list.
        except OSError:  # Handles folders that are not empty or cannot be removed.
            pass  # Keeps non-empty folders without failing the script.

    return deleted_dirs  # Returns the list of deleted empty folders.


def main():  # Defines the main script logic.
    image_paths = list_files(IMAGE_DIR, IMAGE_EXTENSIONS)  # Lists all remaining manually approved cropped images.
    mask_paths = list_files(MASK_DIR, MASK_EXTENSIONS)  # Lists all cropped masks currently present.

    print(f"Images found: {len(image_paths)}")  # Prints how many cropped images were found.
    print(f"Masks found: {len(mask_paths)}")  # Prints how many cropped masks were found.

    image_keys = {crop_key_from_image_path(path) for path in image_paths}  # Builds a set of valid image keys for fast matching.

    deleted_rows = []  # Stores one CSV row for each deleted mask.

    for mask_path in mask_paths:  # Iterates over every mask file.
        mask_key = crop_key_from_mask_path(mask_path)  # Converts the mask path into its comparable image key.

        if mask_key in image_keys:  # Checks whether the corresponding cropped image still exists.
            continue  # Keeps the mask and moves to the next one.

        lesion_id, crop_id = mask_key  # Splits the mask key into lesion_id and crop_id.
        isic_id = isic_id_from_crop_stem(crop_id)  # Extracts the ISIC id from the crop id.

        deleted_rows.append(  # Adds this deleted mask to the CSV log.
            {  # Starts the dictionary representing one CSV row.
                "lesion_id": lesion_id,  # Stores the lesion id folder where the mask was located.
                "isic_id": isic_id,  # Stores the ISIC id of the deleted mask.
                "crop_id": crop_id,  # Stores the crop id, e.g. ISIC_XXXXXXX_crop00.
                "deleted_mask_path": str(mask_path),  # Stores the full path of the deleted mask.
                "reason": "mask has no corresponding cropped image",  # Stores why this mask was deleted.
            }  # Ends the dictionary representing one CSV row.
        )  # Ends the append call.

        mask_path.unlink()  # Deletes the mask file from disk.

    deleted_folders = remove_empty_lesion_folders(MASK_DIR)  # Deletes empty lesion_id folders inside the masks folder.

    deleted_df = pd.DataFrame(  # Creates a pandas DataFrame for the deletion log.
        deleted_rows,  # Uses the list of deleted-mask records as data.
        columns=["lesion_id", "isic_id", "crop_id", "deleted_mask_path", "reason"],  # Fixes column order in the CSV.
    )  # Ends DataFrame creation.
    deleted_df.to_csv(DELETED_LOG_CSV, index=False)  # Saves the deletion log CSV without the pandas index column.

    print(f"Deleted masks: {len(deleted_rows)}")  # Prints how many masks were deleted.
    print(f"Deleted empty mask folders: {len(deleted_folders)}")  # Prints how many empty folders were deleted.
    print(f"Saved deletion log to: {DELETED_LOG_CSV}")  # Prints where the deletion log CSV was saved.

    if deleted_folders:  # Checks whether any empty mask folders were deleted.
        print("\nDeleted empty folders:")  # Prints a section header for deleted folders.
        for folder in deleted_folders:  # Iterates over each deleted folder path.
            print(folder)  # Prints the deleted folder path.


if __name__ == "__main__":  # Runs main only when this file is executed directly.
    main()  # Starts the script.