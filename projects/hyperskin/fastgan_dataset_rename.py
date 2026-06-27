from pathlib import Path
import csv
import shutil


# Source folders.
source_folders = [
    Path("/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/generated_samples_fastgan-melanoma"),
    Path("/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/generated_samples_fastgan-nevi"),
]

# Output dataset folder.
dataset_dir = Path("data/fastgan_dataset")

# Output image folders.
images_dir = dataset_dir / "images"
class_dirs = {
    "melanoma": images_dir / "MM",
    "nevi": images_dir / "DN",
}

# Create required folders.
class_dirs["melanoma"].mkdir(parents=True, exist_ok=True)
class_dirs["nevi"].mkdir(parents=True, exist_ok=True)
(images_dir / "OtherCube").mkdir(parents=True, exist_ok=True)

# Rows for the CSV files.
rename_rows = []
metadata_rows = []

# Process each source folder.
for source_dir in source_folders:

    # Get label from folder name:
    # generated_samples_fastgan-melanoma -> melanoma
    # generated_samples_fastgan-nevi -> nevi
    label = source_dir.name.split("fastgan-")[-1]

    # Get destination folder for this label.
    destination_dir = class_dirs[label]

    # Get files in deterministic order.
    files = sorted([file for file in source_dir.iterdir() if file.is_file()])

    # Rename files from 1 to N without repeated indexes.
    for index, old_path in enumerate(files, start=1):

        # Create new filename, keeping the original extension.
        new_filename = f"fastgan-{label}_{index}{old_path.suffix}"

        # Create final destination path.
        new_path = destination_dir / new_filename

        # Avoid overwriting existing files.
        if new_path.exists():
            raise FileExistsError(f"File already exists: {new_path}")

        # Move and rename the file.
        shutil.move(old_path, new_path)

        # Save old and new names.
        rename_rows.append({
            "old name": old_path.name,
            "new name": new_filename,
        })

        # Save metadata.
        metadata_rows.append({
            "file_path": new_path.as_posix(),
            "label": label,
            "masks": "",
            "filename": new_filename,
        })

# Save rename table.
with (dataset_dir / "rename_mapping.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["old name", "new name"])
    writer.writeheader()
    writer.writerows(rename_rows)

# Save metadata table.
with (dataset_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["file_path", "label", "masks", "filename"])
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"Done. Moved {len(metadata_rows)} files.")
print(f"Dataset created at: {dataset_dir}")