from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import shutil


FILTERED_CSV = Path("/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/milk10k_dataset_full/milk10k_melanoma_nevi_filtered.csv")
ZIP_PATH = Path("/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/milk10k_dataset_full/MILK10k_Training_Input.zip")  # change this
OUTPUT_DIR = Path("/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/milk10k_dataset_full/unsegmented_milk10k")


def main():
    # Create output folders.
    mm_dir = OUTPUT_DIR / "MM"
    dn_dir = OUTPUT_DIR / "DN"
    mm_dir.mkdir(parents=True, exist_ok=True)
    dn_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FILTERED_CSV)

    required_cols = {"isic_id", "label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    # Map each image id to its destination folder.
    label_to_folder = {
        "melanoma": mm_dir,
        "nevi": dn_dir,
    }

    df = df[df["label"].isin(label_to_folder)].copy()
    id_to_folder = {
        row["isic_id"]: label_to_folder[row["label"]]
        for _, row in df.iterrows()
    }

    expected_ids = set(id_to_folder)
    found_ids = set()
    extracted_counts = {"melanoma": 0, "nevi": 0}

    # Extract only wanted files.
    with ZipFile(ZIP_PATH) as zf:
        for zip_info in zf.infolist():
            if zip_info.is_dir():
                continue

            file_path = Path(zip_info.filename)
            image_id = file_path.stem

            if image_id not in id_to_folder:
                continue

            output_path = id_to_folder[image_id] / file_path.name

            # Extract only the file itself, not the full internal zip path.
            with zf.open(zip_info) as src, output_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

            found_ids.add(image_id)

    # Count extracted images by class.
    extracted_df = df[df["isic_id"].isin(found_ids)]
    extracted_counts = extracted_df["label"].value_counts().to_dict()

    missing_ids = expected_ids - found_ids

    print(f"Expected images: {len(expected_ids)}")
    print(f"Extracted images: {len(found_ids)}")
    print(f"Missing images: {len(missing_ids)}")
    print()
    print(f"Melanoma (MM): {extracted_counts.get('melanoma', 0)}")
    print(f"Nevi (DN): {extracted_counts.get('nevi', 0)}")

    if missing_ids:
        print("\nMissing IDs:")
        for image_id in sorted(missing_ids):
            print(image_id)


if __name__ == "__main__":
    main()