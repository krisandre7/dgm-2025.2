from pathlib import Path
import pandas as pd


from pathlib import Path
import pandas as pd


METADATA_CSV = Path("/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/data/milk10k_melanoma_cropped_256/MILK10k_Training_Metadata.csv")
OUTPUT_CSV = Path("/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/milk10k_dataset_full/milk10k_melanoma_nevi_filtered.csv")

SUPPLEMENT_CSV = Path("/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/milk10k_dataset_full/MILK10k_Training_Supplement.csv")


NEVI_EXCLUDE_CSV = Path(
    "/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/data/"
    "milk10k_melanoma_nevus_cropped_256/nevi_image_names.csv"
)

MELANOMA_EXCLUDE_CSV = Path(
    "/mnt/datassd/genai/dgm-2025.2/projects/hyperskin/data/"
    "milk10k_melanoma_cropped_256/melanoma_image_names.csv"
)


def read_image_names(csv_path):
    names = pd.read_csv(csv_path).iloc[:, 0]

    # Accepts either "ISIC_123.jpg" or "ISIC_123".
    return set(
        names.astype(str)
        .str.strip()
        .str.replace(r"\.\w+$", "", regex=True)
    )


excluded_image_ids = read_image_names(NEVI_EXCLUDE_CSV) | read_image_names(MELANOMA_EXCLUDE_CSV)

metadata = pd.read_csv(METADATA_CSV)
supplement = pd.read_csv(SUPPLEMENT_CSV)

df = metadata.merge(supplement, on="isic_id", how="inner", validate="one_to_one")

diagnosis = df["diagnosis_full"].astype(str).str.lower()

df["label"] = None
df.loc[diagnosis.str.contains("melanoma", na=False), "label"] = "melanoma"
df.loc[diagnosis.str.contains(r"\bnevus\b|\bnevi\b", na=False), "label"] = "nevi"

filtered = df[
    df["label"].notna()
    & ~df["isic_id"].isin(excluded_image_ids)
].copy()

filtered.to_csv(OUTPUT_CSV, index=False)

print("Excluded image IDs:", len(excluded_image_ids))
print()
print(filtered["label"].value_counts())
print(f"\nSaved to: {OUTPUT_CSV}")