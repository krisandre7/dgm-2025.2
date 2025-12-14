import os
import pandas as pd

# -------------------------------------------------------------------------
# Load the CSV that contains:
#   old_path : current filename on disk
#   new_path : the target filename to rename to
# -------------------------------------------------------------------------
csv_path = "scripts/renamed_path_mapping.csv"   # <-- Change if needed
df = pd.read_csv(csv_path)

# -------------------------------------------------------------------------
# Loop over every row and perform the rename
# -------------------------------------------------------------------------
for idx, row in df.iterrows():
    old_path = row["old_path"]
    new_path = row["new_path"]

    # Check existence of the old file before renaming
    if not os.path.exists(old_path):
        print(f"WARNING: File not found → {old_path}")
        continue

    # Make sure the destination directory exists
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    # Perform the rename
    try:
        print(f"Renaming:\n  {old_path}\n→ {new_path}")
        os.rename(old_path, new_path)
    except Exception as e:
        print(f"ERROR renaming {old_path} → {new_path}: {e}")

print("✔️ All renaming operations completed.")
