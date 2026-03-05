"""
Data Processing Script for HMM Activity Recognition Project

Collected by:
- Relebohile  | iPhone 12 Pro Max | Sensor Logger | 100Hz  →  Jumping (1).zip style
- Kerie        | iPhone 11         | Sensor Logger | 100Hz  →  jumping1-2026-... style

Kerie duplicates (e.g. still_2-2026-....(1).zip) are automatically skipped.
All 32 unique recordings processed → 80% train / 20% test split.
"""

import os, re, zipfile, shutil, glob
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
UNPROCESSED_DIR = os.path.join(BASE_DIR, "unprocessed")
DATA_DIR        = os.path.join(BASE_DIR, "data")
TEMP_DIR        = os.path.join(BASE_DIR, "_temp_extract")

for folder in ["still", "standing", "walking", "jumping"]:
    os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)

    #  Clean old CSV files from previous runs 
old_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
for f in old_files:
    os.remove(f)
print(f"  Cleaned {len(old_files)} old CSV(s) from previous runs\n")



def classify_activity(filename):
    fname = filename.lower()
    for act in ["still", "standing", "walking", "jumping", "jump"]:
        if act in fname:
            return "jumping" if act == "jump" else act
    return None


def is_kerie_duplicate(filename):
    """
    Kerie's duplicates have BOTH a date pattern AND parens, e.g.:
    still_2-2026-03-03_18-32-40 (1).zip  ← skip
    Jumping (1).zip                        ← keep (Relebohile, no date)
    """
    has_date   = bool(re.search(r'-\d{4}-\d{2}-\d{2}', filename))
    has_parens = bool(re.search(r'\(\d+\)', filename))
    return has_date and has_parens


def make_sample_id(zip_filename, activity):
    """
    Relebohile: 'Jumping (1).zip'              -> jumping_rele_1
    Kerie:      'jumping1-2026-03-04_....zip'  -> jumping_kerie_1
                'jumping_5-2026-03-04_....zip' -> jumping_kerie_5
                'standing-1_2026-03-03_....zip'-> standing_kerie_1
    """
    stem = os.path.splitext(zip_filename)[0]

    # Relebohile — Activity (N), no date in name
    if re.search(r'\(\d+\)', stem) and not re.search(r'-\d{4}-', stem):
        n = re.findall(r'\d+', stem)[-1]
        return f"{activity}_rele_{n}"

    # Kerie — has a date string
    if re.search(r'-\d{4}-', stem):
        # grab the part before the first date segment
        prefix = re.split(r'-\d{4}-', stem)[0]       # e.g. "jumping1", "jumping_5", "standing-1_2026"
        nums   = re.findall(r'\d+', prefix)
        n      = nums[-1] if nums else "0"
        return f"{activity}_kerie_{n}"

    # Fallback
    safe = re.sub(r'[^a-z0-9]', '_', stem.lower())
    return f"{activity}_{safe}"


def process_zip(zip_path, activity, sample_id):
    extract_dir = os.path.join(TEMP_DIR, sample_id)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

    accel_path = os.path.join(extract_dir, "Accelerometer.csv")
    gyro_path  = os.path.join(extract_dir, "Gyroscope.csv")
    meta_path  = os.path.join(extract_dir, "Metadata.csv")

    if not os.path.exists(accel_path) or not os.path.exists(gyro_path):
        print(f"  ⚠  Missing sensor CSV in {zip_path} — skipping")
        return None

    accel = pd.read_csv(accel_path).rename(columns={'x':'accel_x','y':'accel_y','z':'accel_z'})
    gyro  = pd.read_csv(gyro_path ).rename(columns={'x':'gyro_x', 'y':'gyro_y', 'z':'gyro_z'})

    accel = accel.sort_values('seconds_elapsed').reset_index(drop=True)
    gyro  = gyro .sort_values('seconds_elapsed').reset_index(drop=True)

    merged = pd.merge_asof(
        accel[['time','seconds_elapsed','accel_x','accel_y','accel_z']],
        gyro [['seconds_elapsed','gyro_x','gyro_y','gyro_z']],
        on='seconds_elapsed', direction='nearest', tolerance=0.02
    ).dropna()

    merged['activity']  = activity
    merged['sample_id'] = sample_id

    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        if 'device name'  in meta.columns: merged['device']         = meta['device name'].iloc[0]
        if 'sampleRateMs' in meta.columns: merged['sample_rate_ms'] = str(meta['sampleRateMs'].iloc[0]).split('|')[0]

    return merged


# Main
print("=" * 65)
print("  HMM Activity Recognition — Data Processing")
print("  Relebohile (iPhone 12 Pro Max) | Kerie (iPhone 11) | 100Hz")
print("=" * 65)

all_data, seen_ids = [], set()
skipped_dupes      = []

for zip_path in sorted(glob.glob(os.path.join(UNPROCESSED_DIR, "*.zip"))):
    fname    = os.path.basename(zip_path)
    activity = classify_activity(fname)

    if not activity:
        print(f"  SKIP (unrecognised) : {fname}")
        continue

    if is_kerie_duplicate(fname):
        skipped_dupes.append(fname)
        print(f"  SKIP (duplicate)    : {fname}")
        continue

    sample_id = make_sample_id(fname, activity)

    if sample_id in seen_ids:
        print(f"  SKIP (same ID)      : {fname}  →  {sample_id}")
        continue
    seen_ids.add(sample_id)

    print(f"  {fname:52s}  →  {sample_id}")
    df = process_zip(zip_path, activity, sample_id)

    if df is not None:
        df.to_csv(os.path.join(DATA_DIR, activity, f"{sample_id}.csv"), index=False)
        all_data.append(df)
        print(f"      {len(df):>5} rows saved")

#  Combine & split 
print("\n" + "=" * 65)

if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(os.path.join(DATA_DIR, "all_activities_combined.csv"), index=False)

    print(f"\nTotal: {len(combined):,} rows | {combined['sample_id'].nunique()} recordings\n")
    print(combined.groupby('activity').agg(
        recordings = ('sample_id',       'nunique'),
        total_rows = ('activity',         'count'),
        duration_s = ('seconds_elapsed',  lambda x: round(x.max()-x.min(), 2))
    ).to_string())

    # 80/20 stratified split at recording level
    sample_labels = combined[['sample_id','activity']].drop_duplicates()
    train_ids, test_ids = train_test_split(
        sample_labels['sample_id'],
        test_size=0.20, stratify=sample_labels['activity'], random_state=42
    )

    train_df = combined[combined['sample_id'].isin(train_ids)].reset_index(drop=True)
    test_df  = combined[combined['sample_id'].isin(test_ids) ].reset_index(drop=True)

    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    test_df .to_csv(os.path.join(DATA_DIR, "test.csv"),  index=False)

    print(f"\n80/20 Split (stratified by activity, recording-level):")
    print(f"  Train : {len(train_df):,} rows | {train_df['sample_id'].nunique()} recordings")
    print(f"  Test  : {len(test_df) :,} rows | {test_df ['sample_id'].nunique()} recordings")
    print(f"\n  Train per activity:\n{train_df.groupby('activity')['sample_id'].nunique().to_string()}")
    print(f"\n  Test  per activity:\n{test_df .groupby('activity')['sample_id'].nunique().to_string()}")

    if skipped_dupes:
        print(f"\n  Skipped {len(skipped_dupes)} Kerie duplicates: {skipped_dupes}")

if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

print("\n Done")
print("=" * 65)