"""
Data Processing Script for HMM Activity Recognition Project
Collected by: KERIE (iPhone 11, Sensor Logger app, 100Hz sampling rate)

Extracts zip files from unprocessed/, merges accelerometer + gyroscope data,
labels by activity, and saves clean CSVs.
"""

import os
import zipfile
import pandas as pd
import glob
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNPROCESSED_DIR = os.path.join(BASE_DIR, "unprocessed")
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMP_DIR = os.path.join(BASE_DIR, "_temp_extract")

# Create output directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "still"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "standing"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "walking"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "jumping"), exist_ok=True)

def classify_activity(filename):
    """Determine activity label from zip filename."""
    fname = filename.lower()
    if fname.startswith("still"):
        return "still"
    elif fname.startswith("standing"):
        return "standing"
    elif fname.startswith("walking"):
        return "walking"
    elif fname.startswith("jumping"):
        return "jumping"
    return None

def get_sample_number(filename):
    """Extract sample number from filename like 'walking1-...' or 'walking_2-...'."""
    fname = filename.split("-")[0]  # e.g., 'walking1', 'walking_2', 'still_2'
    # Remove activity prefix and underscore to get number
    for prefix in ["jumping", "walking", "standing", "still"]:
        if fname.lower().startswith(prefix):
            num_part = fname[len(prefix):].strip("_")
            try:
                return int(num_part)
            except ValueError:
                return 0
    return 0

def process_zip(zip_path, activity, sample_num):
    """Extract zip, merge accel + gyro, return labeled DataFrame."""
    extract_dir = os.path.join(TEMP_DIR, f"{activity}_{sample_num}")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

    accel_path = os.path.join(extract_dir, "Accelerometer.csv")
    gyro_path = os.path.join(extract_dir, "Gyroscope.csv")
    meta_path = os.path.join(extract_dir, "Metadata.csv")

    if not os.path.exists(accel_path) or not os.path.exists(gyro_path):
        print(f"  WARNING: Missing Accelerometer or Gyroscope CSV in {zip_path}")
        return None

    # Read accelerometer data
    accel = pd.read_csv(accel_path)
    accel = accel.rename(columns={
        'x': 'accel_x',
        'y': 'accel_y',
        'z': 'accel_z'
    })

    # Read gyroscope data
    gyro = pd.read_csv(gyro_path)
    gyro = gyro.rename(columns={
        'x': 'gyro_x',
        'y': 'gyro_y',
        'z': 'gyro_z'
    })

    # Merge on timestamp (nearest match since they may not align perfectly)
    # Both have 'time' and 'seconds_elapsed' columns
    # Use seconds_elapsed for merging via merge_asof
    accel = accel.sort_values('seconds_elapsed').reset_index(drop=True)
    gyro = gyro.sort_values('seconds_elapsed').reset_index(drop=True)

    merged = pd.merge_asof(
        accel[['time', 'seconds_elapsed', 'accel_x', 'accel_y', 'accel_z']],
        gyro[['seconds_elapsed', 'gyro_x', 'gyro_y', 'gyro_z']],
        on='seconds_elapsed',
        direction='nearest',
        tolerance=0.02  # 20ms tolerance
    )

    # Drop rows where merge failed
    merged = merged.dropna()

    # Add labels
    merged['activity'] = activity
    merged['sample_id'] = f"{activity}_{sample_num}"

    # Read metadata for device info
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        if 'device name' in meta.columns:
            merged['device'] = meta['device name'].iloc[0]
        if 'sampleRateMs' in meta.columns:
            merged['sample_rate_ms'] = meta['sampleRateMs'].iloc[0].split('|')[0]

    return merged

# Process all zip files
print("=" * 60)
print("Processing Sensor Logger Data")
print("Collector: KERIE | Device: iPhone 11 | Rate: 100Hz (10ms)")
print("=" * 60)

all_data = []
zip_files = sorted(glob.glob(os.path.join(UNPROCESSED_DIR, "*.zip")))

# Track duplicates (files with "(1)" suffix)
seen = set()
skipped_duplicates = []

for zip_path in zip_files:
    fname = os.path.basename(zip_path)

    # Skip duplicate files (those with "(1)" in name)
    if "(1)" in fname:
        skipped_duplicates.append(fname)
        continue

    activity = classify_activity(fname)
    if activity is None:
        print(f"  SKIP: Cannot classify {fname}")
        continue

    sample_num = get_sample_number(fname)
    key = f"{activity}_{sample_num}"

    if key in seen:
        print(f"  SKIP duplicate: {fname}")
        continue
    seen.add(key)

    print(f"  Processing: {fname} -> {activity} sample {sample_num}")
    df = process_zip(zip_path, activity, sample_num)

    if df is not None:
        # Save individual CSV
        out_path = os.path.join(DATA_DIR, activity, f"{activity}_{sample_num}.csv")
        df.to_csv(out_path, index=False)
        all_data.append(df)
        print(f"    -> Saved {len(df)} rows to {activity}/{activity}_{sample_num}.csv")

# Combine all into one dataset
print("\n" + "=" * 60)
if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(os.path.join(DATA_DIR, "all_activities_combined.csv"), index=False)
    print(f"Combined dataset: {len(combined)} total rows")

    # Summary
    print("\nData Summary:")
    print("-" * 45)
    summary = combined.groupby('activity').agg(
        num_samples=('sample_id', 'nunique'),
        total_rows=('activity', 'count'),
        duration_sec=('seconds_elapsed', lambda x: x.max() - x.min())
    ).round(2)
    print(summary)
    print(f"\nTotal unique recordings: {combined['sample_id'].nunique()}")
    print(f"Total data points: {len(combined)}")

if skipped_duplicates:
    print(f"\nSkipped {len(skipped_duplicates)} duplicate files: {skipped_duplicates}")

# Clean up temp directory
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

print("\nDone! Data saved to data/ directory.")
print("=" * 60)
