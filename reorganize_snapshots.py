"""
Script to reorganize existing snapshots from flat structure to organized structure.
Organizes by violation type and date.

Old format: NO_HELMET_1_20260706_152712_551759.jpg
New format: NO_HELMET/2026-07-06/Track_1_152712_551759.jpg
"""

import os
import shutil
from datetime import datetime
import re

SNAPSHOTS_DIR = "data/snapshots"

def reorganize_snapshots():
    """
    Move snapshots from flat directory structure to organized by type and date.
    """
    if not os.path.exists(SNAPSHOTS_DIR):
        print(f"Snapshots directory not found: {SNAPSHOTS_DIR}")
        return

    # Find all .jpg files in the snapshots root
    files = [f for f in os.listdir(SNAPSHOTS_DIR) if f.endswith('.jpg')]
    
    if not files:
        print("No snapshots to reorganize.")
        return

    print(f"Found {len(files)} snapshots to reorganize...")
    
    moved_count = 0
    
    for filename in files:
        try:
            # Parse filename: e.g., "NO_HELMET_1_20260706_152712_551759.jpg"
            parts = filename.replace('.jpg', '').split('_')
            
            # Extract violation type (can be multiple words with underscores)
            if filename.startswith("NO_HELMET"):
                violation_type = "NO_HELMET"
                rest = "_".join(parts[2:])  # Remove "NO_HELMET_"
            elif filename.startswith("TRIPLE_RIDING"):
                violation_type = "TRIPLE_RIDING"
                rest = "_".join(parts[2:])  # Remove "TRIPLE_RIDING_"
            elif filename.startswith("WRONG_WAY"):
                violation_type = "WRONG_WAY"
                rest = "_".join(parts[2:])  # Remove "WRONG_WAY_"
            else:
                print(f"⚠️ Skipping unrecognized format: {filename}")
                continue
            
            # Extract track_id and timestamp
            # Format: TRACK_ID_YYYYMMDD_HHMMSS_FFFFFF
            match = re.match(r'(\d+)_(\d{8})_(\d{6})_(\d+)', rest)
            if not match:
                print(f"⚠️ Could not parse: {filename}")
                continue
            
            track_id, date_str, time_str, microsec = match.groups()
            
            # Convert date format: 20260706 → 2026-07-06
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            date_folder = date_obj.strftime("%Y-%m-%d")
            
            # Create target directory
            target_dir = os.path.join(SNAPSHOTS_DIR, violation_type, date_folder)
            os.makedirs(target_dir, exist_ok=True)
            
            # New filename: Track_1_152712_551759.jpg
            new_filename = f"Track_{track_id}_{time_str}_{microsec}.jpg"
            
            # Move file
            src_path = os.path.join(SNAPSHOTS_DIR, filename)
            dst_path = os.path.join(target_dir, new_filename)
            
            shutil.move(src_path, dst_path)
            moved_count += 1
            print(f"✓ {filename} → {violation_type}/{date_folder}/{new_filename}")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
    
    print(f"\n✓ Successfully reorganized {moved_count} snapshots!")
    print(f"\nNew structure:")
    print(f"  data/snapshots/")
    print(f"  ├── NO_HELMET/")
    print(f"  │   ├── 2026-07-06/")
    print(f"  │   ├── 2026-07-07/")
    print(f"  │   └── 2026-07-08/")
    print(f"  ├── TRIPLE_RIDING/")
    print(f"  │   ├── 2026-07-06/")
    print(f"  │   └── 2026-07-08/")
    print(f"  └── WRONG_WAY/")
    print(f"      └── ...")

if __name__ == "__main__":
    reorganize_snapshots()
