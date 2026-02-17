#!/usr/bin/env python3
"""
Reorganize RobotPose dataset:
  FROM: /source/G001T000A000R000/atlas/exoL/frames/frame_000000.png
  TO:   /dest/G001T000A000R000/g1/frame_00001.png
"""

import shutil
from pathlib import Path

# === CONFIGURE THESE ===
SOURCE_DIR = Path("/path/to/datasets/RobotPose/processed_data")
DEST_DIR = Path("../../exo_dataset")
USE_SYMLINKS = True  # Set True to create symlinks instead of copying

# Map source subdirs to destination folder names
MAPPING = {
    "atlas/exoL/frames": "atlas/exoL",
    "nao/exoL/frames": "nao/exoL",
    "g1/exoL/frames": "g1/exoL",
    "icub/exoL/frames": "icub/exoL",
    "pepper/exoL/frames": "pepper/exoL",
    # Add more as needed
}
# =======================

for session in sorted(SOURCE_DIR.iterdir()):
    if not session.is_dir() or not session.name.startswith("G"):
        continue
    
    for src_subdir, dest_name in MAPPING.items():
        frames_dir = session / src_subdir
        if not frames_dir.exists():
            continue
        
        dest_dir = DEST_DIR / session.name / dest_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        for frame in sorted(frames_dir.glob("frame_*.png")):
            # Convert frame_000000.png -> frame_00000.png (0-indexed)
            num = int(frame.stem.split("_")[1])
            new_name = f"frame_{num:05d}.png"
            dest_file = dest_dir / new_name
            
            if not dest_file.exists():
                if USE_SYMLINKS:
                    dest_file.symlink_to(frame.resolve())
                else:
                    shutil.copy2(frame, dest_file)
                print(f"{frame} -> {dest_file}")

print("Done!")