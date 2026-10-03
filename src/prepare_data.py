"""Extract BraTS 2020 training data from the Kaggle zip into compressed NIfTI files.

The zip contains uncompressed .nii files (~33 GB). Each file is read straight from the zip and written as
.nii.gz (~10x smaller), so the full dataset never has to exist on disk uncompressed.

Usage:
    python src/prepare_data.py --zip ~/Downloads/BraTS2020_TrainingData.zip
"""
from __future__ import annotations

import argparse
import gzip
import re
import shutil
import zipfile
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "MICCAI_BraTS2020_TrainingData/"
MODALITIES = ("flair", "t1", "t1ce", "t2", "seg")
# Case 355's mask is misnamed in the official release.
MISNAMED = {"BraTS20_Training_355/W39_1998.09.19_Segm.nii": "BraTS20_Training_355_seg"}


def target_name(member: str) -> str | None:
    """Return '<case>/<case>_<modality>' for an image member, or None for anything else."""
    relative = member[len(PREFIX):]
    if relative in MISNAMED:
        case = relative.split("/")[0]
        return f"{case}/{MISNAMED[relative]}"
    match = re.fullmatch(r"(BraTS20_Training_\d{3})/\1_(flair|t1|t1ce|t2|seg)\.nii", relative)
    return f"{match.group(1)}/{match.group(1)}_{match.group(2)}" if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, required=True, help="path to BraTS2020_TrainingData.zip")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "brats2020")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip.expanduser()) as archive:
        for csv_name in ("name_mapping.csv", "survival_info.csv"):
            with archive.open(PREFIX + csv_name) as src, open(args.out / csv_name, "wb") as dst:
                shutil.copyfileobj(src, dst)

        members = [(m, target_name(m)) for m in archive.namelist() if m.startswith(PREFIX)]
        members = [(m, t) for m, t in members if t]
        for member, name in tqdm(members, desc="extract"):
            destination = args.out / f"{name}.nii.gz"
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(".partial")
            with archive.open(member) as src, gzip.open(partial, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            partial.rename(destination)

    cases = sorted(p for p in args.out.iterdir() if p.is_dir())
    incomplete = [c.name for c in cases if len(list(c.glob("*.nii.gz"))) != len(MODALITIES)]
    print(f"{len(cases)} cases written to {args.out}")
    if incomplete:
        print("incomplete cases:", ", ".join(incomplete))


if __name__ == "__main__":
    main()
