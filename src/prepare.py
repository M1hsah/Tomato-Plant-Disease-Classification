"""Build data/manifest.csv: the single table every later step reads.

Steps (each is one function below):
  1. Scan PlantVillage + PlantDoc folders and check image counts against the
     numbers we verified from the repos (fail loudly if anything changed).
  2. Perceptual-hash (phash) every image and record its size.
  3. Group PlantVillage images by physical leaf, so the same leaf can never end
     up in both train and test (that would inflate test scores).
  4. Deduplicate PlantDoc, and drop PlantDoc images that also appear in
     PlantVillage (they would contaminate the field evaluation).
  5. Make a group-aware, stratified 80/10/10 PlantVillage split.

Usage:
    python -m src.prepare --data-dir data/raw --out data/manifest.csv
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display (Colab / servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Classes: the single source of truth, also imported by train.py.
# The label index of each class is its position in CLASSES, so do not reorder
# this list once experiments have started (saved checkpoints depend on it).
# Spider mites (2 PlantDoc images) and target spot (no PlantDoc images) are
# left out on purpose because they cannot be evaluated on field data.
# --------------------------------------------------------------------------- #
# (canonical label, PlantVillage folder in raw/color/, PlantDoc folder in train/ and test/)
CLASSES = [
    ("bacterial_spot", "Tomato___Bacterial_spot", "Tomato leaf bacterial spot"),
    ("early_blight", "Tomato___Early_blight", "Tomato Early blight leaf"),
    ("late_blight", "Tomato___Late_blight", "Tomato leaf late blight"),
    ("leaf_mold", "Tomato___Leaf_Mold", "Tomato mold leaf"),
    ("septoria_leaf_spot", "Tomato___Septoria_leaf_spot", "Tomato Septoria leaf spot"),
    ("mosaic_virus", "Tomato___Tomato_mosaic_virus", "Tomato leaf mosaic virus"),
    ("yellow_leaf_curl_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus"),
    ("healthy", "Tomato___healthy", "Tomato leaf"),
]
LABELS = [label for label, _, _ in CLASSES]
NUM_CLASSES = len(LABELS)
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# label -> (PlantVillage, PlantDoc train, PlantDoc test), counted before dedup.
EXPECTED_COUNTS = {
    "bacterial_spot": (2127, 101, 9),
    "early_blight": (1000, 79, 9),
    "late_blight": (1909, 101, 10),
    "leaf_mold": (952, 85, 6),
    "septoria_leaf_spot": (1771, 140, 11),
    "mosaic_virus": (373, 44, 10),
    "yellow_leaf_curl_virus": (5357, 70, 6),
    "healthy": (1591, 55, 8),
}

MANIFEST_COLUMNS = [
    "path", "source", "label", "label_idx", "orig_split", "group_id",
    "phash", "is_duplicate", "split", "width", "height",
]


# --------------------------------------------------------------------------- #
# 1. Scanning and count check
# --------------------------------------------------------------------------- #
def list_images(folder: Path) -> list[Path]:
    """Sorted image files in a folder (sorted so the manifest is reproducible)."""
    if not folder.is_dir():
        sys.exit(f"ERROR: missing folder: '{folder}'")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def scan_datasets(data_dir: Path) -> pd.DataFrame:
    """One row per image. `path` is relative to data_dir so the manifest is portable."""
    rows = []
    for label, pv_folder, pd_folder in CLASSES:
        for p in list_images(data_dir / "plantvillage" / "raw" / "color" / pv_folder):
            rows.append({"path": p, "source": "plantvillage", "label": label, "orig_split": ""})
        for split in ("train", "test"):
            for p in list_images(data_dir / "plantdoc" / split / pd_folder):
                rows.append({"path": p, "source": "plantdoc", "label": label, "orig_split": split})
    df = pd.DataFrame(rows)
    df["path"] = [p.relative_to(data_dir).as_posix() for p in df["path"]]
    df["label_idx"] = df["label"].map(LABEL_TO_IDX)
    return df


def check_counts(df: pd.DataFrame) -> None:
    """Print per-class counts and exit if any differ from EXPECTED_COUNTS."""
    print(f"\n{'label':<24}{'PV':>7}{'PD train':>10}{'PD test':>9}")
    mismatches = []
    for label in LABELS:
        sub = df[df["label"] == label]
        got = (
            int((sub["source"] == "plantvillage").sum()),
            int((sub["orig_split"] == "train").sum()),
            int((sub["orig_split"] == "test").sum()),
        )
        flag = "" if got == EXPECTED_COUNTS[label] else f"   <-- expected {EXPECTED_COUNTS[label]}"
        print(f"{label:<24}{got[0]:>7}{got[1]:>10}{got[2]:>9}{flag}")
        if flag:
            mismatches.append(label)
    print(f"{'TOTAL':<24}{(df['source'] == 'plantvillage').sum():>7}"
          f"{(df['orig_split'] == 'train').sum():>10}{(df['orig_split'] == 'test').sum():>9}\n")
    if mismatches:
        sys.exit(f"ERROR: image counts differ from expected for: {mismatches}. "
                 "Check the download before continuing.")


# --------------------------------------------------------------------------- #
# 2. Hashing
# --------------------------------------------------------------------------- #
def hash_one(full_path: str) -> tuple[str, int, int]:
    """Return (phash hex, width, height) for one image."""
    with Image.open(full_path) as img:
        width, height = img.size
        return str(imagehash.phash(img.convert("RGB"))), width, height


def hash_images(df: pd.DataFrame, data_dir: Path, workers: int | None) -> pd.DataFrame:
    """Add phash/width/height columns. Uses all CPU cores because this is the slow step."""
    paths = [str(data_dir / p) for p in df["path"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(pool.map(hash_one, paths, chunksize=64),
                            total=len(paths), desc="phash"))
    df = df.copy()
    df["phash"], df["width"], df["height"] = zip(*results)
    return df


def hash_bits(hashes) -> np.ndarray:
    """Hex phash strings -> (n, 64) array of 0/1 floats, for fast pairwise distances."""
    return np.array([imagehash.hex_to_hash(h).hash.flatten() for h in hashes], dtype=np.float32)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between two sets of 64-bit hashes (as 0/1 arrays).

    Counts positions where a=1,b=0 plus positions where a=0,b=1, using two
    matrix products instead of a slow Python double loop.
    """
    return a @ (1 - b).T + (1 - a) @ b.T


class UnionFind:
    """Minimal union-find, used to turn 'these two images are near-duplicates'
    pairs into clusters (if A~B and B~C then A, B, C share one cluster)."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)


def near_duplicate_pairs(bits: np.ndarray, threshold: int) -> list[tuple[int, int, int]]:
    """All (i, j, distance) with i < j and Hamming distance <= threshold."""
    dist = hamming_matrix(bits, bits)
    i, j = np.where(np.triu(dist <= threshold, k=1))
    return [(int(a), int(b), int(dist[a, b])) for a, b in zip(i, j)]


# --------------------------------------------------------------------------- #
# 3. PlantVillage leaf grouping
# --------------------------------------------------------------------------- #
def load_leaf_map(data_dir: Path, pv_folder: str) -> dict[str, int]:
    """Original file name -> leaf number, or {} if this class has no leaf map."""
    csv = data_dir / "plantvillage" / "leaf_grouping" / "filtered_leafmaps" / f"{pv_folder}.csv"
    if not csv.exists():
        return {}
    leaf_map = pd.read_csv(csv)
    return {name: int(float(leaf)) for name, leaf in zip(leaf_map["File Name"], leaf_map["Leaf #"])}


def assign_pv_groups(df: pd.DataFrame, data_dir: Path, threshold: int) -> pd.DataFrame:
    """Set group_id for PlantVillage rows.

    PlantVillage has several photos of the same physical leaf. If those photos
    were split between train and test, the model would be tested on leaves it
    has already seen. So we group by leaf (from the dataset's leaf maps) and
    later split whole groups. Files with no leaf map entry fall back to phash
    grouping: near-identical images share a group, otherwise one group each.
    """
    df = df.copy()
    print("Leaf-map matches per class:")
    for label, pv_folder, _ in CLASSES:
        idx = df.index[(df["source"] == "plantvillage") & (df["label"] == label)]
        leaf_map = load_leaf_map(data_dir, pv_folder)
        # Files are named '<uuid>___<original name>'; the leaf map uses the original name.
        orig_names = [Path(p).name.split("___", 1)[-1] for p in df.loc[idx, "path"]]
        leaf_ids = [leaf_map.get(n) for n in orig_names]
        matched = [i for i, leaf in zip(idx, leaf_ids) if leaf is not None]
        unmatched = [i for i, leaf in zip(idx, leaf_ids) if leaf is None]
        for i, leaf in zip(idx, leaf_ids):
            if leaf is not None:
                df.at[i, "group_id"] = f"pv_{label}_{leaf}"

        # Fallback: cluster the unmatched images by phash.
        n_hash_groups = 0
        if unmatched:
            uf = UnionFind(len(unmatched))
            for a, b, _ in near_duplicate_pairs(hash_bits(df.loc[unmatched, "phash"]), threshold):
                uf.union(a, b)
            roots = [uf.find(k) for k in range(len(unmatched))]
            for i, root in zip(unmatched, roots):
                df.at[i, "group_id"] = f"pv_{label}_h{root}"
            n_hash_groups = len(set(roots))

        note = "" if leaf_map else "  (no leaf map)"
        print(f"  {label:<24}{len(matched):>5}/{len(idx):<5} matched, "
              f"{len(unmatched)} unmatched -> {n_hash_groups} hash groups{note}")
    return df


# --------------------------------------------------------------------------- #
# 4. Deduplication
# --------------------------------------------------------------------------- #
def dedup_plantdoc(df: pd.DataFrame, threshold: int) -> tuple[pd.DataFrame, list[dict]]:
    """Cluster PlantDoc near-duplicates, keep one per cluster, and remove any
    PlantDoc image that is a near-duplicate of a PlantVillage image.

    Returns the updated frame and a list of flagged pairs for the report.
    """
    df = df.copy()
    pd_idx = df.index[df["source"] == "plantdoc"].to_numpy()
    pv_idx = df.index[df["source"] == "plantvillage"].to_numpy()
    pd_bits = hash_bits(df.loc[pd_idx, "phash"])
    report = []

    # Within PlantDoc: the same web photo is often scraped more than once.
    uf = UnionFind(len(pd_idx))
    for a, b, d in near_duplicate_pairs(pd_bits, threshold):
        uf.union(a, b)
        report.append({"type": "within_plantdoc", "a": pd_idx[a], "b": pd_idx[b], "hamming": d})
    clusters: dict[int, list[int]] = {}
    for k in range(len(pd_idx)):
        clusters.setdefault(uf.find(k), []).append(pd_idx[k])
    for cid, (_, members) in enumerate(sorted(clusters.items())):
        for i in members:
            df.at[i, "group_id"] = f"pd_{cid}"
        # Prefer keeping an official test/ image so that subset stays as intact as possible.
        keep = min(members, key=lambda i: (df.at[i, "orig_split"] != "test", df.at[i, "path"]))
        for i in members:
            if i != keep:
                df.at[i, "is_duplicate"] = True

    # Across datasets: a PlantDoc image that matches a PlantVillage image is not a
    # genuine "field" image for our purposes, so it is removed from the field test set.
    dist = hamming_matrix(pd_bits, hash_bits(df.loc[pv_idx, "phash"]))
    for a in np.where(dist.min(axis=1) <= threshold)[0]:
        b = int(dist[a].argmin())
        df.at[pd_idx[a], "is_duplicate"] = True
        report.append({"type": "cross_dataset", "a": pd_idx[a], "b": pv_idx[b],
                       "hamming": int(dist[a, b])})
    return df, report


def write_dedup_report(df: pd.DataFrame, report: list[dict], data_dir: Path,
                       reports_dir: Path, max_pairs: int = 40) -> None:
    """Save the flagged pairs as CSV, plus a contact sheet for a human spot-check."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = [{
        "type": r["type"],
        "path_a": df.at[r["a"], "path"], "label_a": df.at[r["a"], "label"],
        "path_b": df.at[r["b"], "path"], "label_b": df.at[r["b"], "label"],
        "hamming": r["hamming"],
        "same_label": df.at[r["a"], "label"] == df.at[r["b"], "label"],
    } for r in report]
    out = pd.DataFrame(rows, columns=["type", "path_a", "label_a", "path_b",
                                      "label_b", "hamming", "same_label"])
    out.to_csv(reports_dir / "dedup_report.csv", index=False)
    print(f"Wrote {len(out)} flagged pairs to {reports_dir / 'dedup_report.csv'}")

    if out.empty:
        return
    # Show the riskiest pairs first: the largest distances are the ones most
    # likely to be false positives if the threshold is too loose.
    sample = out.sort_values("hamming", ascending=False).head(max_pairs)
    per_row = 4  # pairs per row, two images per pair
    n_rows = int(np.ceil(len(sample) / per_row))
    fig, axes = plt.subplots(n_rows, per_row * 2, figsize=(per_row * 4, n_rows * 2.3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (_, r) in enumerate(sample.iterrows()):
        for side in ("a", "b"):
            ax = axes[k // per_row, (k % per_row) * 2 + (side == "b")]
            with Image.open(data_dir / r[f"path_{side}"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{r[f'label_{side}']}" + (f"\n{r['type']} d={r['hamming']}" if side == "a" else ""),
                         fontsize=7)
    fig.tight_layout()
    fig.savefig(reports_dir / "dedup_pairs.png", dpi=110)
    plt.close(fig)
    print(f"Wrote contact sheet to {reports_dir / 'dedup_pairs.png'}")


# --------------------------------------------------------------------------- #
# 5. Splits
# --------------------------------------------------------------------------- #
def split_plantvillage(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Group-aware, stratified 80/10/10 split of PlantVillage.

    StratifiedGroupKFold with 10 folds gives 10 disjoint ~10% chunks that keep
    whole leaf groups together and roughly preserve class proportions. We use
    one chunk as test, one as val, and the remaining eight as train.
    """
    df = df.copy()
    pv = df[df["source"] == "plantvillage"]
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = sgkf.split(pv.index, pv["label_idx"], groups=pv["group_id"])
    _, test_pos = next(folds)
    _, val_pos = next(folds)
    df.loc[pv.index, "split"] = "train"
    df.loc[pv.index[val_pos], "split"] = "val"
    df.loc[pv.index[test_pos], "split"] = "test"

    # Safety check: no leaf group may appear in more than one split.
    leaky = pv.assign(split=df.loc[pv.index, "split"]).groupby("group_id")["split"].nunique()
    if (leaky > 1).any():
        sys.exit(f"ERROR: {int((leaky > 1).sum())} groups span multiple splits.")

    # PlantDoc is test-only: every non-duplicate image is part of the field test set.
    is_pd = df["source"] == "plantdoc"
    df.loc[is_pd, "split"] = np.where(df.loc[is_pd, "is_duplicate"], "excluded", "field_test")
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Per-class image counts for every split, plus group counts for PlantVillage."""
    table = pd.crosstab(df["label"], [df["source"], df["split"]]).reindex(LABELS)
    print("\nImages per split:")
    print(table.to_string())
    pv = df[df["source"] == "plantvillage"]
    sizes = pv.groupby("group_id").size()
    print(f"\nPlantVillage: {len(sizes)} leaf groups, median size {int(sizes.median())}, "
          f"largest {int(sizes.max())}")
    pd_test = df[(df["source"] == "plantdoc") & (df["orig_split"] == "test")]
    print(f"PlantDoc official test/ kept after dedup: {int((~pd_test['is_duplicate']).sum())}/{len(pd_test)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--hash-threshold", type=int, default=5,
                        help="max phash Hamming distance to count as a near-duplicate")
    parser.add_argument("--seed", type=int, default=42, help="seed for the PlantVillage split")
    parser.add_argument("--workers", type=int, default=None, help="processes for hashing (default: all cores)")
    args = parser.parse_args()

    df = scan_datasets(args.data_dir)
    check_counts(df)
    df = hash_images(df, args.data_dir, args.workers)
    df["group_id"] = ""
    df["is_duplicate"] = False
    df["split"] = ""

    df = assign_pv_groups(df, args.data_dir, args.hash_threshold)
    df, report = dedup_plantdoc(df, args.hash_threshold)
    write_dedup_report(df, report, args.data_dir, args.reports_dir)
    df = split_plantvillage(df, args.seed)
    print_summary(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[MANIFEST_COLUMNS].to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
