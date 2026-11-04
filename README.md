# Tomato-Plant-Disease-Classification

CS/SE 4AL3 Group 12. Does a CNN trained on lab images (PlantVillage) generalize to
real-world field images (PlantDoc)? See `CLAUDE.md` for the full project design.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download the data

Only the tomato colour images of PlantVillage are fetched (sparse checkout).
`data/` is gitignored; never commit images or checkpoints.

```bash
mkdir -p data/raw && cd data/raw
git clone --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git plantvillage
cd plantvillage
git sparse-checkout set --no-cone '/raw/color/Tomato___*' '/leaf_grouping/filtered_leafmaps/Tomato___*'
cd ..
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git plantdoc
cd ../..
```

## Build the manifest

```bash
python -m src.prepare --data-dir data/raw --out data/manifest.csv
```

This checks image counts, groups PlantVillage photos by leaf, deduplicates
PlantDoc, and writes the fixed train/val/test split to `data/manifest.csv`.
It also writes `reports/dedup_report.csv` and `reports/dedup_pairs.png`
(a contact sheet of flagged near-duplicates to spot-check).

**Do not regenerate the manifest after experiments start** — the split must stay fixed.
