# Tomato-Plant-Disease-Classification

CS/SE 4AL3 Group 12. Does a CNN trained on lab images (PlantVillage) generalize to
real-world field images (PlantDoc)?

All commands below are run from the project's root folder (the one containing this README).

## 1. Set up the environment (conda)

Requires [conda](https://docs.conda.io/) (Miniconda or Anaconda) and `git`.

```bash
conda create -n tomato python=3.11 -y
conda activate tomato
pip install -r requirements.txt
```

Run `conda activate tomato` again every time you open a new terminal.

Check that PyTorch can see your GPU (training also works on CPU, just much more slowly):

```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

If this prints `False` on a machine with an NVIDIA GPU, reinstall PyTorch using the
command for your system from https://pytorch.org/get-started/locally/ (choose *Pip*),
inside the activated `tomato` environment.

To use the environment in Jupyter or VS Code notebooks, register it once:

```bash
python -m ipykernel install --user --name tomato --display-name "Python (tomato)"
```

**Using WSL?** Keep the project inside the Linux file system (e.g. under `~`), not under
`/mnt/c/...`. Reading thousands of images from the Windows drive is many times slower.

## 2. Download the data

Only the tomato images are downloaded (sparse checkout), about 1 GB in total.
`data/` is gitignored; never commit images or checkpoints.

```bash
mkdir -p data/raw
cd data/raw

git clone --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git plantvillage
cd plantvillage
git sparse-checkout set --no-cone '/raw/color/Tomato___*' '/leaf_grouping/filtered_leafmaps/Tomato___*'
cd ..

git clone --filter=blob:none --sparse https://github.com/pratikkayal/PlantDoc-Dataset.git plantdoc
cd plantdoc
git sparse-checkout set --no-cone '/train/Tomato*' '/test/Tomato*'
cd ../../..
```

Dataset sources: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)
(Hughes & Salathé, 2015) and [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset)
(Singh et al., 2020).

## 3. Build the manifest (once)

```bash
python -m src.prepare --data-dir data/raw --out data/manifest.csv
```

This checks image counts, groups PlantVillage photos by leaf, deduplicates
PlantDoc, and writes the fixed train/val/test split to `data/manifest.csv`.
It also writes `reports/dedup_report.csv` and `reports/dedup_pairs.png`
(a contact sheet of flagged near-duplicates to spot-check).

Expected split: PlantVillage 12,064 train / 1,508 val / 1,508 test; PlantDoc 710 field test images.

**Do not regenerate the manifest after experiments start** — the split must stay fixed.

## 4. Train and evaluate

```bash
python -m src.train --config configs/E1.yaml --overfit    # sanity check: should reach ~100% on 32 images
python -m src.train --config configs/E1.yaml --debug      # tiny subset, 1 epoch: checks everything runs

python -m src.train --config configs/baseline_majority.yaml   # majority-vote baseline (about 1 minute)
python -m src.train --config configs/E1.yaml --seed 0     # simple CNN, original images
python -m src.train --config configs/E2.yaml --seed 0     # simple CNN, with data augmentation
python -m src.train --evaluate outputs/runs/<run_name>    # re-evaluate a saved run without retraining
```

Omit `--seed` to run every seed listed in the config. If training crashes with an
out-of-memory error, lower `batch_size` in the config; if data loading hangs, set `num_workers: 0`.

Training uses PlantVillage only; the best epoch is picked on PlantVillage val.
Each run is then evaluated on PlantVillage test and on the PlantDoc field images.
Results go to `outputs/results.csv` (one row per run and eval set) and
`outputs/runs/<run_name>/` (config, training history, metrics, confusion matrices).

## 5. View results

Open `results.ipynb` (kernel: *Python (tomato)*) and *Run All*. It only reads `outputs/`
(no training), and shows the dataset split, a results table, the lab-vs-field drop, charts,
per-class recall, confusion matrices and model cost. Charts are also saved to
`reports/figures/` for the report.
