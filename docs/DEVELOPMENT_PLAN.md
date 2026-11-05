# Development Plan: Tomato Disease Classification (Lab → Field)

**Version:** 0.1
**Last updated:** 2026-11-05
**Course:** CS/SE 4AL3, McMaster University, Fall 2026, Group 12
**Team:** Hashim Butt, Alex Liu, Zuning Wang

This is the team's working plan: what we are building, how, and what is out of scope.
When the plan changes, update the relevant section and bump the version (see [Versioning](#versioning)).

---

## 1. Project summary

Single-label, multi-class image classification of tomato leaf diseases.

**Research question:** Does a CNN trained on lab-controlled PlantVillage images generalize to real-world PlantDoc field images, and do data augmentation and a stronger (pretrained) architecture help?

**Design (from the proposal, agreed with TA Abdelaziz Serour):**
1. Train on original PlantVillage images.
2. Train again with data augmentation and compare.
3. Compare a simple CNN against ResNet18 and EfficientNet-B0 (performance and computational cost).
4. Evaluate the models on **unseen PlantDoc field images**.

PlantDoc is **test-only**. Never train on it, and never use it for model selection, early stopping, or hyperparameter tuning.

Proposal success target: 80–85% accuracy (on PlantVillage). PlantDoc scores are expected to be much lower; measuring that drop is the point.

## 2. Scope

**Progress report (due Nov 10):** focus on PlantVillage.
- Majority-vote baseline
- Simple CNN on original PlantVillage images
- Simple CNN with standard augmentation
- PlantDoc numbers are computed automatically, but the report can treat PlantDoc evaluation as "next steps"

**Final project:**
- ResNet18 and EfficientNet-B0, each with and without augmentation
- Small PlantVillage cap comparison (100, 250, uncapped) with the best model
- All models evaluated on PlantDoc; a few plots for the write-up

**Out of scope:** Grad-CAM, the `strong` augmentation preset (exists in code, unused), segmented PlantVillage images, fine-tuning on PlantDoc, FLOPs counting, calibration, large cap sweeps. Running 3 seeds per configuration is optional (only if time allows); 1 seed is the default.

## 3. Classes (8)

The proposal says 9 classes, but its image total (15,824) is the 8-class count. Spider mites is excluded (PlantDoc has only 2 images). Target spot is excluded (no PlantDoc images). Reports should say 8 classes and explain why.

| Canonical label | PlantVillage folder (`raw/color/`) | PlantDoc folder (`train/`, `test/`) |
|---|---|---|
| `bacterial_spot` | `Tomato___Bacterial_spot` | `Tomato leaf bacterial spot` |
| `early_blight` | `Tomato___Early_blight` | `Tomato Early blight leaf` |
| `late_blight` | `Tomato___Late_blight` | `Tomato leaf late blight` |
| `leaf_mold` | `Tomato___Leaf_Mold` | `Tomato mold leaf` |
| `septoria_leaf_spot` | `Tomato___Septoria_leaf_spot` | `Tomato Septoria leaf spot` |
| `mosaic_virus` | `Tomato___Tomato_mosaic_virus` | `Tomato leaf mosaic virus` |
| `yellow_leaf_curl_virus` | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | `Tomato leaf yellow virus` |
| `healthy` | `Tomato___healthy` | `Tomato leaf` |

The mapping lives in `CLASSES` in `src/prepare.py`; import it everywhere. Label index = row order above. Do not reorder. Folder names contain spaces; always quote paths.

## 4. Data

- **PlantVillage:** https://github.com/spMohanty/PlantVillage-Dataset (Hughes & Salathé 2015, arXiv:1511.08060). Lab photos of single leaves on plain backgrounds, labeled by plant pathologists. Use `raw/color/` only.
- **PlantDoc:** https://github.com/pratikkayal/PlantDoc-Dataset (Singh et al. 2020, arXiv:1911.10317). Field photos scraped from the web; some label noise; image sizes vary widely.

Download commands are in the README (sparse checkout of tomato images only). `data/` is gitignored; the datasets are far over 50 MB, so the submission links to them. Never commit images or checkpoints.

Counts (before dedup): PlantVillage 15,080; PlantDoc 675 train + 69 test = 744. `prepare.py` checks these and fails loudly if they differ.

### Manifest (`data/manifest.csv`, built once by `python -m src.prepare`)
Columns: `path, source, label, label_idx, orig_split, group_id, phash, is_duplicate, split, width, height`.

- **Leaf grouping:** PlantVillage has several photos of the same leaf (typically 4, from different angles). The dataset's leaf maps (`leaf_grouping/filtered_leafmaps/`) give leaf IDs for about 9,700 images; the rest fall back to perceptual-hash grouping (mostly one image per group). Limitation worth stating in reports: images without a leaf map could still have a sibling photo in another split.
- **Dedup (phash, Hamming ≤ 5):** PlantDoc near-duplicates keep one copy; if copies have different labels, all are dropped (label noise). PlantDoc images matching a PlantVillage image would be dropped (none found). Report: `reports/dedup_report.csv`, `reports/dedup_pairs.png`.
- **Result:** PlantVillage 12,064 train / 1,508 val / 1,508 test; PlantDoc field test 710 images (official `test/` subset: 68 of 69).

## 5. Splits

- **PlantVillage:** group-aware stratified 80/10/10 (`StratifiedGroupKFold`, seed 42). No leaf in more than one split.
- **PlantDoc:** all non-duplicate images = field test set. Also report the official `test/` subset for comparability.
- **Model selection** (early stopping, epochs, hyperparameters) uses PlantVillage val only.
- **Never regenerate the manifest** after experiments start. If the split must change, rerun everything. When working on another machine, copy `data/manifest.csv` rather than rebuilding it, so the split is guaranteed identical.

## 6. Capping PlantVillage

Per-class cap on the PlantVillage **training** split only (val/test fixed), sampling whole leaf groups. Default **250 per class**. Final cap comparison: 100, 250, None (uncapped). When uncapped, use class-weighted cross-entropy (yellow leaf curl is ~35% of PlantVillage).

## 7. Preprocessing and augmentation

RGB, ImageNet mean/std normalization. Images on disk are never modified; all transforms run on the fly when an image is loaded.
- **Eval / `aug: none`:** resize shorter side to 256, center crop 224.
- **`aug: standard`:** RandomResizedCrop(224, scale 0.6–1.0), horizontal + vertical flip, rotation ±30°, ColorJitter(0.3, 0.3, 0.3, 0.05). Training images only.

## 8. Models

| Name | Details |
|---|---|
| `majority` | Majority-vote baseline: always predicts the most common training class (fit on uncapped train) |
| `simple_cnn` | From scratch: 4 conv blocks (conv3×3 → BN → ReLU → maxpool; 32, 64, 128, 256 channels), global average pool, dropout 0.3, linear to 8 |
| `resnet18` | torchvision, `ResNet18_Weights.IMAGENET1K_V1`, replace `fc` (final project) |
| `efficientnet_b0` | torchvision, `EfficientNet_B0_Weights.IMAGENET1K_V1`, replace classifier (final project) |

Pretrained models: train the new head for 2–3 epochs with the backbone frozen, then unfreeze with a lower backbone LR. Log parameter count and inference time per image (computational cost comparison from the proposal).

## 9. Training

- Loss: cross-entropy. Optimizer: AdamW, weight decay 1e-4. LR 1e-3 (head / simple CNN), 1e-4 backbone after unfreezing; cosine schedule
- Batch size 32, up to 20 epochs, early stopping on PlantVillage val macro-F1 (patience 4)
- Mixed precision on CUDA; must also run on CPU/MPS. Device detected automatically (local, WSL or Colab)
- `--overfit` (fit 32 images, sanity check) and `--debug` (tiny subset, 1 epoch) flags
- Checkpoints: `outputs/checkpoints/<run_name>.pt` (gitignored)

## 10. Evaluation

Every run is evaluated on PlantVillage test, PlantDoc field (all), and the PlantDoc official test subset.
- Accuracy, **macro-F1** (primary, since classes are imbalanced), weighted-F1, top-2 accuracy
- Per-class precision/recall/F1 (`classification_report`), confusion matrix (PNG + CSV)
- 95% bootstrap CI for macro-F1 (PlantDoc sets are small)
- Domain gap = PlantVillage-test macro-F1 − PlantDoc-field macro-F1 (the accuracy drop is shown in `results.ipynb`)
- One row per (run, eval set) appended to `outputs/results.csv`; per-run files in `outputs/runs/<run_name>/`
- `results.ipynb` collects everything into tables and charts, and saves figures to `reports/figures/`

## 11. Experiments

Run name: `<experiment>_<model>_aug-<aug>_cap-<cap>_s<seed>`. One YAML per experiment in `configs/`.

| ID | Stage | Model | Aug | Cap | Config |
|---|---|---|---|---|---|
| baseline | Progress report | majority | none | None | `configs/baseline_majority.yaml` |
| E1 | Progress report | simple_cnn | none | 250 | `configs/E1.yaml` |
| E2 | Progress report | simple_cnn | standard | 250 | `configs/E2.yaml` |
| E3 | Final | resnet18, efficientnet_b0 | none, standard | 250 | to do |
| E4 | Final | best from E3 | best from E3 | 100, 250, None | to do |

Plots (final): macro-F1 bar chart per configuration (PlantVillage vs PlantDoc side by side), cap curve, per-class recall.

## 12. Status

| Item | State |
|---|---|
| Data download, manifest, dedup, split | Done |
| Majority baseline | Done |
| E1 (seed 0) | Done |
| E2 | Config ready; results to be added |
| Results notebook | Done |
| E3, E4 | Not started (final project) |

Results so far (seed 0):

| Run | PlantVillage test acc / macro-F1 | PlantDoc field acc / macro-F1 | Domain gap (macro-F1) |
|---|---|---|---|
| Majority baseline | 0.355 / 0.065 | 0.101 / 0.023 | 0.042 |
| E1 simple CNN, no aug | 0.946 / 0.938 | 0.213 / 0.144 | 0.793 |

E1 observations: the model still improved at epoch 20 (early stopping never triggered); on PlantDoc it predicts late blight for almost half of all images and never recognizes healthy leaves (0% recall), consistent with learning PlantVillage's uniform backgrounds rather than disease symptoms. Training took about 30 minutes on an Apple M-series GPU.

## 13. Repository structure and commands

```
.
├── README.md         # setup (conda), data download, commands
├── requirements.txt
├── results.ipynb     # results tables and charts (reads outputs/, no training)
├── configs/          # one YAML per experiment
├── data/             # gitignored: raw/, manifest.csv
├── outputs/          # gitignored: checkpoints/, runs/, results.csv
├── reports/          # dedup report, figures/ for the write-up
├── docs/             # this plan, LaTeX reports (Progress Report/main.tex)
└── src/
    ├── prepare.py    # classes, manifest, counts check, leaf groups, phash, dedup, splits
    └── train.py      # dataset, transforms, models, training, evaluation (labelled sections)
```

Keep the code to these two Python files: the deliverable is a zip of the code, and the team wants it uncluttered.

```bash
python -m src.prepare --data-dir data/raw --out data/manifest.csv
python -m src.train --config configs/E1.yaml [--seed 0] [--debug] [--overfit]   # trains, then evaluates
python -m src.train --evaluate outputs/runs/<run_name>
```

## 14. Conventions

- Paths and hyperparameters come from configs or CLI args; no hardcoded absolute paths.
- Seed `random`, `numpy`, `torch`; keep dataloaders deterministic where practical.
- Never touch PlantDoc during training or model selection.
- Labels come only from dataset folder names; never generate or modify labels with a model.
- The team is new to ML: keep functions small and add short comments explaining *why*.
- Changes are reviewed before they are committed.

## 15. Progress report checklist

Template: `docs/Progress Report/main.tex`, about 4 pages. Due Nov 10.

1. (15%) Dataset properties and preprocessing: sections 3–6 (counts, leaf grouping, dedup, split, cap)
2. (10%) Introduction and previous work, at least 5 references (section 16)
3. (10%) Model inputs and augmentation: section 7
4. (20%) Model implementation, majority baseline, loss, optimizer: sections 8–9
5. (20%) Evaluation strategy, splits, metrics: sections 5 and 10
6. (20%) TA feedback and next steps: section 1 design, section 2 final-project scope
7. (5%) Team contributions

Submission: zip of report + code, with links to the datasets (over 50 MB).

## 16. References

1. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060.
2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 7, 1419.
3. Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. CoDS-COMAD 2020. DOI 10.1145/3371158.3371196.
4. Smart Detection of Tomato Leaf Diseases Using Transfer Learning-Based CNNs. Agriculture, 2023. https://www.mdpi.com/2077-0472/13/1/139
5. XSE-TomatoNet: Explainable lightweight network for tomato leaf disease classification. 2025. https://www.sciencedirect.com/science/article/pii/S221501612500007X

---

## Versioning

The version is `MAJOR.MINOR` and lives in the **Version** line at the top of this file.

- **Bump MINOR** (0.1 → 0.2) whenever the plan changes: scope, design, experiments, or a status update worth recording.
- **Bump MAJOR** (0.x → 1.0) for the plan as submitted with the final project.
- On every bump, update **Version** and **Last updated** at the top and add a row to the changelog below.

### Changelog

| Version | Date | Changes |
|---|---|---|
| 0.1 | 2026-11-05 | First version: scope reduced to the proposal and TA design; data prep, majority baseline, E1 and results notebook done. |
