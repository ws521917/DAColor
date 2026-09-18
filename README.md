# DAColor

This private research repository contains only the DAColor implementation and its source color-scheme dataset. Baseline implementations, baseline outputs, old checkpoints, and exploratory tracks are intentionally excluded.

## Final version

The current best version is **DAColor-Hierarchical**:

- q1/q2 combine the corrected DAColor neural score with a validation-selected high-order co-occurrence prior;
- q3 uses hierarchical backoff from individual-color pairs to two-color query subsets and then the exact three-color query;
- all mixing weights are selected using validation MRR inside each fold.

The five-fold overall metrics are Acc@1 0.19706, Acc@5 0.42016, Acc@10 0.53385, and MRR 0.31184. The compact fold records and selected weights are stored in `results/dacolor_final_cv_summary.json`.

## Dataset

- 25,040 color schemes in `data/data.txt`;
- 162 source palette rows in `data/colorbrewer.txt`;
- 151 unique RGB candidates after deduplication;
- a source manifest with checksums in `data/source_manifest.json`.

The code treats the input as 8-bit sRGB, scales it to [0,1], converts it to CIELAB using the D65 2-degree reference white `(0.95047, 1.00000, 1.08883)`, and computes CIEDE2000 with `kL=kC=kH=1`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Prepare data

Create the corrected fixed split:

```bash
python -m dacolor.prepare_data
```

Create the deterministic five folds used by the final experiment:

```bash
python scripts/prepare_cv_folds.py
```

Generated split files are excluded from Git because they can be reproduced from the included raw records.

## Reproduce the final DAColor pipeline

```bash
python scripts/run_best_pipeline.py --device auto
```

The pipeline prepares five folds, trains one corrected mixed-query DAColor model per fold, tunes the high-order prior on validation data, applies DAColor reranking for q1/q2, applies hierarchical backoff for q3, and assembles a compact final result file.

This is a computationally intensive run. Use `--force` to retrain existing fold checkpoints.

## Tests

```bash
pytest -q
```

The tests cover palette provenance and deduplication, CIEDE2000 reference values, loss normalization, false-negative masking, auxiliary-task symmetry, and high-order query scoring.

## Repository layout

- `dacolor/`: model, training, inference, color conversion, data preparation, and priors;
- `scripts/`: final cross-validation pipeline plus DAColor ablation and sensitivity scripts;
- `tests/`: correctness tests;
- `data/`: source dataset and palette;
- `results/dacolor_final_cv_summary.json`: compact final result record.

This repository contains unpublished research code. No reuse license is granted unless a license file is added later.
