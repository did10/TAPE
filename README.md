# scaden-pytorch

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](#installation)

A clean **PyTorch reimplementation of [Scaden](https://github.com/KevinMenden/scaden)** — a deep-learning
method for **deconvolving bulk RNA-seq samples into cell-type fractions** using a
single-cell reference.

The original Scaden is implemented in TensorFlow and was published as part of the
*TAPE* paper. This repository removes all TAPE-specific code and ships an
improved, standalone PyTorch version of Scaden with a small CLI, additional
training options, and fail-fast input validation.

---

## Features

- **Ensemble deconvolution** — predictions are averaged over an ensemble of
  independently trained MLPs (softmax outputs = cell-type fractions).
- **Multiple loss functions** — `l1`, `mse`, `ccc` (concordance correlation
  coefficient), `combined` (MSE + CCC), `cross_entropy`.
- **Early stopping** on a validation dataset with patience.
- **Batch normalization**, **learning-rate schedulers** (`plateau`, `cosine`),
  **weight decay**, and **input noise** for regularization.
- **Fail-fast input validation** — gene mismatches between training and test
  data raise an error by default, so you never silently lose genes.
- **Reproducible training** via a fixed random seed.
- **Python API** and a **command-line interface** (`scaden-pytorch`).

---

## Installation

Requires Python 3.9+ and PyTorch (see [pytorch.org](https://pytorch.org) for
the version matching your compute platform — CUDA, ROCm, or CPU).

From GitHub:

```bash
pip install git+https://github.com/did10/TAPE.git
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/did10/TAPE.git
cd TAPE
pip install -e .
```

This installs the `scaden-pytorch` command-line tool and the `scaden_pytorch`
Python package.

---

## Data format

Both `train` and `predict` use `.h5ad` files (AnnData).

**Training data** must contain:

- a gene-expression matrix (samples × genes), and
- ground-truth cell-type proportions in the `obs` table, with column names
  prefixed by `$proportions_`, e.g. `$proportions_B cells`,
  `$proportions_T cells`, ...

The prefix is configurable with `--proportion-prefix`.

**Test data** only needs the gene-expression matrix (samples × genes).

> [!IMPORTANT]
> The gene names in the test dataset must match the genes the model was
> trained on. By default the CLI **fails** if there is any mismatch:
>
> - **Missing genes** (in training, not in test) are always an error — the
>   model's input size is fixed at training time, so they cannot be filled in.
> - **Extra genes** (in test, not in training) are an error by default; if you
>   know what you are doing, pass `--allow-gene-subset` to ignore them (a
>   warning is printed).
>
> Column *order* does not matter — the test columns are reordered to match the
> training genes automatically.

---

## Command-line usage

### Train a model

```bash
scaden-pytorch train \
  --train-h5ad training_simulations.h5ad \
  --output-dir models/my_scaden \
  --epochs 128 \
  --batch-size 128
```

The output directory will contain `architecture.pt` (model metadata, gene and
cell-type names) plus one `model_<i>.pt` weight file per ensemble member.

Key training options (see `scaden-pytorch train --help` for all):

| Option | Default | Description |
| --- | --- | --- |
| `--architecture` | 3 models, e.g. `[[256,128,64,32],[512,256,128,64],[1024,512,256,128]]` | Nested list of hidden layer sizes per ensemble member |
| `--dropout` | matching dropout rates | Nested list of dropout rates per layer per model |
| `--loss-fn` | `l1` | `l1`, `mse`, `ccc`, `combined`, `cross_entropy` |
| `--epochs` | `128` | Training epochs per model |
| `--batch-size` | `128` | Training batch size |
| `--seed` | `0` | Random seed for reproducibility |
| `--patience` / `--val-h5ad` | — | Early stopping patience and validation data |
| `--use-batch-norm` | off | Add BatchNorm1d after each Linear layer |
| `--lr-scheduler` | `none` | `none`, `plateau`, or `cosine` |
| `--weight-decay` | `0.0` | L2 weight decay for Adam |
| `--noise-std` | `0.0` | Std-dev of Gaussian input noise during training |
| `--proportion-prefix` | `$proportions_` | Prefix of ground-truth proportion columns |

### Predict cell-type fractions

```bash
scaden-pytorch predict \
  --model-dir models/my_scaden \
  --test-h5ad bulk_samples.h5ad \
  --output-dir predictions
```

Writes `predictions.tsv` (samples × cell types, fractions summing to 1) into the
output directory.

---

## Python API

```python
import numpy as np
from scaden_pytorch import Scaden, reproducibility

# train_x: (samples, genes), train_y: (samples, cell types)
reproducibility(seed=0)
model = Scaden(
    architectures=[[256, 128, 64, 32], [512, 256, 128, 64]],
    dropouts=[[0, 0.3, 0.2, 0.1], [0, 0.6, 0.3, 0.1]],
    train_x=train_x,
    train_y=train_y,
    epochs=128,
    batch_size=128,
    loss_fn="l1",
)
model.train()
model.save_model("models/my_scaden", gene_names=genes, label_names=cell_types)

# later / in another process:
model = Scaden.from_file("models/my_scaden")
fractions = model.predict(test_x)  # (samples, cell types), rows sum to 1
```

---

## Differences from the original Scaden

- Written in **PyTorch** instead of TensorFlow (the original implementation was
  not easy to run/test, which is why the TAPE authors wrote a PyTorch port).
- Added configurable loss functions (including CCC and a combined loss),
  early stopping, batch normalization, LR schedulers, weight decay, and input
  noise.
- Added a simple CLI and strict gene-name validation between training and test
  data.

---

## Citation

If you use this software, please cite the **Scaden** paper (the method) and,
if relevant, the **TAPE** paper (which popularized the PyTorch port this
repository is derived from):

```bibtex
@article{Menden2020,
   author = {Menden, Kevin and Marouf, Mohamed and Oller, Sergio and Dalmia, Ananya and Magruder, Daniel S. and Klobler, Stefan and Heutink, Peter and Bonn, Stefan},
   title = {Deep learning-based cell composition analysis from tissue expression profiles},
   journal = {Science Advances},
   volume = {6},
   number = {30},
   pages = {eaba2619},
   year = {2020},
   doi = {10.1126/sciadv.aba2619}
}
```

```bibtex
@article{Chen2022,
   author = {Chen, Yanshuo and Wang, Yixuan and Chen, Yuelong and Cheng, Yuqi and Wei, Yumeng and Li, Yunxiang and Wang, Jiuming and Wei, Yingying and Chan, Ting-Fung and Li, Yu},
   title = {Deep autoencoder for interpretable tissue-adaptive deconvolution and cell-type-specific gene analysis},
   journal = {Nature Communications},
   volume = {13},
   number = {1},
   pages = {6735},
   year = {2022},
   doi = {10.1038/s41467-022-34550-9}
}
```

---

## License

This project is a fork of [TAPE](https://github.com/poseidonchan/TAPE) and is
distributed under the [GPL-3.0](LICENSE) license.