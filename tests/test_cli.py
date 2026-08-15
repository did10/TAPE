"""Smoke tests for the scaden-pytorch CLI.

These tests run a tiny train/predict roundtrip on synthetic .h5ad data and
verify the fail-fast gene-matching behavior.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scaden_pytorch import cli

GENES = [f"GENE_{i:02d}" for i in range(20)]
CELL_TYPES = ["B cells", "T cells", "Monocytes"]
N_SAMPLES = 24


def make_h5ad(path, genes=GENES, cell_types=CELL_TYPES, n_samples=N_SAMPLES,
              prop_prefix="$proportions_"):
    """Write a synthetic train/test .h5ad file and return its path."""
    genes = list(genes)
    rng = np.random.default_rng(0)
    expr = rng.lognormal(0, 2, size=(n_samples, len(genes)))
    expr = pd.DataFrame(expr, columns=genes, index=[f"sample_{i:02d}" for i in range(n_samples)])

    props = rng.dirichlet(np.ones(len(cell_types)), size=n_samples)
    obs = pd.DataFrame(
        {f"{prop_prefix}{ct}": props[:, i] for i, ct in enumerate(cell_types)},
        index=expr.index,
    )
    adata = AnnData(expr, obs=obs)
    adata.write_h5ad(path)
    return path


def _tiny_train_args(**overrides):
    args = dict(batch_size=8, epochs=2, seed=0, threads=2,
                architectures=[[16, 8]], dropouts=[[0.1, 0.1]])
    args.update(overrides)
    return args


def test_train_predict_roundtrip(tmp_path):
    train_h5ad = make_h5ad(tmp_path / "train.h5ad")
    test_h5ad = make_h5ad(tmp_path / "test.h5ad", n_samples=6)

    model_dir = cli.train(train_h5ad, tmp_path / "model", **_tiny_train_args())
    assert (model_dir / "architecture.pt").exists()
    assert (model_dir / "model_0.pt").exists()

    out = cli.predict(model_dir, test_h5ad, tmp_path / "pred", threads=2)
    assert out.exists()

    pred = pd.read_csv(out, sep="\t", index_col=0)
    assert list(pred.columns) == CELL_TYPES
    assert len(pred) == 6
    np.testing.assert_allclose(pred.sum(axis=1), 1.0, atol=1e-5)


def test_missing_proportion_columns_raises(tmp_path):
    train_h5ad = tmp_path / "train.h5ad"
    rng = np.random.default_rng(0)
    expr = pd.DataFrame(rng.lognormal(0, 1, size=(10, 5)), columns=GENES[:5])
    adata = AnnData(expr, obs=pd.DataFrame(index=expr.index))
    adata.write_h5ad(train_h5ad)

    with pytest.raises(ValueError, match="No ground-truth columns"):
        cli.train(train_h5ad, tmp_path / "model",
                  batch_size=8, epochs=1, seed=0, threads=2,
                  architectures=[[8]], dropouts=[[0.1]])


def test_duplicate_gene_names_raise(tmp_path):
    train_h5ad = tmp_path / "train.h5ad"
    rng = np.random.default_rng(0)
    cols = GENES[:5] + [GENES[0]]  # duplicate
    expr = pd.DataFrame(rng.lognormal(0, 1, size=(10, len(cols))), columns=cols)
    obs = pd.DataFrame(
        {f"$proportions_{ct}": rng.random(10) for ct in CELL_TYPES}, index=expr.index
    )
    AnnData(expr, obs=obs).write_h5ad(train_h5ad)

    with pytest.raises(ValueError, match="Duplicate gene names"):
        cli.train(train_h5ad, tmp_path / "model",
                  batch_size=8, epochs=1, seed=0, threads=2,
                  architectures=[[8]], dropouts=[[0.1]])


def test_custom_proportion_prefix(tmp_path):
    train_h5ad = make_h5ad(tmp_path / "train.h5ad", prop_prefix="gt_")
    model_dir = cli.train(train_h5ad, tmp_path / "model",
                          **_tiny_train_args(proportion_prefix="gt_"))
    assert (model_dir / "architecture.pt").exists()


def test_reordered_genes_predicts(tmp_path):
    train_h5ad = make_h5ad(tmp_path / "train.h5ad")
    test_h5ad = make_h5ad(tmp_path / "test.h5ad", genes=list(reversed(GENES)), n_samples=4)

    model_dir = cli.train(train_h5ad, tmp_path / "model", **_tiny_train_args())
    out = cli.predict(model_dir, test_h5ad, tmp_path / "pred", threads=2)
    assert out.exists()
    pred = pd.read_csv(out, sep="\t", index_col=0)
    np.testing.assert_allclose(pred.sum(axis=1), 1.0, atol=1e-5)


def test_extra_genes_fail_by_default_and_flag_works(tmp_path):
    train_h5ad = make_h5ad(tmp_path / "train.h5ad")
    test_h5ad = make_h5ad(tmp_path / "test.h5ad", genes=GENES + ["EXTRA_1", "EXTRA_2"], n_samples=4)

    model_dir = cli.train(train_h5ad, tmp_path / "model", **_tiny_train_args())

    with pytest.raises(ValueError, match="--allow-gene-subset"):
        cli.predict(model_dir, test_h5ad, tmp_path / "pred", threads=2)

    out = cli.predict(model_dir, test_h5ad, tmp_path / "pred2", threads=2,
                      allow_gene_subset=True)
    assert out.exists()
    pred = pd.read_csv(out, sep="\t", index_col=0)
    np.testing.assert_allclose(pred.sum(axis=1), 1.0, atol=1e-5)


def test_missing_genes_always_fail(tmp_path):
    train_h5ad = make_h5ad(tmp_path / "train.h5ad")
    test_h5ad = make_h5ad(tmp_path / "test.h5ad", genes=GENES[:15], n_samples=4)

    model_dir = cli.train(train_h5ad, tmp_path / "model", **_tiny_train_args())

    with pytest.raises(ValueError, match="missing"):
        cli.predict(model_dir, test_h5ad, tmp_path / "pred", threads=2)
    with pytest.raises(ValueError, match="missing"):
        cli.predict(model_dir, test_h5ad, tmp_path / "pred2", threads=2,
                    allow_gene_subset=True)
