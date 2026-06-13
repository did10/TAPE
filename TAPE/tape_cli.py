
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .model import reproducibility, scaden

PROPORTION_PREFIX = "$proportions_"

ARCHITECTURES = {
    "m256": ([256, 128, 64, 32], [0, 0, 0, 0]),
    "m512": ([512, 256, 128, 64], [0, 0.3, 0.2, 0.1]),
    "m1024": ([1024, 512, 256, 128], [0, 0.6, 0.3, 0.1]),
}


def _load_training_data(h5ad_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("anndata is required to run the TAPE CLI.") from exc

    adata = ad.read_h5ad(h5ad_path)
    expression = adata.to_df()
    expression.index = adata.obs_names

    prop_cols = [c for c in adata.obs.columns if str(c).startswith(PROPORTION_PREFIX)]
    if not prop_cols:
        raise ValueError(
            f"No ground-truth columns found in '{h5ad_path}'. Expected obs columns starting with '{PROPORTION_PREFIX}'."
        )

    props = adata.obs[prop_cols].copy()
    props.columns = [str(c)[len(PROPORTION_PREFIX):] for c in prop_cols]
    props.index = adata.obs_names
    return expression, props


def _load_expression(h5ad_path: Path) -> pd.DataFrame:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("anndata is required to run the TAPE CLI.") from exc

    adata = ad.read_h5ad(h5ad_path)
    expression = adata.to_df()
    expression.index = adata.obs_names
    return expression


def train(train_h5ad: Path, output_dir: Path, batch_size: int, epochs: int, seed: int) -> Path:
    expression, props = _load_training_data(train_h5ad)
    output_dir.mkdir(parents=True, exist_ok=True)

    reproducibility(seed)
    model = scaden(ARCHITECTURES, expression.to_numpy(), props.to_numpy(), batch_size=batch_size, epochs=epochs)
    model.train()
    model.save_model(str(output_dir), expression.columns.tolist(), props.columns.tolist())
    return output_dir


def predict(model_dir: Path, test_h5ad: Path, output_dir: Path) -> Path:
    expression = _load_expression(test_h5ad)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = scaden.from_file(str(model_dir))
    if model.gene_names != expression.columns.tolist():
        raise ValueError("The gene names in the test dataset do not match those used for training the Scaden model.")

    predictions = model.predict(expression.to_numpy())
    pred_df = pd.DataFrame(predictions, columns=model.label_names, index=expression.index)
    out_file = output_dir / "predictions.tsv"
    pred_df.to_csv(out_file, sep="\t", index=True)
    return out_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark-tape", description="Standalone TAPE/Scaden CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a Scaden model from a .h5ad dataset")
    train_parser.add_argument("--train-h5ad", type=Path, required=True, help="Training dataset in .h5ad format")
    train_parser.add_argument("--output-dir", type=Path, required=True, help="Directory where model files are written")
    train_parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=128)
    train_parser.add_argument("--seed", type=int, default=0)

    predict_parser = subparsers.add_parser("predict", help="Run Scaden inference from a saved model")
    predict_parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing architecture.pt and model weights")
    predict_parser.add_argument("--test-h5ad", type=Path, required=True, help="Test dataset in .h5ad format")
    predict_parser.add_argument("--output-dir", type=Path, required=True, help="Directory where predictions.tsv is written")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        train(args.train_h5ad, args.output_dir, args.batch_size, args.epochs, args.seed)
    elif args.command == "predict":
        predict(args.model_dir, args.test_h5ad, args.output_dir)
    else:
        parser.error(f"Unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

