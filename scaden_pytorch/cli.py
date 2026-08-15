
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import torch
import pandas as pd

from .model import reproducibility, Scaden

PROPORTION_PREFIX = "$proportions_"


def _parse_nested_int_list(value: str) -> list[list[int]]:
    """Parse something like '[[256,128,64,32],[512,256,128,64]]' into a list of lists of ints."""
    try:
        parsed = ast.literal_eval(value.strip())
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            f"Could not parse architecture. Must be a list of lists of ints, "
            f"e.g. [[256,128,64,32],[512,256,128,64]]. Got: {value!r}"
        ) from exc
    if not isinstance(parsed, list) or not all(isinstance(sublist, list) for sublist in parsed):
        raise ValueError(
            f"Must be a list of lists, e.g. [[256,128,64,32],[512,256,128,64]]. Got: {value!r}"
        )
    result = []
    for sublist in parsed:
        items = [int(x) for x in sublist]
        result.append(items)
    return result


def _parse_nested_float_list(value: str) -> list[list[float]]:
    """Parse something like '[[0,0,0,0],[0,0.3,0.2,0.1]]' into a list of lists of floats."""
    try:
        parsed = ast.literal_eval(value.strip())
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            f"Could not parse dropouts. Must be a list of lists of floats, "
            f"e.g. [[0,0,0,0],[0,0.3,0.2,0.1]]. Got: {value!r}"
        ) from exc
    if not isinstance(parsed, list) or not all(isinstance(sublist, list) for sublist in parsed):
        raise ValueError(
            f"Must be a list of lists, e.g. [[0,0,0,0],[0,0.3,0.2,0.1]]. Got: {value!r}"
        )
    result = []
    for sublist in parsed:
        items = [float(x) for x in sublist]
        result.append(items)
    return result


def _load_training_data(h5ad_path: Path, proportion_prefix: str = PROPORTION_PREFIX) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("anndata is required to run the scaden-pytorch CLI.") from exc

    adata = ad.read_h5ad(h5ad_path)
    expression = adata.to_df()
    expression.index = adata.obs_names

    if expression.columns.duplicated().any():
        dupes = expression.columns[expression.columns.duplicated()].unique().tolist()
        raise ValueError(
            f"Duplicate gene names found in '{h5ad_path}': {dupes}. "
            "Please fix the gene annotation before training."
        )

    prop_cols = [c for c in adata.obs.columns if str(c).startswith(proportion_prefix)]
    if not prop_cols:
        raise ValueError(
            f"No ground-truth columns found in '{h5ad_path}'. Expected obs columns "
            f"starting with '{proportion_prefix}' (change with --proportion-prefix)."
        )

    props = adata.obs[prop_cols].copy()
    props.columns = [str(c)[len(proportion_prefix):] for c in prop_cols]
    props.index = adata.obs_names
    return expression, props


def _load_expression(h5ad_path: Path) -> pd.DataFrame:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("anndata is required to run the scaden-pytorch CLI.") from exc

    adata = ad.read_h5ad(h5ad_path)
    expression = adata.to_df()
    expression.index = adata.obs_names
    return expression


def train(train_h5ad: Path, output_dir: Path, batch_size: int, epochs: int, seed: int, threads: int,
          architectures: list[list[int]], dropouts: list[list[float]], loss_fn: str = "l1",
          weight_decay: float = 0.0, patience: int | None = None,
          val_h5ad: Path | None = None, use_batch_norm: bool = False,
          lr_scheduler: str = "none", noise_std: float = 0.0,
          proportion_prefix: str = PROPORTION_PREFIX) -> Path:
    torch.set_num_threads(threads)
    expression, props = _load_training_data(train_h5ad, proportion_prefix=proportion_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_expr, val_props = None, None
    if val_h5ad is not None and patience is not None:
        val_expr, val_props = _load_training_data(val_h5ad, proportion_prefix=proportion_prefix)
    elif val_h5ad is not None and patience is None:
        print("Warning: --val-h5ad provided without --patience. Validation data will be ignored.")
    elif patience is not None and val_h5ad is None:
        print("Warning: --patience provided without --val-h5ad. Early stopping disabled.")

    reproducibility(seed)
    model = Scaden(architectures, dropouts, expression.to_numpy(), props.to_numpy(),
                   batch_size=batch_size, epochs=epochs, loss_fn=loss_fn, weight_decay=weight_decay,
                   patience=patience, use_batch_norm=use_batch_norm, lr_scheduler=lr_scheduler,
                   noise_std=noise_std,
                   val_x=val_expr.to_numpy() if val_expr is not None else None,
                   val_y=val_props.to_numpy() if val_props is not None else None)
    model.train()
    model.save_model(str(output_dir), expression.columns.tolist(), props.columns.tolist())
    print(f"Model saved to {output_dir} (architecture.pt + model_*.pt)")
    return output_dir


def predict(model_dir: Path, test_h5ad: Path, output_dir: Path, threads: int,
            allow_gene_subset: bool = False) -> Path:
    torch.set_num_threads(threads)
    expression = _load_expression(test_h5ad)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Scaden.from_file(str(model_dir))
    train_genes = model.gene_names
    test_genes = expression.columns.tolist()


    missing = [g for g in train_genes if g not in test_genes]
    extra = [g for g in test_genes if g not in train_genes]

    if missing:
        raise ValueError(
            f"The test dataset is missing {len(missing)} of the {len(train_genes)} genes "
            f"the model was trained on (e.g. {missing[:5]}). The model's input size is "
            "fixed at training time, so missing genes cannot be filled in. Fix the test "
            "file or retrain the model with matching genes."
        )

    if extra:
        if not allow_gene_subset:
            raise ValueError(
                f"The test dataset contains {len(extra)} genes that were not used in "
                f"training (e.g. {extra[:5]}). Expected the {len(train_genes)} training "
                "genes. If this is intended, rerun with --allow-gene-subset to ignore "
                "the extra genes."
            )
        print(
            f"Warning: ignoring {len(extra)} genes present in the test data but not in "
            f"training (e.g. {extra[:5]})."
        )


    expression = expression[train_genes]
    predictions = model.predict(expression.to_numpy())
    pred_df = pd.DataFrame(predictions, columns=model.label_names, index=expression.index)
    out_file = output_dir / "predictions.tsv"
    pred_df.to_csv(out_file, sep="\t", index=True)
    print(f"Predictions written to {out_file}")
    return out_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scaden-pytorch",
        description="Deconvolve bulk RNA-seq into cell-type fractions using the PyTorch Scaden implementation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a Scaden model from a .h5ad dataset")
    train_parser.add_argument("--train-h5ad", type=Path, required=True, help="Training dataset in .h5ad format")
    train_parser.add_argument("--output-dir", type=Path, required=True, help="Directory where model files are written")
    train_parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=128)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--threads", type=int, help="Threads used by torch", default=4)
    train_parser.add_argument("--architecture", type=str,
                              help="Nested list of hidden layer sizes per model, e.g. [[256,128,64,32],[512,256,128,64]]", default="[[256, 128, 64, 32],[512, 256, 128, 64],[1024, 512, 256, 128]]")
    train_parser.add_argument("--dropout", type=str,
                              help="Nested list of dropout rates per model, e.g. [[0,0,0,0],[0,0.3,0.2,0.1]]", default="[[0, 0, 0, 0],[0, 0.3, 0.2, 0.1],[0, 0.6, 0.3, 0.1]]")
    train_parser.add_argument("--loss-fn", "--loss_fn", dest="loss_fn", type=str, default="l1",
                              choices=["l1", "mse", "ccc", "combined", "cross_entropy"],
                              help="Loss function to use during training (default: l1)")
    train_parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.0,
                              help="Weight decay (L2 regularization) for Adam optimizer (default: 0.0)")
    train_parser.add_argument("--patience", type=int, default=None,
                              help="Early stopping patience (epochs with no improvement before stopping). Requires --val-h5ad (default: None)")
    train_parser.add_argument("--val-h5ad", "--val_h5ad", dest="val_h5ad", type=Path, default=None,
                              help="Validation dataset in .h5ad format for early stopping (default: None)")
    train_parser.add_argument("--use-batch-norm", "--use_batch_norm", dest="use_batch_norm", action="store_true",
                              help="Add BatchNorm1d after each Linear layer (default: False)")
    train_parser.add_argument("--lr-scheduler", "--lr_scheduler", dest="lr_scheduler", type=str, default="none",
                              choices=["none", "plateau", "cosine"],
                              help="Learning rate scheduler: none, plateau (ReduceLROnPlateau), or cosine (CosineAnnealingLR) (default: none)")
    train_parser.add_argument("--noise-std", "--noise_std", dest="noise_std", type=float, default=0.0,
                              help="Gaussian noise std-dev added to training inputs (default: 0.0)")
    train_parser.add_argument("--proportion-prefix", "--proportion_prefix", dest="proportion_prefix", type=str,
                              default=PROPORTION_PREFIX,
                              help=f"Prefix of the ground-truth proportion columns in the h5ad obs table (default: '{PROPORTION_PREFIX}')")

    predict_parser = subparsers.add_parser("predict", help="Run Scaden inference from a saved model")
    predict_parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing architecture.pt and model weights")
    predict_parser.add_argument("--test-h5ad", type=Path, required=True, help="Test dataset in .h5ad format")
    predict_parser.add_argument("--output-dir", type=Path, required=True, help="Directory where predictions.tsv is written")
    predict_parser.add_argument("--threads", type=int, help="Threads used by torch", default=4)
    predict_parser.add_argument("--allow-gene-subset", "--allow_gene_subset", dest="allow_gene_subset", action="store_true",
                                help="Ignore genes present in the test data but not in training instead of failing (default: False)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        architectures = _parse_nested_int_list(args.architecture)
        dropouts = _parse_nested_float_list(args.dropout)
        if len(architectures) != len(dropouts):
            parser.error("--architecture and --dropout must have the same number of models")
        for i, (arch, do) in enumerate(zip(architectures, dropouts)):
            if len(arch) != len(do):
                parser.error(f"Model {i}: --architecture ({arch}) and --dropout ({do}) must have the same number of layers")
        train(args.train_h5ad, args.output_dir, args.batch_size, args.epochs, args.seed, args.threads,
              architectures, dropouts, loss_fn=args.loss_fn, weight_decay=args.weight_decay,
              patience=args.patience, val_h5ad=args.val_h5ad,
              use_batch_norm=args.use_batch_norm, lr_scheduler=args.lr_scheduler,
              noise_std=args.noise_std, proportion_prefix=args.proportion_prefix)
    elif args.command == "predict":
        predict(args.model_dir, args.test_h5ad, args.output_dir, args.threads,
                allow_gene_subset=args.allow_gene_subset)
    else:
        parser.error(f"Unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

