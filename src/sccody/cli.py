"""Training, known-age generation and unknown-age trajectory localization."""
import argparse
import logging
from pathlib import Path
import sys
from .config import ScCodyConfig


def numbers(value):
    try:
        return [float(v) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated ages") from exc


def parser():
    root = argparse.ArgumentParser(prog="sccody", description=__doc__)
    root.add_argument("--version", action="version", version="scCody 0.1.0")
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-config", help="Write editable settings")
    init.add_argument("path", type=Path)
    train = sub.add_parser("train", help="Fit representation and frozen-latent FlowMap")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--outdir", required=True, type=Path)
    for name in ("generate", "localize"):
        command = sub.add_parser(name)
        command.add_argument("--model", required=True, type=Path)
        command.add_argument("--data", required=True, type=Path)
        command.add_argument("--device", default="auto")
        if name == "generate":
            command.add_argument("--target-age", required=True, type=float)
            command.add_argument("--output", required=True, type=Path)
        else:
            command.add_argument("--candidate-times", required=True, type=numbers)
            command.add_argument("--source-times", type=numbers)
            command.add_argument("--outdir", required=True, type=Path)
            command.add_argument("--max-cells", type=int, default=2000)
            command.add_argument("--min-cells", type=int, default=100)
            command.add_argument("--population-scope", choices=["global", "cell_type"])
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "init-config":
            if args.path.exists():
                raise FileExistsError(args.path)
            ScCodyConfig().save(args.path)
            return 0
        import anndata as ad
        from .model import ScCody
        data = ad.read_h5ad(args.data)
        if args.command == "train":
            ScCody(ScCodyConfig.load(args.config)).fit(data, args.outdir)
        else:
            model = ScCody.load(args.model, device=args.device)
            if args.command == "generate":
                if args.output.exists():
                    raise FileExistsError(args.output)
                model.generate(data, args.target_age).write_h5ad(args.output)
            else:
                model.localize(data, args.candidate_times, source_times=args.source_times,
                               max_cells=args.max_cells, min_cells=args.min_cells,
                               population_scope=args.population_scope, balanced_cells=True).save(args.outdir)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, RuntimeError) as exc:
        print(f"sccody: {exc}", file=sys.stderr)
        return 2
