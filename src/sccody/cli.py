"""Thin command-line interface over the public pipeline API."""
import argparse
import json
import logging
from pathlib import Path
import sys
from .config import PipelineConfig


def numbers(value):
    try:
        return [float(v) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated numbers, such as 0,1,2") from exc


def _evaluate(pipeline, target, times, outdir):
    from .evaluation import localization_metrics, reconstruction_metrics
    from .pipeline import write_json
    localized = pipeline.localize(target, times)
    localized.save(outdir)
    d = pipeline.config.data
    metrics = localization_metrics(localized.predictions, target.obs, sample_key=d.sample_key, time_key=d.time_key,
                                   cell_type_key=d.cell_type_key if pipeline.config.model.scope == "cell_type" else None)
    reconstruction_metrics(pipeline, target).to_csv(outdir / "reconstruction.csv", index=False)
    write_json(outdir / "metrics.json", metrics)
    return metrics


def parser():
    root = argparse.ArgumentParser(prog="sccody", description="Single-cell population transport from your own AnnData")
    root.add_argument("--version", action="version", version="scCody 0.1.0")
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-config", help="Write an editable configuration template")
    init.add_argument("path", type=Path)
    for name in ("validate", "train", "run"):
        command = sub.add_parser(name, help={"validate": "Check schema, splits, and training support", "train": "Fit and save a new pipeline", "run": "Fit and evaluate test samples"}[name])
        command.add_argument("--data", required=True, type=Path)
        command.add_argument("--config", required=True, type=Path)
        if name != "validate":
            command.add_argument("--outdir", required=True, type=Path)
        if name == "run":
            command.add_argument("--candidate-times", required=True, type=numbers)
    for name in ("generate", "localize", "evaluate"):
        command = sub.add_parser(name, help={"generate": "Generate cells at a known time", "localize": "Infer unknown time from expression", "evaluate": "Evaluate known heldout populations"}[name])
        command.add_argument("--model", required=True, type=Path)
        command.add_argument("--data", required=True, type=Path)
        command.add_argument("--device", default="auto")
        if name == "generate":
            command.add_argument("--target-time", required=True, type=float)
            command.add_argument("--output", required=True, type=Path)
        else:
            command.add_argument("--candidate-times", required=True, type=numbers)
            command.add_argument("--outdir", required=True, type=Path)
        if name == "localize":
            command.add_argument("--source-times", type=numbers)
            command.add_argument("--max-cells", type=int, default=512)
            command.add_argument("--min-cells", type=int, default=5)
    demo = sub.add_parser("demo", help="Run a bounded synthetic example on CPU")
    demo.add_argument("--outdir", required=True, type=Path)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "init-config":
            if args.path.exists():
                raise FileExistsError(f"Configuration already exists: {args.path}")
            PipelineConfig().save(args.path)
            print(f"Configuration written to {args.path}")
            return 0
        import anndata as ad
        from .data import prepare
        from .pipeline import Pipeline
        if args.command == "demo":
            from .demo import synthetic_data
            data, config = synthetic_data()
            args.outdir.mkdir(parents=True, exist_ok=False)
            data.write_h5ad(args.outdir / "synthetic.h5ad")
            config.save(args.outdir / "example.yaml")
            pipeline = Pipeline(config).fit(data, args.outdir / "model")
            heldout = data[data.obs[config.data.sample_key].isin(config.split.test_samples)].copy()
            metrics = _evaluate(pipeline, heldout, [0, 1, 2], args.outdir / "evaluation")
            source = data[data.obs[config.data.sample_key] == "replicate_0"].copy()
            pipeline.generate(source, 2).write_h5ad(args.outdir / "generated.h5ad")
            print(json.dumps({"outdir": str(args.outdir), "metrics": metrics}))
            return 0
        data = ad.read_h5ad(args.data)
        if args.command in {"validate", "train", "run"}:
            config = PipelineConfig.load(args.config)
            if args.command == "validate":
                prepared = prepare(data, config)
                print(json.dumps({"genes": len(prepared.genes), "train_cells": len(prepared.train_indices),
                                  "validation_cells": len(prepared.validation_indices), "test_cells": len(prepared.test_indices),
                                  "time_origin": prepared.time_origin, "time_scale": prepared.time_scale}))
            else:
                # A full run needs a real test partition before any training is launched.
                if args.command == "run" and not len(prepare(data, config).test_indices):
                    raise ValueError("run requires test samples; use train for fitting without external evaluation")
                pipeline = Pipeline(config).fit(data, args.outdir)
                if args.command == "run":
                    test_ids = pipeline.split_table.loc[pipeline.split_table["split"] == "test", config.data.sample_key]
                    heldout = data[data.obs[config.data.sample_key].astype(str).isin(test_ids)].copy()
                    _evaluate(pipeline, heldout, args.candidate_times, args.outdir / "evaluation")
                print(f"Pipeline saved to {args.outdir / 'model.pt'}")
        else:
            pipeline = Pipeline.load(args.model, device=args.device)
            if args.command == "generate":
                if args.output.exists():
                    raise FileExistsError(f"Output already exists: {args.output}")
                pipeline.generate(data, args.target_time).write_h5ad(args.output)
                print(f"Generated cells written to {args.output}")
            elif args.command == "localize":
                pipeline.localize(data, args.candidate_times, source_times=args.source_times,
                                  max_cells=args.max_cells, min_cells=args.min_cells).save(args.outdir)
                print(f"Localization written to {args.outdir}")
            else:
                print(json.dumps(_evaluate(pipeline, data, args.candidate_times, args.outdir)))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, RuntimeError) as exc:
        print(f"sccody: {exc}", file=sys.stderr)
        return 2
