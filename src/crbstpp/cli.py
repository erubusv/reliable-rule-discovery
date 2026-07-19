from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RunConfig
from .pipeline import inspect_run, run
from .preprocess.freddie import preprocess_freddie
from .preprocess.ibm import preprocess_ibm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crbstpp")
    commands = parser.add_subparsers(dest="command", required=True)
    preprocess = commands.add_parser("preprocess")
    datasets = preprocess.add_subparsers(dest="dataset_name", required=True)
    freddie = datasets.add_parser("freddie")
    freddie.add_argument("--input-root", type=Path, default=Path("data/freddiemac"))
    freddie.add_argument("--output-root", type=Path, required=True)
    freddie.add_argument("--vintage", action="append", default=[])
    freddie.add_argument("--overwrite", action="store_true")
    ibm = datasets.add_parser("ibm")
    ibm.add_argument("--raw-zip", type=Path, default=Path("data/ibm_aml/raw/HI-Small_Trans.csv.zip"))
    ibm.add_argument("--output-root", type=Path, required=True)
    ibm.add_argument("--max-rows", type=int)
    ibm.add_argument("--overwrite", action="store_true")
    fit = commands.add_parser("fit")
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--run-dir", type=Path)
    resume = commands.add_parser("resume")
    resume.add_argument("run_dir", type=Path)
    inspect_command = commands.add_parser("inspect")
    inspect_command.add_argument("run_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preprocess":
        if args.dataset_name == "freddie":
            output = preprocess_freddie(
                args.input_root, args.output_root,
                vintages=tuple(args.vintage), overwrite=args.overwrite,
            )
        else:
            output = preprocess_ibm(
                args.raw_zip, args.output_root,
                overwrite=args.overwrite, max_rows=args.max_rows,
            )
        print(output)
        return 0
    if args.command == "fit":
        config = RunConfig.from_yaml(args.config)
        report = run(config, run_dir=args.run_dir)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "resume":
        config = RunConfig.from_yaml(args.run_dir / "config.yaml")
        report = run(config, run_dir=args.run_dir, resume=True)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    print(json.dumps(inspect_run(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

