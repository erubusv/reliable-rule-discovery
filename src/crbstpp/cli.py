from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .checkpoint import atomic_json
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
    freddie.add_argument("--test-vintage", action="append", default=[])
    freddie.add_argument("--development-fit-fraction", type=float, default=0.75)
    freddie.add_argument("--partition-seed", type=int, default=111)
    freddie.add_argument("--max-observation-months", type=int, default=36)
    freddie.add_argument("--overwrite", action="store_true")
    ibm = datasets.add_parser("ibm")
    ibm.add_argument(
        "--raw-zip", type=Path, default=Path("data/ibm_aml/raw/HI-Small_Trans.csv.zip")
    )
    ibm.add_argument("--output-root", type=Path, required=True)
    ibm.add_argument("--overwrite", action="store_true")
    fit = commands.add_parser("fit")
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--run-dir", type=Path)
    resume = commands.add_parser("resume")
    resume.add_argument("run_dir", type=Path)
    inspect_command = commands.add_parser("inspect")
    inspect_command.add_argument("run_dir", type=Path)
    return parser


def _oom_kills() -> int | None:
    for path in (
        Path("/sys/fs/cgroup/memory.events.local"),
        Path("/sys/fs/cgroup/memory.events"),
    ):
        try:
            values = {
                key: int(value)
                for key, value in (
                    line.split() for line in path.read_text(encoding="utf-8").splitlines()
                )
            }
        except (OSError, ValueError):
            continue
        if "oom_kill" in values:
            return values["oom_kill"]
    return None


def _supervised_fit(config_path: Path, run_dir: Path) -> int:
    """Run the memory-owning worker while preserving fatal exit evidence."""
    run_dir = Path(run_dir)
    stderr_path = run_dir.with_name(f"{run_dir.name}.stderr.log")
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "crbstpp.cli",
        "fit",
        "--config",
        str(config_path),
        "--run-dir",
        str(run_dir),
    ]
    environment = os.environ.copy()
    environment["CRBSTPP_SUPERVISED_WORKER"] = "1"
    before = _oom_kills()
    started = time.time()
    with stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            env=environment,
            stderr=stderr,
            check=False,
        )
    after = _oom_kills()
    run_dir.mkdir(parents=True, exist_ok=True)
    final_stderr = run_dir / "stderr.log"
    os.replace(stderr_path, final_stderr)
    if completed.returncode != 0:
        oom_delta = (
            None if before is None or after is None else max(0, after - before)
        )
        signal_number = -completed.returncode if completed.returncode < 0 else None
        atomic_json(
            run_dir / "failure.json",
            {
                "schema": "crbstpp.failure.v1",
                "returncode": completed.returncode,
                "signal": signal_number,
                "reason": (
                    "oom_kill"
                    if signal_number == 9 and oom_delta is not None and oom_delta > 0
                    else "worker_failure"
                ),
                "oom_kill_before": before,
                "oom_kill_after": after,
                "oom_kill_delta": oom_delta,
                "elapsed_seconds": time.time() - started,
                "stderr": str(final_stderr),
            },
        )
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preprocess":
        if args.dataset_name == "freddie":
            output = preprocess_freddie(
                args.input_root,
                args.output_root,
                vintages=tuple(args.vintage),
                test_vintages=tuple(args.test_vintage),
                development_fit_fraction=args.development_fit_fraction,
                partition_seed=args.partition_seed,
                max_observation_months=args.max_observation_months,
                overwrite=args.overwrite,
            )
        else:
            output = preprocess_ibm(
                args.raw_zip,
                args.output_root,
                overwrite=args.overwrite,
            )
        print(output)
        return 0
    if args.command == "fit":
        if (
            args.run_dir is not None
            and os.environ.get("CRBSTPP_SUPERVISED_WORKER") != "1"
        ):
            return _supervised_fit(args.config, args.run_dir)
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
