from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .checkpoint import atomic_json
from .baselines.config import BASELINE_NAMES, BaselineConfig
from .baselines.runner import (
    inspect_baseline,
    prepare_baselines,
    run_baseline,
    run_suite,
)
from .config import RunConfig
from .consistency import compare_runs
from .metric_report import collect_metric_report
from .pipeline import inspect_run, run
from .rule_prediction import evaluate_rule_model_landmarks
from .preprocess.aave import (
    DEFAULT_RPC_URLS,
    STATE_RPC_URLS,
    download_aave_pool_logs,
    preprocess_aave_full,
)
from .preprocess.wselob import (
    WSELOB_FILES,
    download_wselob,
    merge_wselob_datasets,
    preprocess_wselob,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crbstpp")
    commands = parser.add_subparsers(dest="command", required=True)
    preprocess = commands.add_parser("preprocess")
    datasets = preprocess.add_subparsers(dest="dataset_name", required=True)
    aave = datasets.add_parser(
        "aave",
        help="download and build Ethereum Aave V2/V3 borrower histories",
    )
    aave.add_argument(
        "--raw-root", type=Path, default=Path("data/aave/raw/ethereum_v2_v3")
    )
    aave.add_argument("--output-root", type=Path)
    aave.add_argument("--download", action="store_true")
    aave.add_argument(
        "--build-dataset",
        action="store_true",
        help="build fit-ready wallet/version debt episodes from downloaded logs",
    )
    aave.add_argument("--partition-seed", type=int, default=111)
    aave.add_argument(
        "--partition-fractions",
        type=float,
        nargs=3,
        metavar=("FIT", "CERT", "TEST"),
        default=(0.5, 0.3, 0.2),
        help="wallet-hash fit/cert/test fractions (v13 default: 0.5 0.3 0.2)",
    )
    aave.add_argument(
        "--rpc-url",
        action="append",
        help="repeat for deterministic RPC failover; public defaults are used if omitted",
    )
    aave.add_argument("--version", action="append", choices=("v2", "v3"))
    aave.add_argument("--end-block", type=int)
    aave.add_argument("--chunk-size", type=int, default=10_000)
    aave.add_argument("--workers", type=int, default=4)
    aave.add_argument(
        "--skip-market-state",
        action="store_true",
        help="download wallet actions only; the primary Aave dataset keeps market state",
    )
    aave.add_argument("--overwrite", action="store_true")
    wselob = datasets.add_parser(
        "wselob",
        help="download and build stock-day WSELOB-2017 microstructure histories",
    )
    wselob.add_argument("--raw-root", type=Path, default=Path("data/wselob_2017/raw"))
    wselob.add_argument("--output-root", type=Path)
    wselob.add_argument("--stock", choices=tuple(WSELOB_FILES), default="PEKAO")
    wselob.add_argument("--download", action="store_true")
    wselob.add_argument("--build-dataset", action="store_true")
    wselob.add_argument(
        "--merge-input-root",
        type=Path,
        nargs="+",
        help="merge separately preprocessed WSELOB stocks",
    )
    wselob.add_argument("--impact-seconds", type=int, default=30)
    wselob.add_argument(
        "--continuous-time-unit",
        choices=("second", "millisecond"),
        default="millisecond",
        help=(
            "model time unit for continuous WSELOB data; millisecond keeps "
            "sub-second formation-window quantiles without discretizing raw timestamps"
        ),
    )
    wselob.add_argument("--kernel-knots", type=int, default=4)
    wselob.add_argument("--baseline-bins", type=int, default=4)
    wselob.add_argument(
        "--partition-fractions",
        type=float,
        nargs=3,
        metavar=("FIT", "CERT", "TEST"),
        default=(0.5, 0.3, 0.2),
    )
    wselob.add_argument(
        "--partition-method",
        choices=("ordered", "month_stratified"),
        default="month_stratified",
    )
    wselob.add_argument("--partition-seed", type=int, default=111)
    wselob.add_argument("--diagnostic-max-days", type=int)
    wselob.add_argument("--target-horizon-seconds", type=int, default=30)
    wselob.add_argument("--target-quantile", type=float, default=0.90)
    wselob.add_argument("--target-rearm-fraction", type=float, default=0.50)
    wselob.add_argument("--overwrite", action="store_true")
    fit = commands.add_parser("fit")
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--run-dir", type=Path)
    resume = commands.add_parser("resume")
    resume.add_argument("run_dir", type=Path)
    inspect_command = commands.add_parser("inspect")
    inspect_command.add_argument("run_dir", type=Path)
    consistency = commands.add_parser("consistency")
    consistency.add_argument("run_dirs", nargs="+", type=Path)
    consistency.add_argument("--output", type=Path, required=True)
    metrics = commands.add_parser(
        "metrics", help="collect completed baseline and rule-search metrics"
    )
    metrics.add_argument("--spec", type=Path, required=True)
    landmark_metrics = commands.add_parser(
        "evaluate-landmarks",
        help="evaluate a frozen discovered rule model on common landmark rows",
    )
    landmark_metrics.add_argument("--run-dir", type=Path, required=True)
    landmark_metrics.add_argument("--baseline-config", type=Path, required=True)
    baseline = commands.add_parser("baseline")
    baseline_commands = baseline.add_subparsers(
        dest="baseline_command", required=True
    )
    baseline_prepare = baseline_commands.add_parser("prepare")
    baseline_prepare.add_argument("--config", type=Path, required=True)
    baseline_prepare.add_argument("--seed", type=int, required=True)
    baseline_fit = baseline_commands.add_parser("fit")
    baseline_fit.add_argument("--config", type=Path, required=True)
    baseline_fit.add_argument("--model", choices=BASELINE_NAMES, required=True)
    baseline_fit.add_argument("--seed", type=int, required=True)
    baseline_suite = baseline_commands.add_parser("suite")
    baseline_suite.add_argument("--config", type=Path, required=True)
    baseline_suite.add_argument("--seed", type=int, required=True)
    baseline_inspect = baseline_commands.add_parser("inspect")
    baseline_inspect.add_argument("run_dir", type=Path)
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
                    line.split()
                    for line in path.read_text(encoding="utf-8").splitlines()
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
        oom_delta = None if before is None or after is None else max(0, after - before)
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
    if args.command == "baseline":
        if args.baseline_command == "inspect":
            payload = inspect_baseline(args.run_dir)
        else:
            config = BaselineConfig.from_yaml(args.config)
            if args.baseline_command == "prepare":
                prepared = prepare_baselines(config, seed=args.seed)
                payload = {
                    "root": str(prepared.root),
                    "manifest": prepared.manifest,
                    "seed": int(args.seed),
                }
            elif args.baseline_command == "fit":
                payload = run_baseline(
                    config, args.model, seed=args.seed
                )
            else:
                payload = run_suite(config, seed=args.seed)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "preprocess":
        if args.dataset_name == "wselob":
            if args.merge_input_root:
                if args.output_root is None:
                    raise SystemExit("wselob merge requires --output-root")
                if args.download or args.build_dataset:
                    raise SystemExit(
                        "wselob merge cannot be combined with download/build-dataset"
                    )
                output = merge_wselob_datasets(
                    tuple(args.merge_input_root),
                    args.output_root,
                    overwrite=args.overwrite,
                )
            elif args.download:
                download_wselob(args.raw_root, stock=args.stock)
            if args.merge_input_root:
                pass
            elif args.build_dataset:
                if args.output_root is None:
                    raise SystemExit("wselob --build-dataset requires --output-root")
                output = preprocess_wselob(
                    args.raw_root,
                    args.output_root,
                    stock=args.stock,
                    bin_seconds=1,
                    continuous=True,
                    continuous_impact_seconds=args.impact_seconds,
                    continuous_knot_count=args.kernel_knots,
                    continuous_baseline_bins=args.baseline_bins,
                    continuous_time_unit=args.continuous_time_unit,
                    partition_fractions=tuple(args.partition_fractions),
                    partition_method=args.partition_method,
                    partition_seed=args.partition_seed,
                    diagnostic_max_days=args.diagnostic_max_days,
                    predicate_schema="mechanism_v5",
                    target_mode="volatility_burst",
                    target_horizon_seconds=args.target_horizon_seconds,
                    target_quantile=args.target_quantile,
                    target_rearm_fraction=args.target_rearm_fraction,
                    context_states=True,
                    overwrite=args.overwrite,
                )
            elif args.download:
                output = args.raw_root
            else:
                raise SystemExit("wselob requires --download and/or --build-dataset")
        else:
            if args.download:
                output = download_aave_pool_logs(
                    args.raw_root,
                    rpc_urls=tuple(args.rpc_url or DEFAULT_RPC_URLS),
                    versions=tuple(args.version or ("v2", "v3")),
                    end_block=args.end_block,
                    chunk_size=args.chunk_size,
                    workers=args.workers,
                    include_market_state=not args.skip_market_state,
                )
            elif args.build_dataset and args.output_root is not None:
                output = preprocess_aave_full(
                    args.raw_root,
                    args.output_root,
                    partition_seed=args.partition_seed,
                    partition_fractions=tuple(args.partition_fractions),
                    rpc_urls=tuple(args.rpc_url or STATE_RPC_URLS),
                    workers=args.workers,
                    include_history_states=False,
                    overwrite=args.overwrite,
                )
            else:
                raise SystemExit(
                    "aave requires --download or --build-dataset with --output-root"
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
    if args.command == "metrics":
        payload = collect_metric_report(args.spec)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate-landmarks":
        payload = evaluate_rule_model_landmarks(
            args.run_dir, args.baseline_config
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "resume":
        config = RunConfig.from_yaml(args.run_dir / "config.yaml")
        report = run(config, run_dir=args.run_dir, resume=True)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "consistency":
        payload = compare_runs(tuple(args.run_dirs), output=args.output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(json.dumps(inspect_run(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
