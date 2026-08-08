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
from .consistency import compare_runs
from .pipeline import inspect_run, run
from .preprocess.aave import (
    DEFAULT_RPC_URLS,
    STATE_RPC_URLS,
    download_aave_pool_logs,
    preprocess_aave_full,
    stage_finsurvival_sample,
)
from .preprocess.freddie import preprocess_freddie
from .preprocess.home_credit import preprocess_home_credit
from .preprocess.ibm import preprocess_ibm
from .preprocess.wselob import WSELOB_FILES, download_wselob, preprocess_wselob


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
    ibm.add_argument("--partition-seed", type=int, default=111)
    ibm.add_argument("--overwrite", action="store_true")
    home_credit = datasets.add_parser(
        "home-credit",
        help=(
            "build client-level multi-product Home Credit histories with a "
            "first 30+ DPD target"
        ),
    )
    home_credit.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "data/home_credit_default_risk/kagglehub_cache/competitions/"
            "home-credit-default-risk"
        ),
    )
    home_credit.add_argument("--output-root", type=Path, required=True)
    home_credit.add_argument("--partition-seed", type=int, default=111)
    home_credit.add_argument("--max-observation-months", type=int, default=36)
    home_credit.add_argument(
        "--partition-fractions",
        type=float,
        nargs=3,
        metavar=("FIT", "CERT", "TEST"),
        default=(0.5, 0.3, 0.2),
        help="client-hash fit/cert/test fractions (default: 0.5 0.3 0.2)",
    )
    home_credit.add_argument(
        "--diagnostic-max-clients",
        type=int,
        help=(
            "target-blind deterministic client cap for smoke tests only; "
            "omit for the primary estimator"
        ),
    )
    home_credit.add_argument(
        "--target-source",
        choices=(
            "pooled_first",
            "unified",
            "bureau",
            "credit_card",
            "pos_cash",
            "all_recurrent",
        ),
        default="pooled_first",
        help=(
            "target process: legacy pooled first 30+ DPD, one recurrent source, "
            "or all three recurrent source datasets"
        ),
    )
    home_credit.add_argument("--overwrite", action="store_true")
    aave = datasets.add_parser(
        "aave",
        help="download Ethereum Aave V2/V3 Pool logs or normalize a raw sample",
    )
    aave.add_argument(
        "--raw-root", type=Path, default=Path("data/aave/raw/ethereum_v2_v3")
    )
    aave.add_argument("--output-root", type=Path)
    aave.add_argument("--sample-csv", type=Path)
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
    aave.add_argument(
        "--history-states",
        action="store_true",
        help=(
            "append fit-frozen recent/recurrent/accelerating/decelerating "
            "history-state predicates (v14)"
        ),
    )
    aave.add_argument("--overwrite", action="store_true")
    wselob = datasets.add_parser(
        "wselob",
        help="download and build stock-day WSELOB-2017 microstructure histories",
    )
    wselob.add_argument("--raw-root", type=Path, default=Path("data/wselob_2017/raw"))
    wselob.add_argument("--output-root", type=Path)
    wselob.add_argument(
        "--stock", choices=tuple(WSELOB_FILES), default="PEKAO"
    )
    wselob.add_argument("--download", action="store_true")
    wselob.add_argument("--build-dataset", action="store_true")
    wselob.add_argument("--bin-seconds", type=int, default=5)
    wselob.add_argument(
        "--continuous",
        action="store_true",
        help="use exact raw-timestamp recurrent Poisson risk intervals",
    )
    wselob.add_argument("--impact-seconds", type=int, default=60)
    wselob.add_argument("--kernel-knots", type=int, default=4)
    wselob.add_argument("--baseline-bins", type=int, default=8)
    wselob.add_argument(
        "--partition-fractions",
        type=float,
        nargs=3,
        metavar=("FIT", "CERT", "TEST"),
        default=(0.5, 0.3, 0.2),
    )
    wselob.add_argument("--diagnostic-max-days", type=int)
    wselob.add_argument(
        "--predicate-schema",
        choices=("legacy", "mechanism_v2"),
        default="legacy",
    )
    wselob.add_argument(
        "--target-mode",
        choices=("down_tick", "adverse_excursion"),
        default="down_tick",
    )
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
        elif args.dataset_name == "ibm":
            output = preprocess_ibm(
                args.raw_zip,
                args.output_root,
                partition_seed=args.partition_seed,
                overwrite=args.overwrite,
            )
        elif args.dataset_name == "home-credit":
            output = preprocess_home_credit(
                args.input_root,
                args.output_root,
                partition_seed=args.partition_seed,
                partition_fractions=tuple(args.partition_fractions),
                max_observation_months=args.max_observation_months,
                diagnostic_max_clients=args.diagnostic_max_clients,
                target_source=args.target_source,
                overwrite=args.overwrite,
            )
        elif args.dataset_name == "wselob":
            if args.download:
                download_wselob(args.raw_root, stock=args.stock)
            if args.build_dataset:
                if args.output_root is None:
                    raise SystemExit("wselob --build-dataset requires --output-root")
                output = preprocess_wselob(
                    args.raw_root,
                    args.output_root,
                    stock=args.stock,
                    bin_seconds=args.bin_seconds,
                    continuous=args.continuous,
                    continuous_impact_seconds=args.impact_seconds,
                    continuous_knot_count=args.kernel_knots,
                    continuous_baseline_bins=args.baseline_bins,
                    partition_fractions=tuple(args.partition_fractions),
                    diagnostic_max_days=args.diagnostic_max_days,
                    predicate_schema=args.predicate_schema,
                    target_mode=args.target_mode,
                    target_horizon_seconds=args.target_horizon_seconds,
                    target_quantile=args.target_quantile,
                    target_rearm_fraction=args.target_rearm_fraction,
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
                    include_history_states=args.history_states,
                    overwrite=args.overwrite,
                )
            elif args.sample_csv is not None and args.output_root is not None:
                output = stage_finsurvival_sample(
                    args.sample_csv,
                    args.output_root,
                    overwrite=args.overwrite,
                )
            else:
                raise SystemExit(
                    "aave requires --download, --build-dataset with --output-root, "
                    "or both --sample-csv and --output-root"
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
    if args.command == "consistency":
        payload = compare_runs(tuple(args.run_dirs), output=args.output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(json.dumps(inspect_run(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
