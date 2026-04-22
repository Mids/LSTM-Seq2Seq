from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


LOSS_PATTERN = re.compile(r"epoch=(\d+)\s+train_loss=([0-9.]+)\s+val_loss=([0-9.]+)")
DEVICE_PATTERN = re.compile(r"training_device=(\w+)")
RUN_DIR_PATTERN = re.compile(r"tensorboard_run_dir=(.+)")
IS_WINDOWS = os.name == "nt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded hyperparameter sweep for lstm-seq2seq.",
    )
    parser.add_argument("--minutes", type=float, default=5.0, help="Time budget per trial.")
    parser.add_argument("--max-runs", type=int, help="Optional cap on the number of trials.")
    parser.add_argument("--csv-path", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-sizes", default="64,128")
    parser.add_argument("--learning-rates", default="0.001,0.0005")
    parser.add_argument("--embedding-dims", default="256")
    parser.add_argument("--hidden-dims", default="512,768")
    parser.add_argument("--num-layers-options", default="1,2")
    parser.add_argument("--max-source-tokens-options", default="128")
    parser.add_argument("--max-target-tokens-options", default="128")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_trials(args: argparse.Namespace) -> list[dict[str, int | float]]:
    trials: list[dict[str, int | float]] = []
    combinations = itertools.product(
        parse_int_list(args.batch_sizes),
        parse_float_list(args.learning_rates),
        parse_int_list(args.embedding_dims),
        parse_int_list(args.hidden_dims),
        parse_int_list(args.num_layers_options),
        parse_int_list(args.max_source_tokens_options),
        parse_int_list(args.max_target_tokens_options),
    )
    for (
        batch_size,
        learning_rate,
        embedding_dim,
        hidden_dim,
        num_layers,
        max_source_tokens,
        max_target_tokens,
    ) in combinations:
        trials.append(
            {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "max_source_tokens": max_source_tokens,
                "max_target_tokens": max_target_tokens,
            }
        )
    if args.max_runs is not None:
        return trials[: args.max_runs]
    return trials


def trial_command(
    trial: dict[str, int | float],
    args: argparse.Namespace,
    run_name: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "lstm-seq2seq",
        "train",
        "--epochs",
        "999999",
        "--csv-path",
        args.csv_path,
        "--artifact-dir",
        args.artifact_dir,
        "--train-size",
        str(args.train_size),
        "--val-size",
        str(args.val_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--batch-size",
        str(trial["batch_size"]),
        "--learning-rate",
        str(trial["learning_rate"]),
        "--embedding-dim",
        str(trial["embedding_dim"]),
        "--hidden-dim",
        str(trial["hidden_dim"]),
        "--num-layers",
        str(trial["num_layers"]),
        "--max-source-tokens",
        str(trial["max_source_tokens"]),
        "--max-target-tokens",
        str(trial["max_target_tokens"]),
        "--run-name",
        run_name,
    ]


def build_trial_run_name(trial_index: int, trial: dict[str, int | float]) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    lr_value = str(trial["learning_rate"]).replace(".", "p")
    return (
        f"sweep-{timestamp}-t{trial_index:03d}"
        f"-bs{trial['batch_size']}"
        f"-lr{lr_value}"
        f"-emb{trial['embedding_dim']}"
        f"-hid{trial['hidden_dim']}"
        f"-layers{trial['num_layers']}"
        f"-src{trial['max_source_tokens']}"
        f"-tgt{trial['max_target_tokens']}"
    )


def run_trial(
    trial_index: int,
    total_trials: int,
    trial: dict[str, int | float],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    run_name = build_trial_run_name(trial_index, trial)
    command = trial_command(trial, args, run_name)
    print(f"[{trial_index}/{total_trials}] running {run_name} with {trial}")
    started_at = time.time()
    completed = False
    timed_out = False

    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=not IS_WINDOWS,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0,
    )
    try:
        stdout, _ = process.communicate(timeout=args.minutes * 60.0)
        completed = process.returncode == 0
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_tree(process)
        stdout, _ = process.communicate()

    elapsed_seconds = time.time() - started_at
    output_path = output_dir / f"{run_name}.log"
    output_path.write_text(stdout, encoding="utf-8")

    metrics = extract_metrics(stdout)
    result: dict[str, object] = {
        "run_name": run_name,
        "trial_index": trial_index,
        "params": trial,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "timed_out": timed_out,
        "completed": completed,
        "returncode": process.returncode,
        "output_path": str(output_path),
        **metrics,
    }
    print(
        "  device={device} last_epoch={epoch} val_loss={val_loss} elapsed={elapsed}s".format(
            device=result.get("training_device", "unknown"),
            epoch=result.get("last_epoch", "n/a"),
            val_loss=result.get("val_loss", "n/a"),
            elapsed=result["elapsed_seconds"],
        )
    )
    return result


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    os.killpg(process.pid, signal.SIGKILL)


def extract_metrics(stdout: str) -> dict[str, object]:
    training_device = None
    tensorboard_run_dir = None
    last_epoch = None
    train_loss = None
    val_loss = None

    device_match = DEVICE_PATTERN.search(stdout)
    if device_match:
        training_device = device_match.group(1)

    run_dir_match = RUN_DIR_PATTERN.search(stdout)
    if run_dir_match:
        tensorboard_run_dir = run_dir_match.group(1).strip()

    for match in LOSS_PATTERN.finditer(stdout):
        last_epoch = int(match.group(1))
        train_loss = float(match.group(2))
        val_loss = float(match.group(3))

    return {
        "training_device": training_device,
        "tensorboard_run_dir": tensorboard_run_dir,
        "last_epoch": last_epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }


def main() -> None:
    args = parse_args()
    trials = build_trials(args)
    sweep_root = Path(args.artifact_dir) / "sweeps"
    sweep_name = time.strftime("sweep-%Y%m%d-%H%M%S")
    output_dir = sweep_root / sweep_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for index, trial in enumerate(trials, start=1):
        result = run_trial(index, len(trials), trial, args, output_dir)
        results.append(result)
        with (output_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=True))
            handle.write("\n")

    ranked = sorted(
        results,
        key=lambda item: (
            float("inf") if item["val_loss"] is None else float(item["val_loss"]),
            float(item["elapsed_seconds"]),
        ),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(ranked, indent=2), encoding="utf-8")

    print("\nTop runs:")
    for result in ranked[:5]:
        print(
            "{run_name} val_loss={val_loss} epoch={epoch} device={device} params={params}".format(
                run_name=result["run_name"],
                val_loss=result["val_loss"],
                epoch=result["last_epoch"],
                device=result["training_device"],
                params=result["params"],
            )
        )
    print(f"\nSaved sweep results to {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
