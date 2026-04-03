from __future__ import annotations

import csv
import itertools
import os
import re
import subprocess
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
EVAL_SCRIPT = ROOT / "scripts" / "run_post_train_evaluation.py"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "reports" / "trigger_param_search"
WER_PATTERNS = (
    re.compile(r"(?m)^WER:\s*([0-9]+(?:\.[0-9]+)?)$"),
    re.compile(r"Token-level WER:\s*`?([0-9]+(?:\.[0-9]+)?)`?"),
)

SEARCH_SPACE = {
    "min_motion_energy": [0.06, 0.08, 0.10],
    "min_action_frames": [12, 15, 18],
    "idle_patience": [5, 8, 12],
    "min_top_margin": [0.40, 0.55, 0.70],
}


@dataclass(frozen=True)
class TriggerParams:
    min_motion_energy: float
    min_action_frames: int
    idle_patience: int
    min_top_margin: float

    def slug(self) -> str:
        motion = f"{self.min_motion_energy:.2f}".replace(".", "p")
        margin = f"{self.min_top_margin:.2f}".replace(".", "p")
        return (
            f"mme_{motion}"
            f"__maf_{self.min_action_frames}"
            f"__idle_{self.idle_patience}"
            f"__margin_{margin}"
        )


@dataclass
class RunResult:
    params: TriggerParams
    wer: float
    returncode: int
    report_path: Path
    output_root: Path
    stdout_path: Path
    stderr_path: Path
    error: str = ""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Grid-search trigger-based realtime spotter parameters.")
    parser.add_argument(
        "--scripted-glob",
        default='i_you_mother_father_techer_sudent_want_like*.mp4',
        help="Scripted regression glob passed through to run_post_train_evaluation.py.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(ROOT / "artifacts_webcam9_relative_coord_v1"),
        help="Artifact directory passed through to run_post_train_evaluation.py.",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Directory for per-run reports and aggregate CSV output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of subprocess evaluations to run in parallel.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Timeout per evaluation run.",
    )
    return parser


def iter_search_space() -> list[TriggerParams]:
    combos = []
    for values in itertools.product(
        SEARCH_SPACE["min_motion_energy"],
        SEARCH_SPACE["min_action_frames"],
        SEARCH_SPACE["idle_patience"],
        SEARCH_SPACE["min_top_margin"],
    ):
        combos.append(
            TriggerParams(
                min_motion_energy=float(values[0]),
                min_action_frames=int(values[1]),
                idle_patience=int(values[2]),
                min_top_margin=float(values[3]),
            )
        )
    return combos


def parse_wer(stdout_text: str) -> float | None:
    for pattern in WER_PATTERNS:
        match = pattern.search(stdout_text)
        if match:
            return float(match.group(1))
    return None


def run_single_combo(
    params: TriggerParams,
    *,
    artifact_dir: Path,
    scripted_glob: str,
    run_root: Path,
    timeout_seconds: int,
) -> RunResult:
    combo_dir = run_root / params.slug()
    combo_dir.mkdir(parents=True, exist_ok=True)
    report_path = combo_dir / "post_train_evaluation_summary.md"
    output_root = combo_dir / "sessions"
    stdout_path = combo_dir / "stdout.log"
    stderr_path = combo_dir / "stderr.log"

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["TRIGGER_MIN_MOTION_ENERGY"] = f"{params.min_motion_energy:.2f}"
    env["TRIGGER_MIN_ACTION_FRAMES"] = str(params.min_action_frames)
    env["TRIGGER_IDLE_PATIENCE"] = str(params.idle_patience)
    env["TRIGGER_MIN_TOP_MARGIN"] = f"{params.min_top_margin:.2f}"

    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--artifact-dir",
        str(artifact_dir),
        "--engine-mode",
        "trigger_based",
        "--scripted-glob",
        str(scripted_glob),
        "--output-root",
        str(output_root),
        "--report-path",
        str(report_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        wer = parse_wer(completed.stdout)
        error = ""
        if completed.returncode != 0:
            error = f"subprocess exited with code {completed.returncode}"
        elif wer is None:
            error = "WER not found in stdout"
            wer = float("inf")
        return RunResult(
            params=params,
            wer=float("inf") if wer is None else float(wer),
            returncode=int(completed.returncode),
            report_path=report_path,
            output_root=output_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error=error,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        return RunResult(
            params=params,
            wer=float("inf"),
            returncode=-1,
            report_path=report_path,
            output_root=output_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error=f"timed out after {timeout_seconds}s",
        )


def write_results_csv(csv_path: Path, results: list[RunResult]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "min_motion_energy",
                "min_action_frames",
                "idle_patience",
                "min_top_margin",
                "wer",
                "returncode",
                "error",
                "report_path",
                "output_root",
                "stdout_path",
                "stderr_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "min_motion_energy": f"{result.params.min_motion_energy:.2f}",
                    "min_action_frames": result.params.min_action_frames,
                    "idle_patience": result.params.idle_patience,
                    "min_top_margin": f"{result.params.min_top_margin:.2f}",
                    "wer": "inf" if result.wer == float("inf") else f"{result.wer:.4f}",
                    "returncode": result.returncode,
                    "error": result.error,
                    "report_path": result.report_path,
                    "output_root": result.output_root,
                    "stdout_path": result.stdout_path,
                    "stderr_path": result.stderr_path,
                }
            )


def print_top_results(results: list[RunResult], top_k: int = 3) -> None:
    ranked = sorted(
        results,
        key=lambda item: (item.wer, item.params.min_motion_energy, item.params.min_action_frames, item.params.idle_patience, item.params.min_top_margin),
    )
    print("")
    print(f"Top {min(top_k, len(ranked))} parameter sets by WER:")
    for index, result in enumerate(ranked[:top_k], start=1):
        print(
            f"{index}. WER={result.wer:.4f} | "
            f"min_motion_energy={result.params.min_motion_energy:.2f}, "
            f"MIN_ACTION_FRAMES={result.params.min_action_frames}, "
            f"IDLE_PATIENCE={result.params.idle_patience}, "
            f"MIN_TOP_MARGIN={result.params.min_top_margin:.2f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    results_root = Path(args.results_root).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = results_root / f"grid_search_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    combos = iter_search_space()
    total = len(combos)
    print(f"Running trigger parameter grid search: {total} combinations")
    print(f"Artifact dir: {artifact_dir}")
    print(f"Results root: {run_root}")
    print(f"Workers: {max(1, int(args.workers))}")

    results: list[RunResult] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_to_params = {
            executor.submit(
                run_single_combo,
                params,
                artifact_dir=artifact_dir,
                scripted_glob=str(args.scripted_glob),
                run_root=run_root,
                timeout_seconds=int(args.timeout_seconds),
            ): params
            for params in combos
        }
        completed_count = 0
        for future in as_completed(future_to_params):
            result = future.result()
            results.append(result)
            completed_count += 1
            status = "OK" if result.returncode == 0 and result.wer != float("inf") else "FAIL"
            print(
                f"[{completed_count}/{total}] {status} "
                f"WER={result.wer:.4f} "
                f"mme={result.params.min_motion_energy:.2f} "
                f"maf={result.params.min_action_frames} "
                f"idle={result.params.idle_patience} "
                f"margin={result.params.min_top_margin:.2f}"
            )
            if result.error:
                print(f"  error: {result.error}")

    results = sorted(
        results,
        key=lambda item: (item.wer, item.params.min_motion_energy, item.params.min_action_frames, item.params.idle_patience, item.params.min_top_margin),
    )
    write_results_csv(run_root / "grid_search_results.csv", results)
    print_top_results(results, top_k=3)

    if not results or results[0].wer == float("inf"):
        raise SystemExit("No successful runs completed.")

    best = results[0]
    print("")
    print("Best result:")
    print(
        f"WER={best.wer:.4f} | "
        f"min_motion_energy={best.params.min_motion_energy:.2f}, "
        f"MIN_ACTION_FRAMES={best.params.min_action_frames}, "
        f"IDLE_PATIENCE={best.params.idle_patience}, "
        f"MIN_TOP_MARGIN={best.params.min_top_margin:.2f}"
    )
    print(f"CSV: {run_root / 'grid_search_results.csv'}")
    print(f"Best report: {best.report_path}")


if __name__ == "__main__":
    main()
