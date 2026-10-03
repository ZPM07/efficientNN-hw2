"""Job orchestrator: queue every run of the design onto the available GPUs.

Phase 1: scoring runs. Barrier: average scores and build subsets.
Phase 2: retraining runs.

Idempotent: jobs whose output file already exists are skipped, so the
orchestrator can be restarted freely. Launch detached, e.g.

    (setsid nohup .venv/bin/python code/orchestrate.py > logs/orchestrator.log 2>&1 < /dev/null &)
"""

import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from el2n_nlp import config  # noqa: E402

PY = str(config.ROOT / ".venv" / "bin" / "python")
GPUS = [0, 1]
CLI = [PY, "-m", "el2n_nlp.cli"]


def build_jobs() -> tuple[list[dict], list[dict]]:
    """Phase 1: scoring runs (must finish before `select`); phase 2: retraining."""
    scoring = []
    for seed in config.scoring_seeds():
        scoring.append(
            {
                "name": f"score_seed{seed}",
                "out": config.score_path(seed),
                "cmd": CLI + ["score", "--seed", str(seed)],
            }
        )
    retraining = []
    for run in range(config.RETRAIN_RUNS):
        retraining.append(
            {
                "name": config.run_tag("full", 100, run),
                "out": config.run_path("full", 100, run),
                "cmd": CLI + ["train", "--mode", "full", "--run", str(run)],
            }
        )
    for mode in ("el2n", "random"):
        for pct in config.SIZES_PCT:
            for run in range(config.RETRAIN_RUNS):
                retraining.append(
                    {
                        "name": config.run_tag(mode, pct, run),
                        "out": config.run_path(mode, pct, run),
                        "cmd": CLI
                        + ["train", "--mode", mode, "--pct", str(pct), "--run", str(run)],
                    }
                )
    return scoring, retraining


def run_phase(jobs: list[dict]) -> list[str]:
    queue = deque(j for j in jobs if not j["out"].exists())
    skipped = len(jobs) - len(queue)
    print(f"{len(queue)} jobs to run, {skipped} already done", flush=True)

    running: list[tuple[subprocess.Popen, dict, int]] = []
    gpu_free = deque(GPUS)
    t0 = time.time()
    failed = []

    while queue or running:
        while queue and gpu_free:
            job = queue.popleft()
            gpu = gpu_free.popleft()
            log = open(config.LOGS_DIR / f"{job['name']}.log", "w")
            proc = subprocess.Popen(
                job["cmd"],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=config.ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            )
            running.append((proc, job, gpu))
            print(
                f"[{time.time() - t0:7.0f}s] START gpu{gpu} {job['name']} "
                f"({len(queue)} queued, {len(running)} running)",
                flush=True,
            )
        time.sleep(5)
        for item in list(running):
            proc, job, gpu = item
            rc = proc.poll()
            if rc is not None:
                running.remove(item)
                gpu_free.append(gpu)
                status = "OK" if rc == 0 else f"FAIL rc={rc}"
                if rc != 0:
                    failed.append(job["name"])
                print(f"[{time.time() - t0:7.0f}s] {status} gpu{gpu} {job['name']}", flush=True)
    if failed:
        print("FAILED JOBS:", failed, flush=True)
    return failed


def main():
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    scoring, retraining = build_jobs()

    failed = run_phase(scoring)
    if failed:
        raise SystemExit("scoring phase incomplete; rerun orchestrator")

    from el2n_nlp.subsets import select_all

    select_all()

    failed = run_phase(retraining)
    print("ALL PHASES FINISHED; failures:", failed, flush=True)


if __name__ == "__main__":
    main()
