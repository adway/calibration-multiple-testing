# run_calibration_submitit.py
import os
import sys
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import submitit

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from calibration_helpers import make_pvals, calibrate_and_assess


M = 500
B = 100_000
pi0s = [0.5, 0.75, 0.9]
alphas = [1.5]
beta = 2.3

n_grid = [500, 1_000, 5_000, 10_000, 50_000]

calibrators = [
    "g-lfdr",
    "q-value",
    "p-value",
    "l-lfdr",
    "s-mle-lfdr",
]

N_JOBS = 500

OUTDIR = ROOT / "calibration_results"
LOGDIR = ROOT / "run_logs"


def split_into_chunks(tasks, n_chunks):
    chunks = [[] for _ in range(n_chunks)]
    for idx, task in enumerate(tasks):
        chunks[idx % n_chunks].append(task)
    return chunks


def run_task_chunk(chunk_id, tasks):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for task in tasks:
        pi0 = task["pi0"]
        alpha = task["alpha"]
        rep = task["rep"]
        n = task["n"]
        seed = task["seed"]

        print(task, flush=True)

        try:
            np.random.seed(seed)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                labels, pvals = make_pvals(
                    n=n,
                    pi0=pi0,
                    alpha=alpha,
                    beta=beta,
                )

                for calibrator in calibrators:
                    try:
                        err = calibrate_and_assess(
                            pvals,
                            labels,
                            calibrator=calibrator,
                            pi0=pi0,
                            alpha=alpha,
                            beta=beta,
                            use_MC=True,
                            B=B,
                        )

                        rows.append({
                            "chunk_id": chunk_id,
                            "rep": rep,
                            "seed": seed,
                            "n": n,
                            "pi0": pi0,
                            "alpha": alpha,
                            "beta": beta,
                            "calibrator": calibrator,
                            "calibration_error": err,
                            "error": None,
                        })

                    except Exception as e:
                        rows.append({
                            "chunk_id": chunk_id,
                            "rep": rep,
                            "seed": seed,
                            "n": n,
                            "pi0": pi0,
                            "alpha": alpha,
                            "beta": beta,
                            "calibrator": calibrator,
                            "calibration_error": None,
                            "error": repr(e),
                        })

        except Exception as e:
            rows.append({
                "chunk_id": chunk_id,
                "rep": rep,
                "seed": seed,
                "n": n,
                "pi0": pi0,
                "alpha": alpha,
                "beta": beta,
                "calibrator": None,
                "calibration_error": None,
                "error": repr(e),
            })

    path = OUTDIR / f"results_chunk_{chunk_id:04d}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    return str(path)


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    tasks = []

    base_seed = 12345
    task_id = 0

    for pi0 in pi0s:
        for alpha in alphas:
            for rep in range(M):
                for n in n_grid:
                    tasks.append({
                        "pi0": pi0,
                        "alpha": alpha,
                        "rep": rep,
                        "n": n,
                        "seed": base_seed + task_id,
                    })
                    task_id += 1

    random.seed(123)
    random.shuffle(tasks)

    chunks = split_into_chunks(tasks, N_JOBS)

    print(f"Total datasets: {len(tasks)}")
    print(f"Total jobs: {len(chunks)}")
    print(f"Tasks per job: about {len(tasks) / len(chunks):.1f}")

    executor = submitit.AutoExecutor(folder=str(LOGDIR))
    executor.update_parameters(
        slurm_job_name="calib",
        slurm_partition="standard",
        # Set this to your cluster account before submitting jobs.
        slurm_account="YOUR_SLURM_ACCOUNT",
        slurm_time=360,
        slurm_mem="16G",
        cpus_per_task=1,
        tasks_per_node=1,
        slurm_setup=[
        f"export PYTHONPATH={SRC}:$PYTHONPATH",
        f"cd {ROOT}"]
    )

    jobs = []
    with executor.batch():
        for chunk_id, chunk in enumerate(chunks):
            jobs.append(
                executor.submit(run_task_chunk, chunk_id, chunk)
            )

    print(f"Submitted {len(jobs)} jobs.")
    print([job.job_id for job in jobs[:10]])
