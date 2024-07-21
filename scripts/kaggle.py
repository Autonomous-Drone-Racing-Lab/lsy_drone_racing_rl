"""Kaggle competition auto-submission script.

Note:
    Please do not alter this script or ask the course supervisors first!
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sim import simulate

from lsy_drone_racing.utils.utils import load_config

logger = logging.getLogger(__name__)


def main():
    """Run the simulation N times and save the results as 'submission.csv'."""
    n_runs = 100
    config: str = "config/level3.yaml"
    checkpoint="Documentation/version_more_noise/new_best_solution/baseline_1/rl_model_2200000_steps.zip" # best thing so far
    controller: str = "lsy_drone_racing/controller/rl_controller.py"

    checkpoint_base_path = Path(checkpoint).parent / "config.yaml"
    if not checkpoint_base_path.exists():
        print("Could not restore config from checkpoint. Make sure that the original folder structure is preserved.")
        exit(1)
    rl_config = load_config(checkpoint_base_path).rl_config
    
    ep_times = simulate(config=config, controller=controller, checkpoint=checkpoint, n_runs=n_runs, gui=True, override_rl_config=rl_config)
    # Log the number of failed runs if any

    if failed := [x for x in ep_times if x is None]:
        logger.warning(f"{len(failed)} runs failed out of {n_runs}!")
    else:
        logger.info("All runs completed successfully!")

    no_failed = len(failed)
    failure_rate = no_failed / n_runs
    success_rate = 1 - failure_rate

    ep_times = [x for x in ep_times if x is not None]
    best_ep_time = min(ep_times)
    print(f"Best episode time: {best_ep_time}")
    data = {"ID": [i for i in range(len(ep_times))], "submission_time": ep_times}

    ep_times = np.array(ep_times)
    mean_time = np.mean(ep_times)
    median_time = np.median(ep_times)
    std_time = np.std(ep_times)
    
    with open("submission.csv", "w") as f:
        f.write(f"Total runs: {n_runs}, runs_completed: {n_runs - no_failed}, runs_failed: {no_failed}, failure_rate: {failure_rate}, success_rate: {success_rate}\n")
        f.write(f"Best episode time: {best_ep_time}, mean_time: {mean_time}, median_time: {median_time}, std_time: {std_time}\n")
    
    pd.DataFrame(data).to_csv("submission.csv", index=False, mode="a")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
