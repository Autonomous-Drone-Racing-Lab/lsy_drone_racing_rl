"""Kaggle competition auto-submission script.

Note:
    Please do not alter this script or ask the course supervisors first!
"""

import logging
import os
from pathlib import Path

import pandas as pd
from sim import simulate

from lsy_drone_racing.utils.utils import load_config

logger = logging.getLogger(__name__)


def main():
    """Run the simulation N times and save the results as 'submission.csv'."""
    n_runs = 10
    config: str = "/home/runner/work/lsy_drone_racing_rl/lsy_drone_racing_rl/config/level3.yaml"
    checkpoint="/home/runner/work/lsy_drone_racing_rl/lsy_drone_racing_rl/Documentation/version_more_noise/new_best_solution/baseline_1/rl_model_2200000_steps"
    controller: str = "lsy_drone_racing/controller/rl_controller.py"
    
    checkpoint_base_path_components = checkpoint.split("/")[:-1]
    checkpoint_base_path = Path(os.path.join(*checkpoint_base_path_components, "config.yaml"))
    if not checkpoint_base_path.exists():
        print("Could not restore config from checkpoint. Make sure that the original folder structure is preserved.")
    rl_config = load_config(checkpoint_base_path).rl_config
    
    ep_times = simulate(config=config, controller=controller, checkpoint=checkpoint, n_runs=n_runs, gui=False, override_rl_config=rl_config)
    # Log the number of failed runs if any

    if failed := [x for x in ep_times if x is None]:
        logger.warning(f"{len(failed)} runs failed out of {n_runs}!")
    else:
        logger.info("All runs completed successfully!")

    # Abort if all runs failed
    if len(failed) > n_runs / 2:
        logger.error("More than 50% of all runs failed! Aborting submission.")
        raise RuntimeError("Too many runs failed!")

    ep_times = [x for x in ep_times if x is not None]
    data = {"ID": [i for i in range(len(ep_times))], "submission_time": ep_times}
    pd.DataFrame(data).to_csv("submission.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()