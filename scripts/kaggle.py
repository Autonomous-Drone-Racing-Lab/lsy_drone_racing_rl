"""Kaggle competition auto-submission script.

Note:
    Please do not alter this script or ask the course supervisors first!
"""

import logging

import pandas as pd
from sim import simulate

logger = logging.getLogger(__name__)


def main():
    """Run the simulation N times and save the results as 'submission.csv'."""
    n_runs = 50
    config: str = "config/level3_extra.yaml"
    checkpoint: str = "logs/next_gate_rp_relative_vel_more_disturb_1/best_model.zip"
    controller: str = "lsy_drone_racing/controller/rl_controller.py"
    
    ep_times = simulate(config=config, controller=controller, checkpoint=checkpoint, n_runs=n_runs, gui=True)
    # Log the number of failed runs if any

    if failed := [x for x in ep_times if x is None]:
        logger.warning(f"{len(failed)} runs failed out of {n_runs}!")
    else:
        logger.info("All runs completed successfully!")

    # Abort if all runs failed
    # if len(failed) > n_runs / 2:
    #     logger.error("More than 50% of all runs failed! Aborting submission.")
    #     raise RuntimeError("Too many runs failed!")

    no_failed = len(failed)
    failure_rate = no_failed / n_runs
    success_rate = 1 - failure_rate

    print(f"Complted runs: {n_runs - no_failed}. This is failure rate: {failure_rate} and success rate: {success_rate}")
    with open("submission.csv", "w") as f:
        f.write(f"Total runs: {n_runs}, runs_completed: {n_runs - no_failed}, runs_failed: {no_failed}, failure_rate: {failure_rate}, success_rate: {success_rate}\n")
    
    ep_times = [x for x in ep_times if x is not None]
    best_ep_time = min(ep_times)
    print(f"Best episode time: {best_ep_time}")
    data = {"ID": [i for i in range(len(ep_times))], "submission_time": ep_times}
    pd.DataFrame(data).to_csv("submission.csv", index=False, mode="a")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
