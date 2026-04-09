from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath("src"))

import logging
import multiprocessing
import time
import csv

import matplotlib.pyplot as plt
import numpy as np

from robots.build_robot_swarm import (
    build_random_robot_swarm,
    build_simulation_robot,
)

from robots.robot_swarm import WarehouseSwarm
from robots.save_and_load_swarm import save_robot_swarm
from robots.sensors.robot_simulation import Hera
from warehouse_env.warehouse import Warehouse


if __name__ == "__main__":

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.WARNING)

    config = {
        "n_runs": 10,
        "n_min_robots": 7,
        "n_max_robots": 17,
        "ts_communicate": 1.2,
        "ts_control": 0.2,
        "max_vel": 0.4,
        "wait_for_ts_communicate": True,
    }

    warehouse = Warehouse(
        graphml_path="data/warehouse/Warehouse_preprocessed.graphml"
    )

    # -------------------------------
    # Create CSV file for baseline
    # -------------------------------
    with open("baseline_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "n_robots", "tasks", "runtime", "throughput"])

    for run in range(config["n_runs"]):

        print("##############################################################")
        print("Run:", run)
        print("##############################################################")
        print("Run started at ", time.strftime("%H:%M:%S"))

        np.random.seed(run)

        n_robots = np.random.randint(
            config["n_min_robots"],
            config["n_max_robots"] + 1
        )

        swarm, sensors = build_random_robot_swarm(
            n_robots=n_robots,
            build_robot_fn=build_simulation_robot,
            simulation_type=Hera,
            swarm_type=WarehouseSwarm,
            ts_communicate=config["ts_communicate"],
            ts_control=config["ts_control"],
            max_vel=config["max_vel"],
            wait_for_ts_communicate=config["wait_for_ts_communicate"],
        )

        swarm.start_run(warehouse, sensors)
        time.sleep(1)

        # -------------------------------
        # START BASELINE TIMER
        # -------------------------------
        start_time = time.time()

        tasks = int(n_robots * 2)

        swarm.run_swarm_task(
            max_tasks=tasks,
            seed=run
        )

        # -------------------------------
        # END TIMER
        # -------------------------------
        end_time = time.time()

        runtime = end_time - start_time
        throughput = tasks / runtime

        print("Robots:", n_robots)
        print("Tasks:", tasks)
        print("Runtime:", runtime)
        print("Throughput:", throughput)

        swarm.stop_run(sensors)
        time.sleep(1)

        # -------------------------------
        # SAVE RESULTS
        # -------------------------------
        with open("baseline_results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                run,
                n_robots,
                tasks,
                runtime,
                throughput
            ])

        # -------------------------------
        # SAVE SWARM DATA
        # -------------------------------
        save_robot_swarm(
            swarm,
            f"data/swarm{run}",
        )

        # -------------------------------
        # CREATE ANIMATION
        # -------------------------------
        swarm.visualization_module.animate_run(
            folder_path=f"data/run_animations/swarm{run}",
            every_nth_img=10,
        )