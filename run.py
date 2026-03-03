import logging
import multiprocessing
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src.robots.build_robot_swarm import (
    build_random_robot_swarm,
    build_simulation_robot,
)
from src.robots.robot_swarm import WarehouseSwarm
from src.robots.save_and_load_swarm import save_robot_swarm
from src.robots.sensors.robot_simulation import Hera
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

    warehouse = Warehouse(graphml_path="data/warehouse/Warehouse_preprocessed.graphml")

    for run in range(config["n_runs"]):
        print("##############################################################")
        print(run)
        print("##############################################################")
        print("Run started at ", time.strftime("%H:%M:%S"))
        np.random.seed(run)
        n_robots = np.random.randint(config["n_min_robots"], config["n_max_robots"] + 1)

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

        swarm.run_swarm_task(max_tasks=int(n_robots * 2), seed=run)

        swarm.stop_run(sensors)
        time.sleep(1)

        save_robot_swarm(
            swarm,
            f"data/swarm{run}",
        )
        swarm.visualization_module.animate_run(
            folder_path=f"data/run_animations/swarm{run}",
            every_nth_img=10,
        )

        # os.chdir(f"data/example_run_animation/swarm{run}")
        # os.system("ffmpeg -i frame_%04d.png -vcodec mpeg4 -r 10 swarm.avi")
