from __future__ import annotations

import abc
import asyncio
import logging
import time
from multiprocessing import Array, Manager, Process, Queue
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING, List, Sequence

import numpy as np
from robots.robot_modules.state_handler import RobotState
from robots.robot_modules.swarm_communication_handler import CommunicatedRobotState
from visualization.swarm_visualization import SwarmVisualization
from warehouse_env.task import Task
from warehouse_env.warehouse import Warehouse

from learning_antagonistic_behavior.robots.sensors.position_monitor import (
    PositionMonitor,
)

if TYPE_CHECKING:
    from anomaly_detectors.anomaly_detector import AnomalyDetector
    from robots.deployment_robot import WarehouseRobot
    from robots.robot import Robot
    from robots.sensors.position_monitor import PositionMonitor
    from src.deployment_area.polygon import PolygonWrapper


class RobotSwarm(object, metaclass=abc.ABCMeta):

    def __init__(
        self,
        swarm_robots: Sequence[Robot],
        ts_communicate: float,
        anomaly_detector: AnomalyDetector | None = None,
        **kwargs,
    ):
        """Create a robot swarm from a list of robots.

        Args:
            swarm_robots (list[Robot]): robot entities belonging to the swarm
        """
        self.swarm_robots = swarm_robots
        self.n_robots = len(swarm_robots)
        self.ts_communicate = ts_communicate

        state_mgr = Manager()
        self.communicated_states_history = state_mgr.dict(
            {robot.id: [] for robot in self.swarm_robots}
        )
        self.finished_tasks_info = state_mgr.list([])
        self.last_communicated_states = state_mgr.dict(
            {
                robot.id: CommunicatedRobotState(RobotState())
                for robot in self.swarm_robots
            }
        )
        self._state_lock = state_mgr.Lock()

        self.set_anomaly_detector(anomaly_detector)
        self.visualization_module = SwarmVisualization(self)  # type: ignore

    @abc.abstractmethod
    def run_swarm_task(
        self,
        **kwargs,
    ) -> None:
        raise NotImplementedError()

    def run_anomaly_detection(
        self, step: int, last_positions: np.ndarray, actions: np.ndarray
    ) -> None:

        if self.anomaly_detector is not None:

            self.anomaly_detector.evaluate_step(
                pos=last_positions,
                actions=actions,
                deployment_area=self.deployment_area,
                n_samples=20,
                mask_anomal=True,
            )
            print(
                f"action {step}:, anomaly recognized: {np.where(self.anomaly_detector.get_anomaly_prediction_of_swarm())[0].tolist()}"
            )

    def is_anomal(self, robot: Robot, time_stamp: float = -1) -> bool:
        """Check if a robot agent has been classified as anomal.

        Args:
            robot (Robot): robot agent

        Returns:
            bool: anomaly prediction
        """
        if (
            self.anomaly_detector is None
            or self.anomaly_detector.detection_method is None
        ):
            return False
        else:

            return self.anomaly_detector.is_robot_anomal(
                robot=robot, time_stamp=time_stamp
            )

    def get_all_communicated_positions(self) -> np.ndarray:

        all_communicated_positions = np.array(
            [
                [state.position for state in self.communicated_states_history[robot.id]]
                for robot in self.swarm_robots
            ],
            dtype=np.ndarray,
        )  # shape: (n_robots, n positions, 2)
        # ).transpose(1, 0, 2)
        return all_communicated_positions

    def get_last_action(self) -> np.ndarray:
        """Get the last actions of the swarm robots.

        Returns:
            np.ndarray: last actions
        """
        return np.array(
            [
                robot.get_communicated_position()
                - self.communicated_states_history[robot.id][-1].position
                for robot in self.swarm_robots
            ]
        )

    def add_communicated_state(self, robot_id, state: CommunicatedRobotState) -> None:
        with self._state_lock:
            self.last_communicated_states[robot_id] = state
        self.communicated_states_history[robot_id] += [state]
        return

    def get_last_communicated_states(self) -> dict:
        with self._state_lock:
            states = dict(self.last_communicated_states)
        return states

    def get_vel_history(
        self,
    ) -> list:
        """Get the swarm robots' motions.

        Returns
        -------
        list
            Motion history per swarm robot.
        """
        return [robot.get_vel_history() for robot in self.swarm_robots]

    def set_area_bounds(self, **kwargs) -> None:
        """Define the area that the robots are allowed to move in. In case of the deployment problem, this area corresponds to the area that the robot swarm will cover.

        Parameters
        ----------
        deployment_area : PolygonWrapper

        """
        """Set the area that will be covered by the robot swarm.

        Args:
            robot_area (PolygonWrapper): 
                Area that the .
            boundary (PolygonWrapper): Convex coverage area.
        """
        area = kwargs.get("deployment_area", None)
        self.deployment_area = area
        for robot in self.swarm_robots:
            robot.set_deployment_area(area)
        return

    def set_anomaly_detector(self, anomaly_detector: AnomalyDetector | None) -> None:
        """Set the anomaly detector and pass information about the number of robots in the swarm.

        Args:
            anomaly_detector (AnomalyDetector): Contains a model that has been trained to detect anomalies.
        """
        self.anomaly_detector = anomaly_detector
        if self.anomaly_detector is not None:
            self.anomaly_detector.initialize_run_prediction(self)  # type: ignore
            for robot in self.swarm_robots:
                robot.set_anomaly_detector(self.anomaly_detector)
        return

    def start_run(
        self,
        deployment_area: object,
        external_state_monitors: list[PositionMonitor] | None = None,
        **kwargs,
    ) -> None:
        """Start the handler modules/threads of each robot and the communication/monitoring threads of external state monitors.

        Args:
            external_state_monitors (list[StateMonitor]): external state monitors that are not part of the robot swarm
        """
        self.set_area_bounds(deployment_area=deployment_area)
        self.external_state_monitors = external_state_monitors

        if self.external_state_monitors is not None:
            for sm in self.external_state_monitors:
                sm.start(deployment_area, self.n_robots)
        for robot in self.swarm_robots:
            robot.start(self)

        return

    def stop_run(
        self, external_state_monitors: list[PositionMonitor] | None = None
    ) -> None:
        """Stop the handler modules/threads of each robot and the communication/monitoring threads of external state monitors.

        Args:
            external_state_monitors (list[StateMonitor]): external state monitors that are not part of the robot swarm
        """
        for robot in self.swarm_robots:
            robot.stop()
        if external_state_monitors is not None:
            for sm in external_state_monitors:
                sm.stop()

        return

    def run_info(self, print_run: bool = True) -> None:

        # positions = {r.id: r._get_current_position() for r in self.swarm_robots}
        # if print_run:
        #     print(f"Number of robots: {self.n_robots}, number of steps: {self.step}")
        # for robot in self.swarm_robots:
        #     if print_run:
        #         print(
        #             f"Success of {robot.__class__.__name__} with final position {robot.get_communicated_position()}: {'number of tasks placeholder?'}"
        #         )
        return


class WarehouseSwarm(RobotSwarm):

    def __init__(
        self,
        swarm_robots: Sequence[WarehouseRobot],
        ts_communicate: float,
        anomaly_detector: AnomalyDetector | None = None,
        **kwargs,
    ):
        super().__init__(swarm_robots, ts_communicate, anomaly_detector, **kwargs)
        self.swarm_robots: Sequence[WarehouseRobot] = swarm_robots
        self.step = 0

    def run_swarm_task(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        max_tasks: int,
        seed: int,
        **kwargs,
    ) -> None:
        """
        Args:
            max_steps (int, optional): Maximum number of steps that will be performed.
        """
        np.random.seed(seed)
        robot_mgr = Manager()
        available_robots = robot_mgr.Queue()
        stop_event = robot_mgr.Event()
        for robot in self.swarm_robots:
            available_robots.put(robot.id)

        tasks = [self.request_task(task_number) for task_number in range(max_tasks)]
        started_tasks = []

        for task in tasks:
            robot_id = available_robots.get()

            task_proc = Process(
                target=self.assign_task_to_robot,
                args=(robot_id, task, available_robots, stop_event, seed),
            )
            task_proc.start()
            started_tasks.append(task_proc)

            # actions = self.get_last_action()
            # if self.anomaly_detector is not None:
            #     self.run_anomaly_detection(
            #         step=self.step,
            #         last_positions=positions,
            #         actions=actions,
            #     )

        robot_id = available_robots.get()  # wait until the next robot finishes
        stop_event.set()
        self.run_duration = np.round(time.time() - self.start_time, decimals=2)

        for task_proc in started_tasks:
            task_proc.join()

        for task_proc in started_tasks:
            task_proc.close()

        logging.warning(
            f"{len(self.finished_tasks_info)} tasks done for {self.n_robots} robots in {self.run_duration} seconds"
        )

        return

    def request_task(self, task_number: int) -> Task:
        assert type(self.deployment_area) is Warehouse
        return self.deployment_area.get_next_task(task_number)

    def assign_task_to_robot(
        self,
        robot_id: int,
        task: Task,
        robot_queue: Queue,
        stop_event: Event,
        seed: int,
    ) -> None:
        # print(f"Task {task.task_number} assigned to robot {robot_id}")
        robot = self.swarm_robots[robot_id]
        time.sleep(0.05)
        print(
            f"robot {robot_id} starts task at {np.round(time.time() - self.start_time, decimals=2)}"
        )
        robot_id = robot.execute_warehouse_task(task, self, stop_event, seed)
        if not stop_event.is_set():
            self.finished_tasks_info.append(
                {
                    "ts": np.round(time.time() - self.start_time, decimals=2),
                    "id": robot_id,
                    "task": task.__dict__,
                }
            )

        # print("Task done by robot ", robot_id)
        robot_queue.put(robot_id)
        return

    def start_run(
        self,
        deployment_area: object,
        external_state_monitors: List[PositionMonitor] | None = None,
        **kwargs,
    ) -> None:
        self.start_time = time.time()
        super().start_run(deployment_area, external_state_monitors, **kwargs)
        for robot in self.swarm_robots:
            if external_state_monitors is not None:
                robot.lidar_sensor.start(
                    [
                        sm.simulated_robot
                        for sm in external_state_monitors
                        if sm.simulated_robot.id != robot.id
                    ]
                )
