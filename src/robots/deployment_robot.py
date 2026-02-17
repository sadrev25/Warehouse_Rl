from __future__ import annotations

import abc
import copy
import logging
import multiprocessing
import multiprocessing.synchronize
import time
from multiprocessing import Manager
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING, List

import networkx as nx
import numpy as np
from helper_functions import l2_norm
from robots.robot import Robot
from robots.robot_modules.move_handler import MoveHandler, SimulationMoveHandler
from robots.robot_modules.state_handler import StateHandler
from robots.robot_modules.swarm_communication_handler import SwarmCommunicationHandler
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal, weibull_min
from scipy.stats._multivariate import multivariate_normal_frozen
from warehouse_env.path import Path
from warehouse_env.task import Task
from warehouse_env.warehouse import Warehouse

if TYPE_CHECKING:
    from robots.robot_modules.state_handler import RobotState, StateHandler
    from robots.robot_modules.swarm_communication_handler import (
        SwarmCommunicationHandler,
    )
    from robots.robot_swarm import RobotSwarm
    from robots.sensors.lidar import Lidar
    from robots.sensors.load_sensor import LoadSensor


class WarehouseRobot(Robot):

    def __init__(
        self,
        robot_id: int,
        color: str,
        state_handler: StateHandler,
        move_handler: MoveHandler,
        swarm_communication_handler: SwarmCommunicationHandler,
        load_sensor: LoadSensor,
        lidar_sensor: Lidar,
        ts_control: float,
        ts_communicate: float,
        max_vel: float,
        wait_for_ts_communicate: bool,
        **kwargs,
    ) -> None:
        """Robot solving a deployment task.

        Parameters
        ----------
        robot_id : int
            robot id
        state_handler : StateHandler
            Collects state information, e.g. the current position.
        move_handler : MoveHandler
            Triggers the movement of the robot.
        swarm_communication_handler : SwarmCommunicationHandler
            Used to communicate with the robot swarm.
        ts_control : float
            Every ts_control seconds the target velocity is updated based on the current position.
        max_vel : float
            Maximum velocity in metres per second that the robot can achieve.
        wait_for_ts_communicate : bool
            If True, the robot waits for the duration of ts_control before triggering the next movement.
        """

        super().__init__(
            robot_id,
            color=color,
            state_handler=state_handler,
            move_handler=move_handler,
            swarm_communication_handler=swarm_communication_handler,
            ts_communicate=ts_communicate,
            ts_control=ts_control,
            max_vel=max_vel,
        )
        self.load_sensor = load_sensor
        self.state_handler.set_load_sensor(load_sensor)
        self.lidar_sensor = lidar_sensor
        self.state_handler.set_lidar_sensor(lidar_sensor)

        self.wait_for_ts_communicate = wait_for_ts_communicate
        self.ts_update_positions = (
            self.ts_communicate - 0.2
        )  # total iteration time minus the approx. time required for computations related to, e.g., the path, collisions

        self.max_dist_to_target = 0.08  # in m
        self.fail_safe_trigger = 2 * self.max_vel * self.ts_control
        self.collision_ts_counter = 0
        self.collision_ts_max = 1 * (
            self.ts_communicate / self.ts_control
        )  # 1.5 * ts_communicate seconds
        self.was_colliding = False
        self.motion_deviation_threshold = np.deg2rad(5)

        self.lookahead_dist_waypoints = 15 * self.max_vel
        assert self.lookahead_dist_waypoints > self.max_vel, print(
            "Please increase the lookahead distance to make sure that the robot has time to stop before a crossing!"
        )
        self.crossing_buffer_time = 5  # in seconds

        self.vel_per_ts_comm = self.max_vel * self.ts_update_positions
        self.adapted_vel = self.max_vel  # holds individually for each axis of the robot

        # selected such that the robot maintains the maximum velocity per ts_communicate. Additionally, it has to be large enough to yield a suitable next target on the path even if the robot deviates from the path due to obstacles
        self.segment_length = 2 * self.vel_per_ts_comm

        state_mgr = Manager()
        self.state_history = state_mgr.list([])
        self._state_lock = state_mgr.Lock()

    def execute_warehouse_task(
        self,
        task: Task,
        robot_swarm: RobotSwarm,
        stop_execution: Event | None,
    ) -> int:

        assert type(self.deployment_area) is Warehouse
        shortest_path_to_current_station, shortest_path_current_to_target_station = (
            self.plan_path(task)
        )

        # move to current station of the object
        pos_current_station = shortest_path_to_current_station.final_target_pos
        while not self._is_target_position_reached(pos_current_station) and (
            stop_execution is None or not stop_execution.is_set()
        ):

            # t0 = time.time()
            swarm_state = self.update_swarm_state(
                robot_swarm=robot_swarm,
                task=task,
                path=shortest_path_to_current_station,
                target_pos=pos_current_station,
            )
            if swarm_state[self.id].time_stamp > 1000 and stop_execution is not None:
                stop_execution.set()
                logging.warning(f"max time exceeded")

            self.update_path_info(
                swarm_info=swarm_state,
                path=shortest_path_to_current_station,
            )
            self.update_target_position(
                path=shortest_path_to_current_station,
            )

        self.manage_load("add")
        # print(self.id, " add load")

        # move to target station of the object
        pos_target_station = shortest_path_current_to_target_station.final_target_pos
        while not self._is_target_position_reached(pos_target_station) and (
            stop_execution is None or not stop_execution.is_set()
        ):

            # t0 = time.time()
            swarm_state = self.update_swarm_state(
                robot_swarm=robot_swarm,
                task=task,
                path=shortest_path_current_to_target_station,
                target_pos=pos_target_station,
            )
            if swarm_state[self.id].time_stamp > 1000 and stop_execution is not None:
                stop_execution.set()
                logging.warning(f"max time exceeded")

            self.update_path_info(
                swarm_info=swarm_state,
                path=shortest_path_current_to_target_station,
            )
            self.update_target_position(
                path=shortest_path_current_to_target_station,
            )
            # print(self.id, " ", np.round(time.time() - t0, decimals=1))

        self.manage_load("remove")
        # print(self.id, " remove load")
        return self.id

    def update_swarm_state(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        robot_swarm: RobotSwarm,
        task: Task,
        path: Path,
        target_pos: np.ndarray,
    ) -> dict:

        next_waypoints, distance_to_waypoints = path.get_next_path_waypoints(
            max_dist=self.lookahead_dist_waypoints,
            current_pos=self._get_current_position(),
        )

        state = self.state_handler.get_current_state()
        state.set_priority(task_priority=task.priority)
        state.set_final_target_position(final_target_position=target_pos)
        state.set_path_info(next_waypoints, distance_to_waypoints, max_vel=self.max_vel)

        self.swarm_communication_handler.send_info_to_swarm(
            own_state=state, robot_swarm=robot_swarm
        )

        swarm_state = self.swarm_communication_handler.gather_swarm_info(
            robot_swarm=robot_swarm
        )

        return swarm_state

    def update_path_info(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        swarm_info: dict,
        path: Path,
        **args,
    ) -> None:

        # - check if any robot with higher prio within estimated arrival time to waypoint +- puffer time
        # if no, continue
        # if yes,
        #     (- possibly recompute path)
        #     - keep the previous target position
        if self.are_next_crossings_available(swarm_info):
            self.adapted_vel = self.max_vel
        elif self.adapted_vel > 0.02:  # maintain minimum velocity to prevent deadlocks
            self.adapted_vel *= 0.8

        if (
            l2_norm(path.final_target_pos - self._get_current_position())
            <= 3 * self.vel_per_ts_comm
        ):
            # print("close to target")
            self.adapted_vel = min(self.adapted_vel, self.max_vel / 2)

        return

    def update_target_position(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, path: Path
    ) -> None:
        """
        Executes the robot actions during one communication timestep ts_communicate. Based on the current target position and the information of the lidar sensor, a motion_vector is determined and the motion of the robot is triggered.
        """
        # move for the duration of the control timestep ts_control before adapting the velocity
        n_iter = int(self.ts_update_positions / self.ts_control)
        for i in range(n_iter):

            rcp = self._get_current_position()
            self.current_target = path.get_next_path_target(
                current_pos=rcp,
                segment_length=self.segment_length,
            )
            current_target_vector = self.current_target - rcp

            own_state = self.state_handler.get_current_state()
            own_state.set_current_target_position(self.current_target)
            # if self.id == 12:
            #     print(own_state.time_stamp)

            (
                robot_positions_in_range,
                motion_info_robots_in_range,
                shelf_positions_in_range,
                shelf_geometries_in_range,
            ) = self.lidar_sensor.get_lidar_info(
                rcp=rcp,
                robot_target=self.current_target,
                warehouse=self.deployment_area,
                path=path,
            )
            own_state.set_lidar_info(shelf_positions_in_range, robot_positions_in_range)

            robot_geometries = []
            # faster robot moving in similar direction: ignore for collision avoidance
            for robot_info in motion_info_robots_in_range:
                if robot_info[
                    "motion_deviation"
                ] < self.motion_deviation_threshold and robot_info["speed"] >= l2_norm(
                    own_state.velocity
                ):
                    continue
                else:
                    robot_geometries.append(robot_info["geometry"])

            # fail safe collision avoidance assuming that the robot cannot change its direction of motion fast enough
            is_colliding, collision_distance = self.lidar_sensor.check_for_collision(
                rcp=rcp,
                target_pos=rcp + own_state.velocity,
                object_geometries=robot_geometries + shelf_geometries_in_range,
            )

            trigger_waiting_to_avoid_collision = self.collision_ts_counter > 0
            stop_waiting_to_avoid_collision = (
                self.collision_ts_counter > self.collision_ts_max
            )

            if (
                is_colliding
                or collision_distance < self.fail_safe_trigger
                or trigger_waiting_to_avoid_collision
            ) and not stop_waiting_to_avoid_collision:
                if is_colliding and self.collision_ts_counter == 0:
                    logging.warning(
                        f"robot {self.id} collision at {own_state.time_stamp}"
                    )
                    self.was_colliding = True

                motion_vector = np.zeros_like(current_target_vector)
                self.collision_ts_counter += 1
            else:
                motion_vector = self.lidar_sensor.adapt_target_to_lidar_info(
                    rcp=rcp,
                    target_vector=current_target_vector,
                    object_geometries_in_range=robot_geometries
                    + shelf_geometries_in_range,
                    velocity=own_state.velocity,
                    # id=self.id,
                )
                if (
                    not (is_colliding or collision_distance < self.fail_safe_trigger)
                    and stop_waiting_to_avoid_collision
                ):
                    if self.was_colliding:
                        logging.warning(
                            f"robot {self.id} collision resolved at {own_state.time_stamp}"
                        )
                        self.was_colliding = False
                    self.collision_ts_counter = 0

            scaled_motion_vector = self.trigger_movement(
                motion_vector, adapted_vel=self.adapted_vel
            )

            own_state.set_adapted_target_position(rcp + scaled_motion_vector)
            with self._state_lock:
                self.state_history += [own_state]

            if self.wait_for_ts_communicate:
                time.sleep(self.ts_control)

        return

    def plan_path(self, task: Task) -> tuple[Path, Path]:
        graph = self.deployment_area.graph

        # find access points for reaching and leaving the current station of the object
        accessible_edges = graph.nodes(data=True)[task.current_station][
            "object"
        ].accessible_edges
        access_object_nodes = [n1 for n1, _ in accessible_edges]
        start_task_nodes = [n2 for _, n2 in accessible_edges]

        # find access points for reaching the target station of the object
        accessible_edges = graph.nodes(data=True)[task.target_station][
            "object"
        ].accessible_edges
        complete_task_nodes = [n1 for n1, _ in accessible_edges]

        current_pos = self._get_current_position()
        # check all possible access points to the current station of the object
        shortest_path_to_current_station = self.compute_shortest_path(
            start_node=self.deployment_area.get_node_from_pos(current_pos),
            start_waypoints=self.deployment_area.get_waypoints_from_pos(current_pos),
            target_waypoints=access_object_nodes,  # type: ignore
            target_station=task.current_station,
            graph=graph,
            task=task,
        )
        # find shortest path between start and target stations of the task
        shortest_path_current_to_target_station = self.compute_shortest_path(
            start_node=task.current_station,
            start_waypoints=start_task_nodes,  # type: ignore
            target_waypoints=complete_task_nodes,  # type: ignore
            target_station=task.target_station,
            graph=graph,
            task=task,
        )
        return (
            shortest_path_to_current_station,
            shortest_path_current_to_target_station,
        )

    def compute_shortest_path(
        self,
        start_node: str,
        start_waypoints: list,
        target_waypoints: list,
        target_station: str,
        graph: nx.Graph,
        task: Task,
    ) -> Path:

        shortest_path = Path()

        for tnode in target_waypoints:
            try:
                plen, nodes = nx.multi_source_dijkstra(
                    graph, start_waypoints, tnode, weight="eucl_dist"
                )
                assert type(plen) is float or type(plen) is int
                if plen < shortest_path.length:
                    shortest_path = Path(plen, nodes)
            except:
                print(
                    self.id,
                    "task: ",
                    task.__dict__,
                    "\nnodes: ",
                    start_node,
                    start_waypoints,
                    target_waypoints,
                    target_station,
                    tnode,
                )

        shortest_path.set_waypoint_coords(
            waypoint_coords=np.array(
                [
                    self.deployment_area.get_coords(node)
                    for node in shortest_path.waypoints
                ]
            ),
            start_node=start_node,
            start_pos_of_path=self.deployment_area.get_coords(start_node),
            final_target_of_path=target_station,
            final_target_pos_of_path=self.deployment_area.get_coords(target_station),
        )

        return shortest_path

    def are_next_crossings_available(self, swarm_info: dict):
        # check distance to waypoint/node
        # check if any incoming lanes
        # check communication of other robots
        # if robot reaches crossing at a similar time (TODO: how to determine this? communicate?): check load, velocity (if it is waiting anyways) and task priority (if all is equal, the closer robot wins)
        # slow down if no priority of way: set target to waiting point until other robot has passed, reduce velocity to avoid braking?

        own_state = self.state_handler.get_current_state()

        for robot_id, robot_state in swarm_info.items():
            if robot_id == self.id or robot_state is None:
                continue
            elif own_state.has_load and not robot_state.has_load:
                continue
            elif (
                own_state.priority > robot_state.priority
                and own_state.has_load == robot_state.has_load
            ):
                continue
            elif all(
                w1 == w2
                for w1, w2 in zip(robot_state.next_waypoints, own_state.next_waypoints)
            ):
                continue
            else:
                for w0, t0 in zip(
                    robot_state.next_waypoints, robot_state.estimated_time_to_waypoints
                ):
                    for w1, t1 in zip(
                        own_state.next_waypoints, own_state.estimated_time_to_waypoints
                    ):
                        if w0 == w1 and np.abs(t0 - t1) < self.crossing_buffer_time:
                            return False
        return True

    def _is_target_position_reached(self, final_target_pos: np.ndarray) -> bool:
        distance_to_final_target = l2_norm(
            final_target_pos - self._get_current_position()
        )

        if distance_to_final_target < self.max_dist_to_target:
            return True
        else:
            return False

    def manage_load(self, activity: str) -> None:
        if self.load_sensor is None:
            return
        match activity:
            case "add":
                self.load_sensor.add_load()
            case "remove":
                self.load_sensor.remove_load()
        return

    def set_deployment_area(self, area: object) -> None:
        assert type(area) is Warehouse
        self.deployment_area = area
        return


class AntagonisticWarehouseRobot(WarehouseRobot):

    def __init__(
        self,
        robot_id,
        color,
        state_handler,
        move_handler,
        swarm_communication_handler,
        load_sensor,
        lidar_sensor,
        ts_control,
        ts_communicate,
        max_vel,
        wait_for_ts_communicate,
        **kwargs,
    ):
        super().__init__(
            robot_id,
            color,
            state_handler,
            move_handler,
            swarm_communication_handler,
            load_sensor,
            lidar_sensor,
            ts_control,
            ts_communicate,
            max_vel,
            wait_for_ts_communicate,
            **kwargs,
        )

    def choose_antagonistic_behavior(
        self, swarm_info: dict, lidar_info: list, own_state: RobotState
    ) -> dict:
        """
        - embed previous state
            - use both own communicated and own correct state?
            - include lidar information
        - predict antagonistic behavior:
        [action, speed, load, priority]
        - convert 3d action to 2d action
        """
        return {}

    def update_swarm_state(
        self,
        robot_swarm: RobotSwarm,
        task: Task,
        path: Path,
        target_pos: np.ndarray,
    ) -> dict:
        """
        - get own state and most recent swarm state
        - predict antagonistic behavior
        - compute next state to communicate based on the selected behavior
        - communicate to swarm
        """

        swarm_state = self.swarm_communication_handler.gather_swarm_info(
            robot_swarm=robot_swarm
        )
        # TODO: move partially to state handler, only pass high level info
        # i.e., state handler has access to all sensors
        # use state handler to save states
        correct_state = self.state_handler.get_current_state()

        correct_state.set_priority(task_priority=task.priority)
        correct_state.set_final_target_position(final_target_position=target_pos)
        correct_state.set_load_info(
            self.load_sensor.is_carrying_load() if self.load_sensor else False
        )
        next_waypoints, distance_to_waypoints = path.get_next_path_waypoints(
            max_dist=self.lookahead_dist_waypoints,
            current_pos=self._get_current_position(),
        )
        correct_state.set_path_info(
            next_waypoints, distance_to_waypoints, max_vel=self.max_vel
        )

        (
            _,
            object_positions_in_range,
            _,
            _,
        ) = self.lidar_sensor.get_lidar_info(
            rcp=self._get_current_position(),
            robot_target=self.current_target,
            warehouse=self.deployment_area,
            path=path,
        )

        antagonistic_behavior = self.choose_antagonistic_behavior(
            swarm_info=swarm_state,
            own_state=correct_state,
            lidar_info=object_positions_in_range,
        )

        antagonistic_state = RobotState()
        antagonistic_state.set_motion_info(
            position=correct_state.position,  # type: ignore
            velocity=antagonistic_behavior["speed"],
        )
        antagonistic_state.set_time_stamp(
            time_stamp=correct_state.time_stamp,
        )
        antagonistic_state.set_priority(task_priority=antagonistic_behavior["priority"])
        antagonistic_state.set_load_info(antagonistic_behavior["load"])
        self.next_action = antagonistic_behavior["action"]

        antagonistic_state.set_final_target_position(final_target_position=target_pos)
        antagonistic_state.set_path_info(
            next_waypoints, distance_to_waypoints, max_vel=self.max_vel
        )

        self.swarm_communication_handler.send_info_to_swarm(
            own_state=antagonistic_state, robot_swarm=robot_swarm
        )

        return swarm_state

    def update_target_position(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        **args,
    ) -> None:

        self.current_target = self._get_current_position() + self.next_action
        return

    def update_position(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, swarm_info: dict, path: Path
    ) -> None:
        """
        Executes the robot actions during one communication timestep ts_communicate. Based on the current target position and the information of the lidar sensor, a motion_vector is determined and the motion of the robot is triggered.
        """
        own_state = swarm_info[self.id]
        # move for the duration of the control timestep ts_control before adapting the velocity
        n_iter = int(self.ts_communicate / self.ts_control)
        for i in range(n_iter):

            (
                object_geometries_in_range,
                object_positions_in_range,
                _,
                _,
            ) = self.lidar_sensor.get_lidar_info(
                rcp=self._get_current_position(),
                robot_target=self.current_target,
                warehouse=self.deployment_area,
                path=path,
            )
            own_state.set_lidar_info(object_positions_in_range)
            with self._state_lock:
                self.state_history += [own_state]

            current_motion_vector = self.current_target - self._get_current_position()

            # fail safe collision avoidance assuming that the robot cannot change its direction fast enough
            is_colliding, collision_distance = self.lidar_sensor.check_for_collision(
                rcp=self._get_current_position(),
                target_pos=self.current_target,
                object_geometries=object_geometries_in_range,
            )
            if (
                is_colliding or collision_distance < self.fail_safe_trigger
            ) and own_state.speed > 1e-2:
                current_motion_vector = np.zeros_like(current_motion_vector)

            self.trigger_movement(current_motion_vector, self.adapted_vel)

            if self.wait_for_ts_communicate:
                time.sleep(self.ts_control)

    def stop(self) -> None:
        super().stop()
        # TODO: add run to replay buffer
