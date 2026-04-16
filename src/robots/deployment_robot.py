from __future__ import annotations

import abc
import copy
import datetime
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
from robots.robot_modules.state_handler import RobotState, StateHandler
from robots.robot_modules.swarm_communication_handler import (
    CommunicatedRobotState,
    SwarmCommunicationHandler,
)
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal, weibull_min
from scipy.stats._multivariate import multivariate_normal_frozen
from warehouse_env.path import Path
from warehouse_env.task import Task
from warehouse_env.warehouse import Warehouse

if TYPE_CHECKING:
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

        self.control_iterations_per_planning_iteration = 5

        self.max_dist_to_target = 0.08  # in m
        self.fail_safe_trigger = 2 * self.max_vel * self.ts_control
        self.collision_ts_counter = 0
        self.collision_ts_max = 1 * (
            self.ts_communicate / self.ts_control
        )  # 1.5 * ts_communicate seconds
        self.was_colliding = False
        self.motion_deviation_threshold = np.deg2rad(5)

        # threshold distance to waypoints that will be communicated to the swarm
        self.lookahead_dist_waypoints = 20 * self.max_vel
        # threshold time to waypoints included in checking the right of way
        self.lookahead_time_right_of_way = 12
        self.crossing_buffer_time = 6  # in seconds

        self.vel_per_ts_comm = self.max_vel * self.ts_communicate
        self.vel_per_ts_control = self.max_vel * self.ts_control
        self.adapted_vel = self.max_vel  # holds individually for each axis of the robot

        # selected such that the robot maintains the maximum velocity per ts_communicate while moving on a straight segment between waypoints. Additionally, it has to be large enough to yield a suitable next target on the path even if the robot deviates from the path due to obstacles.
        self.target_dist_straight_segment = 2 * self.vel_per_ts_comm
        # When close to a waypoint/crossing, the distance to the next path target is reduced, instead using
        self.max_dist_to_next_waypoint = 4 * self.vel_per_ts_control

        state_mgr = Manager()
        self.state_history = state_mgr.list([])
        self._state_lock = state_mgr.Lock()

    def execute_warehouse_task(
        self,
        task: Task,
        robot_swarm: RobotSwarm,
        stop_execution: Event | None,
        seed: int,
    ) -> int:

        np.random.seed(seed)

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
            swarm_state = self.swarm_communication_handler.get_swarm_info()

            if swarm_state[self.id].time_stamp > 2000 and stop_execution is not None:
                stop_execution.set()
                logging.warning(f"max time exceeded")

            self.update_path_planning(
                swarm_info=swarm_state,
                path=shortest_path_to_current_station,
            )
            for _ in range(self.control_iterations_per_planning_iteration):
                self.update_position(path=shortest_path_to_current_station, task=task)

        self.manage_load("add")
        # print(self.id, " add load")

        # move to target station of the object
        pos_target_station = shortest_path_current_to_target_station.final_target_pos
        while not self._is_target_position_reached(pos_target_station) and (
            stop_execution is None or not stop_execution.is_set()
        ):

            # t0 = time.time()
            swarm_state = self.swarm_communication_handler.get_swarm_info()

            if swarm_state[self.id].time_stamp > 2000 and stop_execution is not None:
                stop_execution.set()
                logging.warning(f"max time exceeded")

            self.update_path_planning(
                swarm_info=swarm_state,
                path=shortest_path_current_to_target_station,
            )
            for _ in range(self.control_iterations_per_planning_iteration):
                self.update_position(
                    path=shortest_path_current_to_target_station, task=task
                )
            # print(self.id, " ", np.round(time.time() - t0, decimals=1))

        self.manage_load("remove")
        # print(self.id, " remove load")
        return self.id

    def update_path_planning(  # pyright: ignore[reportIncompatibleMethodOverride]
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
        if self.are_next_crossings_available(swarm_info, path):
            self.adapted_vel = self.max_vel
        else:
            # print(self.color, " wait ", self.state_handler.get_time_stamp())
            self.adapted_vel *= 0.5

        return

    def update_position(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, path: Path, task: Task
    ) -> None:
        """
        Executes the robot actions during one control timestep ts_control. Based on the current target position and the information of the lidar sensor, a motion_vector is determined and the motion of the robot is triggered.
        """
        rcp = self._get_current_position()
        rcv = self._get_current_velocity()
        self.current_target = path.get_next_path_target(
            current_pos=rcp,
            target_dist_straight_segment=self.target_dist_straight_segment,
            max_dist_waypoint=self.max_dist_to_next_waypoint,
        )
        current_target_vector = self.current_target - rcp

        motion_vector, robot_positions_in_range, shelf_positions_in_range = (
            self.adapt_motion_to_avoid_collisions(
                current_target_vector=current_target_vector, rcp=rcp, rcv=rcv, path=path
            )
        )

        scaled_motion_vector = self.trigger_movement(
            motion_vector, adapted_vel=self.adapted_vel
        )

        self.update_robot_state(
            task,
            path,
            shelf_positions_in_range,
            robot_positions_in_range,
            adapted_target_position=rcp + scaled_motion_vector,
        )

        time.sleep(self.ts_control)

        return

    def adapt_motion_to_avoid_collisions(
        self, current_target_vector, rcp, rcv, path
    ) -> tuple[np.ndarray, list, list]:

        (
            robot_positions_in_range,
            motion_info_robots_in_range,
            shelf_positions_in_range,
            shelf_geometries_in_range,
        ) = self.lidar_sensor.get_lidar_info(
            rcp=self._get_current_position(),
            robot_target=self.current_target,
            warehouse=self.deployment_area,
            path=path,
        )

        robot_geometries = []
        # faster robot moving in similar direction: ignore for collision avoidance
        for robot_info in motion_info_robots_in_range:
            if robot_info[
                "motion_deviation"
            ] < self.motion_deviation_threshold and robot_info["speed"] >= l2_norm(rcv):
                continue
            else:
                robot_geometries.append(robot_info["geometry"])

        # fail safe collision avoidance assuming that the robot cannot change its direction of motion fast enough
        is_colliding, collision_distance = self.lidar_sensor.check_for_collision(
            rcp=rcp,
            target_pos=rcp + rcv,
            object_geometries=robot_geometries + shelf_geometries_in_range,
        )

        trigger_waiting_to_avoid_collision = self.collision_ts_counter > 0
        stop_waiting_to_avoid_collision = (
            self.collision_ts_counter > self.collision_ts_max
        )

        # new collision
        if (
            is_colliding
            or collision_distance < self.fail_safe_trigger
            or trigger_waiting_to_avoid_collision
        ) and not stop_waiting_to_avoid_collision:
            if is_colliding and self.collision_ts_counter == 0:
                logging.warning(
                    f"robot {self.id} {self.color} collision at {self.state_handler.get_time_stamp()}"
                )
                self.was_colliding = True

            motion_vector = np.zeros_like(current_target_vector)
            self.collision_ts_counter += 1
        else:
            motion_vector = self.lidar_sensor.adapt_target_to_lidar_info(
                rcp=rcp,
                target_vector=current_target_vector,
                object_geometries_in_range=robot_geometries + shelf_geometries_in_range,
                velocity=rcv,
                # id=self.id,
            )
            if (
                not (is_colliding or collision_distance < self.fail_safe_trigger)
                and stop_waiting_to_avoid_collision
            ):
                if self.was_colliding:
                    logging.warning(
                        f"robot {self.id} {self.color} collision resolved at {self.state_handler.get_time_stamp()}"
                    )
                    self.was_colliding = False
                self.collision_ts_counter = 0

        return motion_vector, robot_positions_in_range, shelf_positions_in_range

    def update_robot_state(
        self,
        task: Task,
        path: Path,
        shelf_positions_in_range: list,
        robot_positions_in_range: list,
        adapted_target_position: np.ndarray,
    ) -> None:

        next_waypoints, time_to_next_waypoints = path.get_next_path_waypoints_info(
            max_dist=self.lookahead_dist_waypoints,
            current_pos=self._get_current_position(),
            max_vel=self.max_vel,
        )

        self.state_handler.update_task_info(
            task_priority=task.priority,
            final_target_position=path.final_target_pos,
            next_waypoints=next_waypoints,
            time_to_next_waypoints=time_to_next_waypoints,
        )

        self.state_handler.update_lidar_info(
            shelf_positions_in_range,
            robot_positions_in_range,
            self.current_target,
            adapted_target_position,
        )

        with self._state_lock:
            self.state_history += [self.state_handler.get_current_state()]
        return

    def are_next_crossings_available(
        self, swarm_info: dict[int, CommunicatedRobotState], path: Path
    ):
        # check swarm info for the waypoints and estimated time to waypoints
        # if a robot follows the same path/waypoints: continue as normal, passing the robot is allowed regardless of priority
        # if robot reaches crossing at a similar time: check load and task priority
        # check if a robot is in front (on the same edge and closer to the waypoint) or behind
        # slow down if no priority of way: reduce velocity

        own_state = self.state_handler.get_current_state()
        swarm_next_waypoints_info = [
            np.pad(
                np.hstack(
                    (
                        robot_state.next_waypoints,
                        robot_state.time_to_next_waypoints.reshape(-1, 1),
                    )
                ),
                ((0, 0), (0, 1)),
                "constant",
                constant_values=robot_id,
            )
            for robot_id, robot_state in swarm_info.items()
            if robot_id != self.id and robot_state.next_waypoints.size > 0
        ]
        if len(swarm_next_waypoints_info) == 0:
            return True

        tmp = np.vstack(swarm_next_waypoints_info)
        swarm_next_waypoints_info = np.rec.fromarrays(
            [
                tmp[:, 0].astype(str),
                tmp[:, 1].astype(str),
                tmp[:, 2].astype(float),
                tmp[:, 3].astype(int),
            ],
            dtype=[
                ("nodes", "U20"),
                ("edges", "U20"),
                ("est_time", np.float32),
                ("robot_id", int),
            ],
        )

        robots_with_right_of_way_solved = []

        for (node, edge), est_time in zip(
            own_state.next_waypoints, own_state.time_to_next_waypoints
        ):

            if not node in swarm_next_waypoints_info["nodes"]:
                continue
            if est_time > self.lookahead_time_right_of_way:
                continue

            next_node_crossings = swarm_next_waypoints_info[
                (swarm_next_waypoints_info["nodes"] == node)
                & (
                    np.abs(swarm_next_waypoints_info["est_time"] - est_time)
                    < self.crossing_buffer_time
                )
            ]
            if next_node_crossings.size == 0:
                continue

            for adj_edge in np.unique(next_node_crossings["edges"]):
                # find all robots not yet considered crossing the node from edge adj_edge
                next_crossings_from_edge = next_node_crossings[
                    (next_node_crossings["edges"] == adj_edge)
                    & (
                        ~np.isin(
                            next_node_crossings["robot_id"],
                            robots_with_right_of_way_solved,
                        )
                    )
                ]
                if next_crossings_from_edge.size == 0:
                    continue

                # find the smallest estimated time of the next crossing and the id of the robot that crosses it
                _, _, next_crossing_time, robot_id = next_crossings_from_edge[
                    np.argmin(next_crossings_from_edge["est_time"])
                ]
                # check if the robot accesses the node via the same edge and is in front -> wait until robot in front passes the crossing
                if adj_edge == edge and next_crossing_time < est_time:
                    return False

                lower_task_priority = own_state.priority < swarm_info[robot_id].priority
                lower_load_priority = (
                    swarm_info[robot_id].has_load and not own_state.has_load
                )
                higher_load_priority = (
                    own_state.has_load and not swarm_info[robot_id].has_load
                )
                # check if the robot crosses from another edge, but has a higher priority -> wait until the robot passes
                if adj_edge != edge:
                    if lower_load_priority:
                        return False
                    if lower_task_priority and not higher_load_priority:
                        return False

            robots_with_right_of_way_solved += [
                rid for rid in np.unique(next_node_crossings["robot_id"])
            ]

        # TODO: communicate more waypoints for path planning, but check only the next n waypoints for availability? or only up until certain time limit?
        #     for w0, t0 in zip(
        #         robot_state.next_waypoints, robot_state.estimated_time_to_waypoints
        #     ):
        #         for w1, t1 in zip(
        #             own_state.next_waypoints, own_state.estimated_time_to_waypoints
        #         ):
        #             if w0 == w1 and np.abs(t0 - t1) < self.crossing_buffer_time:
        #                 return False
        return True

    def _is_target_position_reached(self, final_target_pos: np.ndarray) -> bool:
        distance_to_final_target = l2_norm(
            final_target_pos - self._get_current_position()
        )

        if distance_to_final_target < self.max_dist_to_target:
            return True
        else:
            return False

    def plan_path(self, task: Task) -> tuple[Path, Path]:
        graph = self.deployment_area.graph
        current_pos = self._get_current_position()

        # find closest node, if not a waypoint, add access points of the station to the graph
        current_node = self.deployment_area.get_node_from_pos(current_pos)
        if graph.nodes(data=True)[current_node]["description"] != "waypoint":
            start_task_station_data = graph.nodes(data=True)[current_node]["object"]
            graph.add_nodes_from(start_task_station_data.access_nodes)
            graph.add_edges_from(start_task_station_data.access_edges)

        # add access points for reaching and leaving the station where the item will be fetched from, add them to the graph
        fetch_station_data = graph.nodes(data=True)[task.fetch_item_station]["object"]
        graph.add_nodes_from(fetch_station_data.access_nodes)
        graph.add_edges_from(fetch_station_data.access_edges)

        # get access points for reaching and leaving the station where the item will be dropped off, add them to the graph
        dropoff_station_data = graph.nodes(data=True)[task.drop_item_station]["object"]
        graph.add_nodes_from(dropoff_station_data.access_nodes)
        graph.add_edges_from(dropoff_station_data.access_edges)

        self.deployment_area.update_distances()

        # check all possible access points to the current station of the object
        shortest_path_current_to_fetch_station = self.compute_shortest_path(
            start_node=current_node,
            target_station=task.fetch_item_station,
            graph=graph,
        )
        # find shortest path between start and target stations of the task
        shortest_path_fetch_to_dropoff_station = self.compute_shortest_path(
            start_node=task.fetch_item_station,
            target_station=task.drop_item_station,
            graph=graph,
        )
        return (
            shortest_path_current_to_fetch_station,
            shortest_path_fetch_to_dropoff_station,
        )

    def compute_shortest_path(
        self,
        start_node: str,
        target_station: str,
        graph: nx.Graph,
    ) -> Path:

        plen, nodes = nx.single_source_dijkstra(
            graph, start_node, target_station, weight="eucl_dist"
        )
        assert type(plen) is float or type(plen) is np.float32
        shortest_path = Path(plen, nodes)
        assert type(nodes) == list and len(nodes) > 1
        edge_ids = [graph.edges[n1, n2]["id"] for n1, n2 in zip(nodes[:-1], nodes[1:])]

        shortest_path.set_edges(edge_ids)
        shortest_path.set_waypoint_coords(
            waypoint_coords=np.array(
                [
                    self.deployment_area.get_coords(node)
                    for node in shortest_path.waypoints
                ]
            )
        )

        return shortest_path

    def manage_load(self, activity: str) -> None:
        if self.load_sensor is None:
            return
        match activity:
            case "add":
                self.load_sensor.add_load()
            case "remove":
                self.load_sensor.remove_load()
        return

    def get_last_recorded_state(self) -> RobotState:
        if len(self.state_history) == 0:
            return RobotState()
        with self._state_lock:
            last_recorded_state = self.state_history[-1]
        return last_recorded_state

    def set_deployment_area(self, area: object) -> None:
        assert type(area) is Warehouse
        self.deployment_area = area
        return


class AntagonisticWarehouseRobot(WarehouseRobot):
    """
    A WarehouseRobot that broadcasts a manipulated state to the swarm.

    PPO sets these three attributes via the RL env before each episode:
        _fake_priority    (float 0-10)  — broadcasted task priority
        _fake_has_load    (bool)        — broadcasted load status
        _fake_time_offset (float, sec)  — shift added to est. waypoint arrival times
                                          use negative values to claim early arrival
                                          → forces other robots to yield at crossings

    Physical movement, lidar, and path planning are identical to WarehouseRobot.
    Only what gets broadcast to the swarm is manipulated.

    How the attack works:
        BasicCommunicationHandler._send_info_to_swarm() calls
        self.robot.get_last_recorded_state() every ts_communicate seconds
        and pushes the result into swarm.add_communicated_state().
        Every cooperative robot then reads this inside
        are_next_crossings_available() and decides whether to yield.
        Lying here causes real traffic disruption without any collision.
    """

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
        # PPO sets these before each episode. None = no manipulation.
        self._fake_priority    = None
        self._fake_has_load    = None
        self._fake_time_offset = 0.0

    def get_last_recorded_state(self) -> RobotState:
        real_state = super().get_last_recorded_state()
        if self._fake_priority is None and self._fake_has_load is None:
            return real_state
        if real_state.position is None:
            return real_state
        fake_state = RobotState()
        fake_state.set_time_stamp(time_stamp=real_state.time_stamp)
        fake_state.set_motion_info(position=real_state.position, velocity=real_state.velocity)
        fake_state.set_priority(task_priority=self._fake_priority if self._fake_priority is not None else real_state.priority)
        fake_state.set_load_info(self._fake_has_load if self._fake_has_load is not None else real_state.has_load)
        fake_state.set_final_target_position(final_target_position=getattr(real_state, 'final_target_position', np.zeros(2)))
        if real_state.next_waypoints is not None and real_state.next_waypoints.size > 0:
            safe_offset = np.clip(self._fake_time_offset, -3.0, 0.0)
            fake_time = np.clip(real_state.time_to_next_waypoints + safe_offset, a_min=0.0, a_max=None)
            fake_state.set_path_info(next_waypoints=real_state.next_waypoints, time_to_next_waypoints=fake_time)
        else:
            fake_state.set_path_info(next_waypoints=real_state.next_waypoints, time_to_next_waypoints=real_state.time_to_next_waypoints)
        return fake_state
