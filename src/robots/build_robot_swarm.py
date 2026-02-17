from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

import numpy as np
from robots.deployment_robot import WarehouseRobot
from robots.robot_modules.move_handler import LCM_MoveHandler, SimulationMoveHandler
from robots.robot_modules.state_handler import BasicStateHandler, LCM_StateHandler
from robots.robot_modules.swarm_communication_handler import (
    BasicCommunicationHandler,
    LCM_CommunicationHandler,
)
from robots.robot_swarm import RobotSwarm
from robots.sensors.lidar import Lidar
from robots.sensors.load_sensor import LoadSensor
from robots.sensors.position_monitor import SimulatedOptitrack, SimulatedPositionSensor

if TYPE_CHECKING:
    from robots.robot import Robot
    from robots.sensors.robot_simulation import RobotSimulation


def build_random_robot_swarm(
    n_robots: int,
    build_robot_fn: Callable,
    simulation_type: Callable[..., RobotSimulation] | None,
    ts_communicate: float,
    swarm_type: Callable[..., RobotSwarm] = RobotSwarm,
    max_vel: float = 0.4,  # in m/s
    ts_control: float = 0.2,
    n_anomal_robots=0,
    anomal_types: List = [],
    p_anomal: List | None = None,
    communication_ids=None,
    **kwargs,
):

    sensors = []
    colors = [
        "cornflowerblue",
        "orange",
        "green",
        "red",
        "purple",
        "yellow",
        "pink",
        "brown",
        "gray",
        "olive",
        "cyan",
        "magenta",
        "lime",
    ]
    colors = colors * 3

    robot_types = [WarehouseRobot] * (n_robots - n_anomal_robots)
    if n_anomal_robots > 0:
        if len(anomal_types) == 0:
            print("please speficy anomaly types")
        else:
            robot_types += list(
                np.random.choice(
                    np.array(anomal_types),
                    replace=True,
                    p=p_anomal,
                    size=n_anomal_robots,
                )
            )
    robots = {}
    ids = np.random.permutation(np.arange(n_robots))
    for i, rtype in zip(ids, robot_types):
        new_robot, new_sensor = build_robot_fn(
            rtype,
            robot_id=i,
            color=colors[i],
            simulation_type=simulation_type,
            ts_communicate=ts_communicate,
            ts_control=ts_control,
            max_vel=max_vel,
            communication_ids=communication_ids,
            **kwargs,
        )
        robots[i] = new_robot
        sensors.append(new_sensor)

    swarm_robots = [robots[key] for key in np.arange(n_robots)]
    robot_swarm = swarm_type(
        swarm_robots,
        ts_communicate=ts_communicate,
    )

    return robot_swarm, sensors


def build_simulation_robot(
    rtype: Callable[..., Robot],
    simulation_type: Callable[..., RobotSimulation],
    robot_id: int,
    color: str,
    ts_communicate: float,
    ts_control: float,
    max_vel: float,
    wait_for_ts_communicate: bool = True,
    **kwargs,
):
    robot_simulation = simulation_type(robot_id, use_lcm=False, ts_control=ts_control)
    position_sensor = SimulatedPositionSensor(robot_id, robot_simulation, ts_control)
    load_sensor = LoadSensor(robot_simulation)
    lidar_sensor = Lidar(
        lidar_range=max_vel * 5,
        sector_size_in_deg=5,
        ts_control=ts_control,
        max_vel=max_vel,
    )
    state_handler = BasicStateHandler(position_sensor, ts_control)
    swarm_communication_handler = BasicCommunicationHandler(
        robot_id, ts_communicate, position_sensor
    )
    move_handler = SimulationMoveHandler(robot_simulation, ts_control)

    robot = rtype(
        robot_id=robot_id,
        color=color,
        state_handler=state_handler,
        move_handler=move_handler,
        swarm_communication_handler=swarm_communication_handler,
        load_sensor=load_sensor,
        lidar_sensor=lidar_sensor,
        ts_communicate=ts_communicate,
        ts_control=ts_control,
        max_vel=max_vel,
        wait_for_ts_communicate=wait_for_ts_communicate,
        **kwargs,
    )
    return robot, position_sensor


def build_lcm_simulation_robot(
    rtype: Callable,
    simulation_type: Callable[..., RobotSimulation],
    robot_id: int,
    color: str,
    ts_communicate: float,
    ts_control: float,
    max_vel: float,
    **kwargs,
):
    robot_simulation = simulation_type(robot_id, use_lcm=True, ts_control=ts_control)
    position_sensor = SimulatedOptitrack(robot_id, robot_simulation)
    load_sensor = LoadSensor(robot_simulation)
    lidar_sensor = Lidar(
        lidar_range=max_vel * 5,
        sector_size_in_deg=5,
        ts_control=ts_control,
        max_vel=max_vel,
    )
    state_handler = LCM_StateHandler(communication_id=robot_id, ts_control=ts_control)
    swarm_communication_handler = LCM_CommunicationHandler(robot_id, ts_communicate)
    move_handler = LCM_MoveHandler(communication_id=robot_id, ts_control=ts_control)

    robot = rtype(
        robot_id=robot_id,
        color=color,
        state_handler=state_handler,
        move_handler=move_handler,
        swarm_communication_handler=swarm_communication_handler,
        load_sensor=load_sensor,
        lidar_sensor=lidar_sensor,
        ts_communicate=ts_communicate,
        ts_control=ts_control,
        max_vel=max_vel,
        wait_for_ts_communicate=True,
        **kwargs,
    )
    return robot, position_sensor


def build_lcm_robot(
    rtype: Callable,
    robot_id: int,
    color: str,
    ts_communicate: float,
    ts_control: float,
    communication_ids,
    **kwargs,
):
    state_handler = LCM_StateHandler(
        communication_id=communication_ids[robot_id], ts_control=ts_control, ttl=1
    )
    swarm_communication_handler = LCM_CommunicationHandler(
        robot_id, ts_communicate, ttl=1
    )
    move_handler = LCM_MoveHandler(
        communication_id=communication_ids[robot_id], ts_control=ts_control, ttl=1
    )

    robot = rtype(
        robot_id=robot_id,
        color=color,
        state_handler=state_handler,
        swarm_communication_handler=swarm_communication_handler,
        load_sensor=None,
        move_handler=move_handler,
        ts_communicate=ts_communicate,
        ts_control=ts_control,
        wait_for_ts_communicate=True,
        **kwargs,
    )
    return robot, None
