import numpy as np
from helper_functions import l2_norm, signed_angle_2d, smallest_angle_2d
from robots.sensors.robot_simulation import RobotSimulation
from shapely import LineString, Point
from shapely.ops import nearest_points
from warehouse_env.path import Path
from warehouse_env.warehouse import Warehouse


class Lidar:

    def __init__(
        self,
        lidar_range: float = 2,
        sector_size_in_deg: float = 5,
        ts_control: float = 0.2,
        max_vel: float = 0.4,
        **kwargs,
    ) -> None:

        self.robot_radius = 0.29 / 2

        self.lidar_range = lidar_range
        self.sector_size = sector_size_in_deg
        self.ts_control = ts_control
        self.max_vel = max_vel

        self.target_cost_modifier = 1.2  # 3 * max_vel

    def start(self, simulated_robots: list[RobotSimulation]):
        self.simulated_robots = simulated_robots

    def adapt_target_to_lidar_info(
        self,
        rcp: np.ndarray,
        target_vector: np.ndarray,
        object_geometries_in_range: list,
        velocity: np.ndarray,
        # id,
    ):

        alpha = (self.sector_size / 360) * 2 * np.pi
        polar_sectors = np.arange(0, 2 * np.pi, alpha)
        # instead of grid cells with angle beta, use one 'beam' line for each alpha_k and check if there exists an intersection between the line and any obstacle
        beam_points = self.lidar_range * np.hstack(
            [np.cos(polar_sectors.reshape(-1, 1)), np.sin(polar_sectors.reshape(-1, 1))]
        )
        # cost of sectors according to their deviation from the target direction, [0, pi]
        target_cost_per_sector = smallest_angle_2d(target_vector, beam_points)

        motion_change_cost_per_sector = self.motion_change_cost_per_sector(
            velocity=velocity, polar_sectors=polar_sectors, beam_points=beam_points
        )
        motion_change_modifier = 2 * l2_norm(
            velocity
        )  # larger cost of changing the motion if larger velocity
        # for the maximum velocity of 0.4 (in one direction), the radial change in motion needs to be 1.5 times the radial deviation from the target direction in order to yield a motion change cost equal to the target cost
        # e.g., a motion change of (1.5*np.pi) / 8 times a motion change cost of 2 * l2_norm([0.4,0]) has the same total cost as a deviation of np.pi / 8 from the target times the target cost 1.2

        collision_cost_per_sector = self.collision_cost_per_sector(
            rcp,
            polar_sectors=polar_sectors,
            beam_points=beam_points,
            object_geometries_in_range=object_geometries_in_range,
            sector_size=self.sector_size,
        )
        # collision cost reduced if robot not at max speed
        self.collision_modifier = 4 * np.pi * l2_norm(velocity) / self.max_vel

        total_cost = (
            target_cost_per_sector * self.target_cost_modifier
            + motion_change_cost_per_sector * motion_change_modifier
            + collision_cost_per_sector * self.collision_modifier
        )
        best_sector = np.argmin(total_cost)
        direction_modifier = signed_angle_2d(
            target_vector, beam_points[best_sector].reshape(-1, 2)
        )

        # if id == 12:
        #     print(
        #         id,
        #         " bs ",
        #         np.argmin(target_cost_per_sector),
        #         np.argmin(motion_change_cost_per_sector),
        #         np.argmin(collision_cost_per_sector),
        #         best_sector,
        #     )

        # shift vector to target by angle direction_modifier
        rotation_matrix = np.array(
            [
                [np.cos(direction_modifier), -np.sin(direction_modifier)],
                [np.sin(direction_modifier), np.cos(direction_modifier)],
            ]
        ).reshape(2, 2)
        relative_robot_target_adapted = rotation_matrix @ (target_vector)

        assert np.isclose(
            signed_angle_2d(target_vector, relative_robot_target_adapted),
            direction_modifier,
        )
        assert np.isclose(
            l2_norm(target_vector),
            l2_norm(relative_robot_target_adapted),
        )
        assert np.isclose(
            relative_robot_target_adapted / l2_norm(relative_robot_target_adapted),
            beam_points[best_sector] / l2_norm(beam_points[best_sector]),
        ).all(), print(relative_robot_target_adapted, beam_points[best_sector])

        return relative_robot_target_adapted

    def motion_change_cost_per_sector(
        self, velocity: np.ndarray, polar_sectors: np.ndarray, beam_points: np.ndarray
    ):
        # cost of changing the motion direction from the last selected motion direction
        if l2_norm(velocity) > 1e-2:
            motion_change_cost_per_sector = smallest_angle_2d(velocity, beam_points)
        else:
            motion_change_cost_per_sector = np.zeros_like(polar_sectors)
        return motion_change_cost_per_sector

    def collision_cost_per_sector(
        self,
        rcp: np.ndarray,
        polar_sectors: np.ndarray,
        beam_points: np.ndarray,
        object_geometries_in_range: list,
        sector_size: float,
    ):

        # cost of sectors according to their proximity to an obstacle (0, 1]
        collision_cost_per_sector = np.zeros_like(polar_sectors)

        for alpha_i, bp in enumerate(beam_points):
            _, d_ij = self.check_for_collision(
                rcp, target_pos=rcp + bp, object_geometries=object_geometries_in_range
            )

            collision_cost = np.exp(-(d_ij / (self.max_vel * self.ts_control)) * 0.3)
            collision_cost_per_sector[alpha_i] += collision_cost
            for i in range(1, 3):
                collision_cost_per_sector[
                    (alpha_i - i) % len(beam_points)
                ] += collision_cost / max(1, i * sector_size)
                collision_cost_per_sector[
                    (alpha_i + i) % len(beam_points)
                ] += collision_cost / max(1, i * sector_size)
        return collision_cost_per_sector

    def get_lidar_info(
        self,
        rcp: np.ndarray,
        robot_target: np.ndarray,
        warehouse: Warehouse,
        path: Path,
    ) -> tuple[list, list, list, list]:

        robot_padding = (
            2 * self.robot_radius + 0.2
        )  # own radius and radius of other robot
        shelf_padding = (
            self.robot_radius + 0.05
        )  # pad objects directly instead of using gamma_ij

        # robots
        robots_in_range: list[RobotSimulation] = []
        for sr in self.simulated_robots:
            pos = sr.get_robot_state()[-2:]
            if l2_norm(rcp - pos) < self.lidar_range:
                robots_in_range.append(sr)

        robot_positions_in_range = [
            rob.get_robot_state()[-2:] for rob in robots_in_range
        ]
        motion_info_robots_in_range = []

        for robot in robots_in_range:
            velocity = robot.get_current_velocity()
            motion_deviation = smallest_angle_2d(robot_target - rcp, velocity)
            speed = l2_norm(velocity)
            r_geo = Point(robot.get_robot_state()[-2:]).buffer(robot_padding)

            motion_info_robots_in_range.append(
                {
                    "geometry": r_geo,
                    "motion_deviation": motion_deviation,
                    "speed": speed,
                }
            )

        # shelves
        shelves = [
            warehouse.graph.nodes(data=True)[shelf_id]["object"]
            for shelf_id in warehouse.shelves
            if not shelf_id in path.all_path_waypoints
        ]
        shelf_positions_in_range = [
            shelf.position
            for shelf in shelves
            if l2_norm(rcp - shelf.position) < self.lidar_range
        ]

        shelf_geometries_in_range = [
            shelf.compute_extension(padding=shelf_padding)
            for shelf in shelves
            if l2_norm(rcp - shelf.position) < self.lidar_range
        ]

        # walls
        wall_geometries = [
            warehouse.graph.nodes(data=True)[wall_id]["object"].buffer(
                shelf_padding, cap_style="square"
            )
            for wall_id in warehouse.walls
        ]

        return (
            robot_positions_in_range,
            motion_info_robots_in_range,
            shelf_positions_in_range,
            shelf_geometries_in_range + wall_geometries,
        )

    def check_for_collision(
        self, rcp, target_pos, object_geometries
    ) -> tuple[bool, float]:

        is_colliding = np.array(Point(rcp).within(object_geometries)).any()

        line_to_target = LineString([rcp, target_pos])
        is_close_to_object_border = np.array(
            line_to_target.intersects(object_geometries)
        )

        if not is_close_to_object_border.any():
            assert not is_colliding
            return bool(is_colliding), float(np.inf)

        intersection_lines = line_to_target.intersection(
            np.hstack(object_geometries)[is_close_to_object_border]
        )
        intersection_points = np.vstack(
            [np.array(line.xy).T for line in intersection_lines]
        )

        # distances to intersections
        d_ij = l2_norm(intersection_points - rcp)

        if not is_colliding:
            return bool(is_colliding), float(np.min(d_ij))

        else:
            # exclude intersections with distance 0 that arise from the collision(s)
            d_ij = np.atleast_1d(np.array(d_ij, dtype=float))
            d_ij_exclude_collisions = d_ij[d_ij > 1e-5]
            # preferable direction since all collisions distances are close to 0
            if len(d_ij_exclude_collisions) == 0:
                return bool(is_colliding), float(np.inf)

            # check if the robot fits into the space between the collision object border and the border of the next closest object
            sorted_distances = np.sort(d_ij_exclude_collisions)
            no_second_collision_object = len(sorted_distances) == 1
            enough_space_between_objects = len(sorted_distances) > 1 and (
                l2_norm(sorted_distances[1] - sorted_distances[0])
                > self.robot_radius * 2
            )
            if no_second_collision_object or enough_space_between_objects:
                return bool(is_colliding), -np.min(d_ij_exclude_collisions)
            else:
                return bool(is_colliding), -np.max(d_ij_exclude_collisions)
