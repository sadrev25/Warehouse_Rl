import numpy as np
from helper_functions import l2_norm
from scipy.interpolate import make_interp_spline


class Path:

    def __init__(
        self, path_length: float = np.inf, waypoints=[], waypoint_coords: list = []
    ):

        self.length = path_length
        self.waypoints = waypoints

        if len(waypoint_coords) > 0:
            self.set_waypoint_coords

    def set_edges(self, edge_ids: list[str]):
        # first waypoint in list has no corresponding edge
        self.edge_ids = [None] + edge_ids
        assert len(self.edge_ids) == len(self.waypoints)

    def set_waypoint_coords(
        self,
        waypoint_coords: np.ndarray,
    ):
        """Assign coordinates to the waypoints and pass information about the final target position. This position might be one of the nodes of the path or any other position.

        Args:
            waypoint_coords (np.ndarray): 2d-coordinates
            final_target_of_path (np.ndarray): Target position after the last segment has been reached
        """
        self.waypoint_2d_pos = waypoint_coords
        self.final_target_pos = waypoint_coords[-1]

        self.all_path_waypoints = (
            self.waypoints.copy()
        )  # keep a copy of all waypoints that is not adapted according to the waypoints that a robot has reached

        self.waypoint_1d_pos_on_path = self.compute_waypoint_1d_pos()

        self.path_spl = make_interp_spline(
            x=self.waypoint_1d_pos_on_path,
            y=self.waypoint_2d_pos,
            k=1,
        )

    def compute_waypoint_1d_pos(self) -> np.ndarray:

        waypoint_distances = l2_norm(
            self.waypoint_2d_pos[1:] - self.waypoint_2d_pos[:-1]
        )
        waypoint_1d_pos_on_path = np.pad(np.cumsum(waypoint_distances), (1, 0))
        try:
            assert np.isclose(waypoint_1d_pos_on_path[-1], self.length)
        except:
            print(self.length, waypoint_1d_pos_on_path[-1])

        return waypoint_1d_pos_on_path

    def get_next_path_waypoints_info(
        self, max_dist: float, current_pos: np.ndarray, max_vel: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Considering the current path segment, get the names of the next waypoints that lie within distance max_dist, as well as the distance from the current position to the waypoints

        Args:
            max_dist (float): maximum distance to waypoints that are considered 'next'
            current_pos (np.ndarray): 2d position of entity on path or close to path
            max_vel float: maximum velocity of robots

        Returns:
            array containing the names of the next waypoints and the names of the edges towards the waypoints, and array containing the estimated time until reaching the waypoint
        """
        current_1d_pos_on_path = self.get_current_1d_pos_on_path(current_pos)

        distance_to_waypoints = self.waypoint_1d_pos_on_path - current_1d_pos_on_path
        waypoint_mask = np.where((distance_to_waypoints < max_dist))[0]

        next_waypoints = np.array(self.waypoints)[waypoint_mask]
        edges_to_next_waypoints = np.array(self.edge_ids)[waypoint_mask]
        estimated_time_to_next_waypoints = (
            distance_to_waypoints[waypoint_mask]
            / max_vel  # assume that the robot can quickly reach the maximum velocity even if idle
        )
        return (
            np.c_[next_waypoints, edges_to_next_waypoints],
            estimated_time_to_next_waypoints,
        )

    def get_next_path_target(
        self,
        current_pos: np.ndarray,
        target_dist_straight_segment: float,
        max_dist_waypoint: float = 1,
    ) -> np.ndarray:

        current_1d_pos_on_path = self.get_current_1d_pos_on_path(current_pos)
        dist_to_next_waypoint = l2_norm(self.get_next_waypoint_coords() - current_pos)

        if current_1d_pos_on_path + max_dist_waypoint > self.length:
            return self.final_target_pos

        if target_dist_straight_segment < dist_to_next_waypoint:
            target_1d_pos_on_path = (
                current_1d_pos_on_path + target_dist_straight_segment
            )
        else:
            target_1d_pos_on_path = current_1d_pos_on_path + max_dist_waypoint

            if dist_to_next_waypoint < max_dist_waypoint:
                self.adapt_remaining_waypoints()

        target_2d_pos = self.path_spl(target_1d_pos_on_path)

        return target_2d_pos

    def get_current_1d_pos_on_path(self, current_pos: np.ndarray) -> float:

        next_waypoint_1d_pos_on_path = self.waypoint_1d_pos_on_path[0]
        dist_to_next_waypoint = l2_norm(self.get_next_waypoint_coords() - current_pos)
        current_1d_pos_on_path = next_waypoint_1d_pos_on_path - dist_to_next_waypoint
        return current_1d_pos_on_path

    def adapt_remaining_waypoints(self) -> None:

        if self.waypoint_2d_pos.size > 1 and self.waypoint_1d_pos_on_path.size > 1:
            self.waypoints = self.waypoints[1:]
            self.edge_ids = self.edge_ids[1:]
            self.waypoint_2d_pos = self.waypoint_2d_pos[1:]
            self.waypoint_1d_pos_on_path = self.waypoint_1d_pos_on_path[1:]
        return

    def get_path_attr(self):
        return (
            self.length,
            self.waypoints,
            self.waypoint_1d_pos_on_path,
            self.waypoint_2d_pos,
        )

    def get_next_waypoint_coords(self):
        return self.waypoint_2d_pos[0]
