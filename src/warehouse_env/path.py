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

    def set_waypoint_coords(
        self,
        waypoint_coords: np.ndarray,
        start_node: str,
        start_pos_of_path: np.ndarray,
        final_target_of_path: str,
        final_target_pos_of_path: np.ndarray,
    ):
        """Assign coordinates to the waypoints and pass information about the final target position. This position might be one of the nodes of the path or any other position.

        Args:
            waypoint_coords (np.ndarray): 2d-coordinates
            final_target_of_path (np.ndarray): Target position after the last segment has been reached
        """
        self.waypoint_2d_pos = waypoint_coords
        if not np.isclose(start_pos_of_path, waypoint_coords[0]).all():
            self.length += l2_norm(start_pos_of_path - waypoint_coords[0])
            self.waypoint_2d_pos = np.vstack((start_pos_of_path, self.waypoint_2d_pos))
            self.waypoints = [start_node] + self.waypoints

        self.final_target_pos = final_target_pos_of_path
        if not np.isclose(self.final_target_pos, waypoint_coords[-1]).all():
            self.length += l2_norm(self.final_target_pos - waypoint_coords[-1])
            self.waypoint_2d_pos = np.vstack(
                (self.waypoint_2d_pos, self.final_target_pos)
            )
            self.waypoints.append(final_target_of_path)

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
            print(self.get_path_attr(), waypoint_1d_pos_on_path[-1])

        return waypoint_1d_pos_on_path

    def get_next_path_waypoints(
        self, max_dist: float, current_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Considering the current path segment, get the names of the next waypoints that lie within distance max_dist, as well as the distance from the current position to the waypoints

        Args:
            max_dist (float): maximum distance to waypoints that are considered 'next'
            current_pos (np.ndarray): 2d position of entity on path or close to path

        Returns:
            tuple[np.ndarray, np.ndarray]: names of and distances to next waypoints
        """
        current_1d_pos_on_path = self.get_current_1d_pos_on_path(current_pos)

        distance_to_waypoints = self.waypoint_1d_pos_on_path - current_1d_pos_on_path
        waypoint_mask = np.where((distance_to_waypoints < max_dist))[0]

        relevant_waypoints = np.array(self.waypoints)[waypoint_mask]
        return relevant_waypoints, distance_to_waypoints[waypoint_mask]

    def get_next_path_target(
        self, current_pos: np.ndarray, segment_length: float
    ) -> np.ndarray:

        current_1d_pos_on_path = self.get_current_1d_pos_on_path(current_pos)
        target_1d_pos_on_path = current_1d_pos_on_path + segment_length

        if current_1d_pos_on_path + segment_length > self.length:
            target_2d_pos = self.final_target_pos
        else:
            target_2d_pos = self.path_spl(target_1d_pos_on_path)

        if current_1d_pos_on_path + segment_length > self.waypoint_1d_pos_on_path[0]:
            self.adapt_remaining_waypoints()

        return target_2d_pos

    def get_current_1d_pos_on_path(self, current_pos: np.ndarray) -> float:

        next_waypoint_2d_pos = self.waypoint_2d_pos[0]
        next_waypoint_1d_pos_on_path = self.waypoint_1d_pos_on_path[0]

        dist_to_next_waypoint = l2_norm(next_waypoint_2d_pos - current_pos)
        current_1d_pos_on_path = next_waypoint_1d_pos_on_path - dist_to_next_waypoint
        return current_1d_pos_on_path

    def adapt_remaining_waypoints(self) -> None:

        if self.waypoint_2d_pos.size > 1 and self.waypoint_1d_pos_on_path.size > 1:
            self.waypoints = self.waypoints[1:]
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

    # def get_next_segment(self) -> tuple[np.ndarray, np.ndarray]:
    #     """Drop the current path segment. Move on to the next segment and return its coordinates.

    #     Returns:
    #         tuple[np.ndarray, np.ndarray]: 1d position of segment on path, 2d-coordinates of the end of the segment
    #     """
    #     if len(self.path_segments_2d) > 1:
    #         self.path_segments_2d = self.path_segments_2d[1:]
    #         self.path_segments_1d = self.path_segments_1d[1:]
    #     else:
    #         self.path_segments_2d = np.array([])
    #         self.path_segments_1d = np.array([])
    #     return self.get_current_segment()

    # def get_current_segment(self) -> tuple[np.ndarray, np.ndarray]:
    #     """Get the 1d and 2d coordinates of the current segment, or of the final target, if there is no segment left.

    #     Returns:
    #         tuple[np.ndarray, np.ndarray]: 1d position of segment on path, 2d-coordinates of the end of the segment
    #     """
    #     if len(self.path_segments_2d) == 0:
    #         return (
    #             self.length + self.distance_to_final_target,
    #             self.final_target_of_path,
    #         )
    #     return self.path_segments_1d[0], self.path_segments_2d[0]
