from typing import Dict

import numpy as np
from shapely import Geometry, LineString
from shapely.geometry import Polygon
from warehouse_env.polygon import PolygonWrapper


class Station:
    def __init__(self, station_id: str, data: Dict, **kwargs) -> None:
        super().__init__(**kwargs)
        """
        Create a complex polygon.

        Parameters
        ----------
        vertices: np.ndarray
            (x, y) coordinates of the polygon vertices.
        """
        self.station_id = station_id
        self.upper_left_corner = np.array([data["x"], data["y"]])
        self.position = self.upper_left_corner + 0.5 * np.array(
            [data["width"], data["height"]]
        )
        self.width = data["width"]
        self.height = data["height"]

        self.geometry = self.compute_extension(padding=0)

    def compute_extension(self, padding=0) -> Geometry:

        vertices = [
            self.position
            + 0.5 * np.array([-self.width, -self.height])
            + padding * np.array([-1, -1]),
            self.position
            + 0.5 * np.array([-self.width, self.height])
            + padding * np.array([-1, 1]),
            self.position
            + 0.5 * np.array([self.width, -self.height])
            + padding * np.array([1, -1]),
            self.position
            + 0.5 * np.array([self.width, self.height])
            + padding * np.array([1, 1]),
        ]

        return Polygon(vertices).convex_hull

    def compute_access_points(self, edges, max_dist_center_to_lane) -> None:

        possible_access_lines = [
            LineString(
                [
                    self.position,
                    self.position + max_dist_center_to_lane * np.array([-1, 0]),
                ]
            ),
            LineString(
                [
                    self.position,
                    self.position + max_dist_center_to_lane * np.array([0, -1]),
                ]
            ),
            LineString(
                [
                    self.position,
                    self.position + max_dist_center_to_lane * np.array([1, 0]),
                ]
            ),
            LineString(
                [
                    self.position,
                    self.position + max_dist_center_to_lane * np.array([0, 1]),
                ]
            ),
        ]

        self.access_nodes = []
        self.access_edges = []

        for n1, n2, data in edges:
            edge_geometry = data.get("geometry")
            intersection_lines = edge_geometry.intersection(possible_access_lines)
            intersection_points = np.vstack(
                [np.array(line.xy).T for line in intersection_lines]
            )
            for ipx, ipy in intersection_points:
                ip_name = n1 + n2 + str(round(ipx, 1)) + str(round(ipy, 1))
                access_node = (
                    ip_name,
                    {"description": "access point", "x": ipx, "y": ipy},
                )
                self.access_nodes.append(access_node)
                self.access_edges.append((n1, ip_name, {"id": f"e_{ip_name}"}))
                self.access_edges.append(
                    (ip_name, self.station_id, {"id": f"e_{ip_name}"})
                )
                self.access_edges.append(
                    (self.station_id, ip_name, {"id": f"e_{ip_name}"})
                )
                # for the edge towards the next crossing n2 use the same id as for the edge from n1 to n2
                self.access_edges.append((ip_name, n2, {"id": data["id"]}))

        return

    def get_position(self) -> np.ndarray:
        return self.position
