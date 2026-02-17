from typing import Dict

import numpy as np
from shapely import Geometry
from shapely.geometry import Polygon
from warehouse_env.polygon import PolygonWrapper


class Station:
    def __init__(self, data: Dict, **kwargs) -> None:
        super().__init__(**kwargs)
        """
        Create a complex polygon.

        Parameters
        ----------
        vertices: np.ndarray
            (x, y) coordinates of the polygon vertices.
        """
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

    def compute_access_points(self, edges, padding=0) -> None:

        accessible_edges = []
        while len(accessible_edges) == 0:
            possible_access_points = [
                self.position + (0.5 + padding) * np.array([-self.width, 0]),
                self.position + (0.5 + padding) * np.array([0, -self.height]),
                self.position + (0.5 + padding) * np.array([self.width, 0]),
                self.position + (0.5 + padding) * np.array([0, self.height]),
            ]

            for n1, n2, data in edges:
                edge_geometry = data.get("geometry")
                if Polygon(possible_access_points).convex_hull.intersection(
                    edge_geometry
                ):
                    accessible_edges.append((n1, n2))

            padding += 0.1

        self.accessible_edges = accessible_edges
        return

    def get_position(self) -> np.ndarray:
        return self.position


class Shelf(Station):
    def __init__(self, data: Dict, **kwargs) -> None:
        super().__init__(data, **kwargs)
