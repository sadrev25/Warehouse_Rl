from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from itm_pythonfig.pythonfig import PythonFig
from matplotlib.path import Path
from shapely.geometry import Point, Polygon

class PolygonWrapper:
    def __init__(self, **kwargs) -> None:
       return

    def compute_polygon_area(self) -> float:
        return self.polygon.area

    def get_corners(self):
        return self.vertices[:-1]

    def pad_polygon(self, padding: float):
        self.polygon = self.polygon.buffer(padding, join_style="mitre")
        self.vertices = np.vstack(self.polygon.exterior.xy).T
        return

    def contains_point(self, point):
        return self.polygon.contains(Point(point[0], point[1]))

    def intersect(self, polygon2):
        poly_intersection = self.polygon.intersection(polygon2.polygon)
        return PolygonWrapper(np.vstack(poly_intersection.convex_hull.exterior.xy).T)

    def compute_center(self) -> np.ndarray:
        return np.array(self.polygon.centroid.xy).flatten()

    def compute_extension(self):
        x_ext = max(self.vertices[:, 0]) - min(self.vertices[:, 0])
        y_ext = max(self.vertices[:, 1]) - min(self.vertices[:, 1])
        ext_1d = np.max([x_ext, y_ext])
        ext_2d = np.sqrt((x_ext) ** 2 + (y_ext) ** 2)

        border = ext_1d * 0.02
        ratio = (x_ext + 2 * border) / (y_ext + 2 * border)

        return ext_1d, ext_2d, border, ratio, x_ext, y_ext

    def compute_polygon_points(self, step_size=0.1) -> Tuple[np.ndarray, np.ndarray]:
        # make a polygon
        polygon_path = Path(self.vertices)

        # make a canvas with coordinates
        gx, gy = np.meshgrid(
            np.arange(
                np.min(self.vertices[:, 0]), np.max(self.vertices[:, 0]), step_size
            ),
            np.arange(
                np.min(self.vertices[:, 1]), np.max(self.vertices[:, 1]), step_size
            ),
        )
        grid_points = np.vstack((gx.flatten(), gy.flatten())).T

        grid = polygon_path.contains_points(grid_points)
        mask = grid.reshape(gx.shape[0], gx.shape[1])

        polygon_points = np.hstack((gx[mask].reshape(-1, 1), gy[mask].reshape(-1, 1)))
        outside_points = np.hstack(
            (gx[1 - mask].reshape(-1, 1), gy[1 - mask].reshape(-1, 1))
        )
        return polygon_points, outside_points

    def plot_polygon(
        self,
        fig,
        ax,
        color,
        zorder=2,
        label="deployment area",
        **kwargs,
    ) -> None:

        if ax is None:
            ax = fig.gca()

        # plot the polygon edges
        poly = ax.plot(
            self.vertices[:, 0],
            self.vertices[:, 1],
            label=label,
            color=color,
            zorder=zorder,
            **kwargs,
        )

        return
