import pickle

import networkx as nx
import numpy as np
from helper_functions import l2_norm
from shapely import LineString
from warehouse_env.shelf import Shelf, Station
from warehouse_env.task import Task


class Warehouse:

    def __init__(
        self,
        graphml_path="../../data/warehouse/Warehouse_preprocessed.graphml",
        min_task_prio: float = 1.0,
        max_task_prio: float = 5.0,
    ):
        self.TASKS = ["restock", "reorder", "retrieve"]
        self.graph = nx.read_graphml(graphml_path)
        self.initialize_env_types_with_nodes()

        for n1, n2 in self.graph.edges:
            self.graph.edges[n1, n2]["geometry"] = LineString(
                [self.get_coords(n1), self.get_coords(n2)]
            )
        self.initialize_stations()
        self.initialize_walls()
        self.min_task_priority = min_task_prio
        self.max_task_priority = max_task_prio
        return

    def get_next_task(self, task_number: int) -> Task:

        task_type = np.random.choice(self.TASKS, p=[0.15, 0.6, 0.25])
        match task_type:
            case "restock":
                current_station = np.random.choice(self.delivery_stations)
                target_station = np.random.choice(self.shelves)
            case "reorder":
                current_station, target_station = np.random.choice(
                    self.shelves, size=2, replace=False
                )
            case "retrieve":
                current_station = np.random.choice(self.shelves)
                target_station = np.random.choice(self.machines)
        priority = np.random.uniform(self.min_task_priority, self.max_task_priority)

        task = Task(
            task_number,
            task_type,
            priority,
            current_station,  # type: ignore
            target_station,  # type: ignore
        )
        return task

    def initialize_env_types_with_nodes(self) -> None:

        self.shelves = []
        self.waypoints = []
        self.delivery_stations = []
        self.machines = []
        self.doors = []
        self.charging_stations = []
        self.walls = []

        for node_id in self.graph.nodes:

            if "description" in self.graph.nodes[node_id]:
                if self.graph.nodes[node_id]["description"] == "machine":
                    self.machines.append(node_id)
                elif self.graph.nodes[node_id]["description"] == "delivery":
                    self.delivery_stations.append(node_id)
                elif self.graph.nodes[node_id]["description"] == "door":
                    self.doors.append(node_id)
                elif self.graph.nodes[node_id]["description"] == "shelf":
                    self.shelves.append(node_id)
                elif self.graph.nodes[node_id]["description"] == "charging station":
                    self.charging_stations.append(node_id)
                elif self.graph.nodes[node_id]["description"] == "wall":
                    self.walls.append(node_id)
                elif (
                    self.graph.nodes[node_id]["description"] == "waypoint"
                    and self.graph.adj[node_id] != {}
                ):
                    self.waypoints.append(node_id)
            else:
                print(f"Uncategorized node: {node_id, self.graph.nodes[node_id]}")
        return

    def initialize_stations(self) -> None:
        for node_id in self.shelves:
            shelf = Shelf(self.graph.nodes(data=True)[node_id])
            shelf.compute_access_points(self.graph.edges(data=True))
            self.graph.nodes[node_id]["object"] = shelf
        for node_id in self.machines + self.delivery_stations:
            station = Station(self.graph.nodes(data=True)[node_id])
            station.compute_access_points(self.graph.edges(data=True))
            self.graph.nodes[node_id]["object"] = station
        return

    def initialize_walls(self) -> None:
        for node_id in self.walls:
            wall_data = self.graph.nodes(data=True)[node_id]
            if wall_data["width"] > wall_data["height"]:
                self.graph.nodes[node_id]["object"] = LineString(
                    [
                        (wall_data["x"], wall_data["y"]),
                        (wall_data["x"] + wall_data["width"], wall_data["y"]),
                    ],
                )
            else:
                self.graph.nodes[node_id]["object"] = LineString(
                    [
                        (wall_data["x"], wall_data["y"]),
                        (wall_data["x"], wall_data["y"] + wall_data["height"]),
                    ],
                )

    def get_coords(self, node_id):
        data = self.graph.nodes(data=True)[node_id]
        if (
            data["description"] == "shelf"
            or data["description"] == "delivery"
            or data["description"] == "machine"
        ):
            return data["object"].get_position()  # type: ignore
        return np.array([data["x"], data["y"]])

    def get_node_from_pos(self, position: np.ndarray, max_dist: float = 0.1) -> str:
        for node_id in self.graph.nodes:
            if l2_norm(position - self.get_coords(node_id)) < max_dist:
                return node_id
        max_dist += 0.1
        return self.get_node_from_pos(position, max_dist)

    def get_waypoints_from_pos(
        self, position: np.ndarray, max_dist: float = 0.1
    ) -> list[str]:
        for node_id in self.waypoints:
            if l2_norm(position - self.get_coords(node_id)) < max_dist:
                return [node_id]
        for node_id in self.shelves:
            if l2_norm(position - self.get_coords(node_id)) < max_dist:
                shelf: Shelf = self.graph.nodes(data=True)[node_id]["object"]
                access_points = [n2 for _, n2 in shelf.accessible_edges]
                return access_points
        max_dist += 0.1
        return self.get_waypoints_from_pos(position, max_dist=max_dist)
