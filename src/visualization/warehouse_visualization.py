import matplotlib.pyplot as plt
import networkx as nx
from itm_pythonfig.pythonfig import PythonFig
from matplotlib.patches import Circle, Rectangle
from warehouse_env.warehouse import Warehouse


def draw_warehouse(ax, warehouse: Warehouse):

    warehouse_graph = warehouse.graph
    ax.invert_yaxis()

    # positions from parsed GraphML
    pos = {
        node: (data["x"], data["y"]) for node, data in warehouse_graph.nodes(data=True)
    }

    # draw edges first (so rectangles appear on top)
    nx.draw_networkx_edges(
        warehouse_graph,
        pos,
        ax=ax,
        edge_color="lightblue",
        alpha=0.8,
        arrowsize=3,
        width=0.2,
        node_size=5,
        min_source_margin=0,
        min_target_margin=0,
    )

    # draw rectangles for each node using width/height from GraphML
    for n, data in warehouse_graph.nodes(data=True):
        if data.get("description") == "waypoint":
            draw_waypoint(ax, n, data, color="lightblue")
        if data.get("description") == "shelf":
            draw_env_item(ax, data, color="gray")
        if data.get("description") == "door":
            draw_env_item(ax, data, color="yellow")
        if data.get("description") == "machine":
            draw_env_item(ax, data, color="darkblue")
        if data.get("description") == "delivery":
            draw_env_item(ax, data, color="darkgreen")
        if data.get("description") == "charging station":
            draw_env_item(ax, data, color="palevioletred", edgecolor="gray", zorder=1)
        if data.get("description") == "wall":
            draw_env_item(ax, data, color="silver", edgecolor="dimgray", zorder=0)

    # ax.autoscale_view()
    ax.set_aspect("equal")
    plt.axis("off")
    return


def draw_env_item(ax, data, color, edgecolor="black", zorder=2):

    rect = Rectangle(
        (data.get("x"), data.get("y")),
        data.get("width"),
        data.get("height"),
        facecolor=color,
        edgecolor=edgecolor,
        zorder=zorder,
    )
    ax.add_patch(rect)

    # label = data.get("description", node)
    # ax.text(x, y, label, ha="center", va="center", fontsize=6, zorder=3)
    return


def draw_waypoint(ax, n, data, color):

    circle = Circle(
        (data.get("x"), data.get("y")),
        radius=0.03,
        facecolor=color,
        edgecolor=None,
        zorder=2,
    )
    ax.add_patch(circle)

    return
