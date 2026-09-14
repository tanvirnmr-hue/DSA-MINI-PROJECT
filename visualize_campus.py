"""
Optional visualization for the Smart Campus Navigator.
Draws the full campus graph and highlights a computed route.

Usage:
    python3 visualize_campus.py
(Edit START / END / ALGORITHM below, or import draw_campus() elsewhere.)
"""

import matplotlib.pyplot as plt
from campus_navigator import build_sample_campus, RouteOptimizer


def draw_campus(graph, path=None, save_path="campus_map.png"):
    fig, ax = plt.subplots(figsize=(9, 8))

    # draw all edges (light gray)
    for a, b, w in graph.all_edges():
        ax_, ay_ = graph.nodes[a]["x"], graph.nodes[a]["y"]
        bx_, by_ = graph.nodes[b]["x"], graph.nodes[b]["y"]
        ax.plot([ax_, bx_], [ay_, by_], color="lightgray", zorder=1, linewidth=1.5)

    # highlight the route, if given
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            ax_, ay_ = graph.nodes[a]["x"], graph.nodes[a]["y"]
            bx_, by_ = graph.nodes[b]["x"], graph.nodes[b]["y"]
            ax.plot([ax_, bx_], [ay_, by_], color="crimson", zorder=2, linewidth=3)

    # draw nodes
    for name, data in graph.nodes.items():
        color = "crimson" if path and name in path else "steelblue"
        ax.scatter(data["x"], data["y"], s=250, color=color, zorder=3,
                   edgecolors="black")
        ax.annotate(name, (data["x"], data["y"]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)

    ax.set_title("Smart Campus Map" + (" — Highlighted Route" if path else ""))
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved map to {save_path}")


if __name__ == "__main__":
    graph = build_sample_campus()
    optimizer = RouteOptimizer(graph)

    START = "Main Gate"
    END = "Girls Hostel"
    path, cost = optimizer.shortest_route(START, END, "dijkstra")
    print(f"Shortest route {START} -> {END}: {path} (distance={cost}m)")

    draw_campus(graph, path=path, save_path="campus_map.png")