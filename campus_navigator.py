"""
==================================================================
 SMART CAMPUS NAVIGATION & ROUTE OPTIMIZER
 A Data Structures & Algorithms Mini Project
==================================================================

DATA STRUCTURES USED
---------------------
1. Graph (Adjacency List)   -> models the campus map (locations & paths)
2. Priority Queue (heapq)   -> used inside Dijkstra's and A* algorithms
3. Hash Map / Dictionary    -> O(1) lookup of locations, distances, parents
4. Disjoint Set / Union-Find-> used by Kruskal's MST (network optimizer)
5. Queue (collections.deque)-> used by BFS

ALGORITHMS IMPLEMENTED
-----------------------
1. BFS                -> fewest number of "hops" between two locations
2. Dijkstra's Algorithm-> shortest weighted route (distance/time)
3. A* Search           -> heuristic-guided shortest route (faster in practice)
4. Kruskal's MST        -> minimum total road length needed to keep every
                           building connected (campus infrastructure planning)
5. Alternate Routes     -> simple k-shortest-path variant (edge removal)

Author: (your name here)
==================================================================
"""

import heapq
import math
from collections import deque, defaultdict


# ------------------------------------------------------------------
# 1. CORE DATA STRUCTURE: THE CAMPUS GRAPH
# ------------------------------------------------------------------
class CampusGraph:
    """
    Weighted, undirected graph representing the campus.
    Nodes  = campus locations (buildings, gates, blocks, etc.)
    Edges  = walkable paths/roads between them, weight = distance in metres
    """

    def __init__(self):
        self.nodes = {}                       # name -> {"x":.., "y":.., "category":..}
        self.adj = defaultdict(list)          # name -> [(neighbor, weight), ...]

    def add_location(self, name, x, y, category="general"):
        self.nodes[name] = {"x": x, "y": y, "category": category}
        if name not in self.adj:
            self.adj[name] = []

    def add_path(self, a, b, weight=None):
        """Add an undirected path between two locations.
        If weight is not given, it is auto-computed from coordinates
        (straight-line distance) which keeps the sample data compact."""
        if weight is None:
            weight = self._euclidean(a, b)
        self.adj[a].append((b, weight))
        self.adj[b].append((a, weight))

    def _euclidean(self, a, b):
        ax, ay = self.nodes[a]["x"], self.nodes[a]["y"]
        bx, by = self.nodes[b]["x"], self.nodes[b]["y"]
        return round(math.hypot(ax - bx, ay - by), 1)

    def neighbors(self, node):
        return self.adj[node]

    def locations_by_category(self, category):
        return [n for n, data in self.nodes.items() if data["category"] == category]

    def all_edges(self):
        """Return each undirected edge once as (a, b, weight)."""
        seen = set()
        edges = []
        for a in self.adj:
            for b, w in self.adj[a]:
                key = tuple(sorted((a, b)))
                if key not in seen:
                    seen.add(key)
                    edges.append((a, b, w))
        return edges


# ------------------------------------------------------------------
# 2. ALGORITHMS
# ------------------------------------------------------------------
def bfs_path(graph: CampusGraph, start, goal):
    """Fewest-hops path (ignores distance weights)."""
    if start not in graph.nodes or goal not in graph.nodes:
        return None, None
    visited = {start}
    parent = {start: None}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current == goal:
            return _reconstruct(parent, goal), len(_reconstruct(parent, goal)) - 1
        for neighbor, _ in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    return None, None


def dijkstra(graph: CampusGraph, start, goal=None):
    """
    Shortest weighted distance from `start` to every node (or just `goal`).
    Uses a binary heap (priority queue) -> O((V+E) log V)
    Returns: (distances dict, parent dict)
    """
    distances = {node: math.inf for node in graph.nodes}
    parent = {node: None for node in graph.nodes}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()

    while pq:
        dist_u, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if goal is not None and u == goal:
            break

        for v, weight in graph.neighbors(u):
            new_dist = dist_u + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))

    return distances, parent


def astar(graph: CampusGraph, start, goal):
    """
    A* search: like Dijkstra but guided by a heuristic (straight-line
    distance to the goal), which usually explores far fewer nodes.
    """
    def h(node):  # heuristic = straight-line distance to goal
        nx, ny = graph.nodes[node]["x"], graph.nodes[node]["y"]
        gx, gy = graph.nodes[goal]["x"], graph.nodes[goal]["y"]
        return math.hypot(nx - gx, ny - gy)

    g_score = {node: math.inf for node in graph.nodes}
    g_score[start] = 0
    parent = {start: None}
    open_set = [(h(start), start)]
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = _reconstruct(parent, goal)
            return path, g_score[goal]
        if current in visited:
            continue
        visited.add(current)

        for neighbor, weight in graph.neighbors(current):
            tentative = g_score[current] + weight
            if tentative < g_score[neighbor]:
                g_score[neighbor] = tentative
                parent[neighbor] = current
                f_score = tentative + h(neighbor)
                heapq.heappush(open_set, (f_score, neighbor))

    return None, math.inf


def _reconstruct(parent, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()
    return path if path and path[0] is not None else path


# ---- Union-Find (Disjoint Set) for Kruskal's MST -------------------
class DisjointSet:
    def __init__(self, items):
        self.parent = {i: i for i in items}
        self.rank = {i: 0 for i in items}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def kruskal_mst(graph: CampusGraph):
    """
    Minimum Spanning Tree -> the shortest total length of road network
    that still keeps every campus location reachable. Useful for campus
    infrastructure / new-road planning.
    """
    edges = sorted(graph.all_edges(), key=lambda e: e[2])
    ds = DisjointSet(graph.nodes.keys())
    mst_edges = []
    total = 0
    for a, b, w in edges:
        if ds.union(a, b):
            mst_edges.append((a, b, w))
            total += w
    return mst_edges, round(total, 1)


# ------------------------------------------------------------------
# 3. ROUTE OPTIMIZER  (the "application layer")
# ------------------------------------------------------------------
class RouteOptimizer:
    def __init__(self, graph: CampusGraph):
        self.graph = graph

    def shortest_route(self, start, end, algorithm="dijkstra"):
        if start not in self.graph.nodes or end not in self.graph.nodes:
            return None, None
        if algorithm == "bfs":
            path, cost = bfs_path(self.graph, start, end)
            return path, cost
        elif algorithm == "astar":
            return astar(self.graph, start, end)
        else:  # dijkstra (default) - true shortest distance
            distances, parent = dijkstra(self.graph, start, end)
            if distances[end] == math.inf:
                return None, None
            return _reconstruct(parent, end), round(distances[end], 1)

    def nearest_facility(self, start, category):
        """
        Find the closest location of a given category (e.g. 'canteen',
        'washroom', 'atm') from the current position using a single
        Dijkstra run from `start`.
        """
        targets = self.graph.locations_by_category(category)
        if not targets:
            return None, None, None
        distances, parent = dijkstra(self.graph, start)
        best = min(targets, key=lambda t: distances.get(t, math.inf))
        if distances[best] == math.inf:
            return None, None, None
        return best, _reconstruct(parent, best), round(distances[best], 1)

    def alternate_route(self, start, end):
        """
        A simple 'second best' route: find the shortest path, temporarily
        remove one of its edges, and re-run Dijkstra. This is a light-
        weight stand-in for Yen's k-shortest-paths algorithm.
        """
        primary, cost = self.shortest_route(start, end, "dijkstra")
        if not primary or len(primary) < 2:
            return primary, cost, None, None

        # Remove the first edge of the shortest path and search again
        u, v = primary[0], primary[1]
        removed = []
        for lst, a, b in ((self.graph.adj[u], u, v), (self.graph.adj[v], v, u)):
            for i, (n, w) in enumerate(lst):
                if n == b:
                    removed.append((a, i, (n, w)))

        for owner, idx, edge in removed:
            self.graph.adj[owner].remove(edge)

        alt_path, alt_cost = self.shortest_route(start, end, "dijkstra")

        # restore the graph
        for owner, idx, edge in removed:
            self.graph.adj[owner].insert(idx, edge)

        if alt_path == primary:
            alt_path, alt_cost = None, None
        return primary, cost, alt_path, alt_cost

    def campus_network_plan(self):
        """Return the minimum road network (MST) connecting all buildings."""
        return kruskal_mst(self.graph)


# ------------------------------------------------------------------
# 4. SAMPLE CAMPUS DATA
# ------------------------------------------------------------------
def build_sample_campus():
    g = CampusGraph()

    # name, x, y, category
    locations = [
        ("Main Gate",        0,  0, "gate"),
        ("Admin Block",      2,  1, "office"),
        ("Library",          4,  2, "study"),
        ("Auditorium",       6,  1, "hall"),
        ("CS Department",    3,  4, "department"),
        ("ECE Department",   5,  4, "department"),
        ("Mechanical Dept",  7,  4, "department"),
        ("Canteen 1",        3,  1, "canteen"),
        ("Canteen 2",        6,  5, "canteen"),
        ("Boys Hostel",      1,  6, "hostel"),
        ("Girls Hostel",     8,  6, "hostel"),
        ("Sports Complex",   2,  8, "recreation"),
        ("Medical Center",   5,  0, "medical"),
        ("Parking",         -1,  2, "parking"),
        ("Back Gate",        9,  8, "gate"),
        ("ATM",              1,  1, "atm"),
        ("Washroom Block A", 4,  1, "washroom"),
        ("Washroom Block B", 6,  4, "washroom"),
    ]
    for name, x, y, cat in locations:
        g.add_location(name, x, y, cat)

    # Walkable connections (auto-weighted by straight-line distance)
    paths = [
        ("Main Gate", "Admin Block"),
        ("Main Gate", "Parking"),
        ("Main Gate", "ATM"),
        ("Admin Block", "Library"),
        ("Admin Block", "Canteen 1"),
        ("Library", "CS Department"),
        ("Library", "Auditorium"),
        ("Library", "Medical Center"),
        ("CS Department", "ECE Department"),
        ("ECE Department", "Mechanical Dept"),
        ("ECE Department", "Canteen 2"),
        ("ECE Department", "Washroom Block B"),
        ("CS Department", "Boys Hostel"),
        ("Mechanical Dept", "Girls Hostel"),
        ("Mechanical Dept", "Back Gate"),
        ("Boys Hostel", "Sports Complex"),
        ("Girls Hostel", "Back Gate"),
        ("Canteen 1", "Washroom Block A"),
        ("Auditorium", "Medical Center"),
        ("Auditorium", "Canteen 2"),
        ("Parking", "Boys Hostel"),
        ("Canteen 2", "Girls Hostel"),
    ]
    for a, b in paths:
        g.add_path(a, b)

    return g


# ------------------------------------------------------------------
# 5. COMMAND-LINE INTERFACE
# ------------------------------------------------------------------
def print_path(path, cost, unit="m"):
    if not path:
        print("  No route found.")
        return
    print("  Route: " + "  ->  ".join(path))
    print(f"  Total distance: {cost} {unit}   |   Stops: {len(path)}")


def list_locations(graph):
    print("\nAvailable locations:")
    for i, name in enumerate(sorted(graph.nodes), 1):
        cat = graph.nodes[name]["category"]
        print(f"  {i:2d}. {name}  [{cat}]")


def main():
    graph = build_sample_campus()
    optimizer = RouteOptimizer(graph)

    menu = """
============================================
   SMART CAMPUS NAVIGATION & ROUTE OPTIMIZER
============================================
1. List all campus locations
2. Find shortest route (Dijkstra)
3. Find shortest route (A*)
4. Find fewest-stops route (BFS)
5. Find nearest facility (e.g., canteen, washroom, atm, medical, parking)
6. Get primary + alternate route
7. View minimum road network plan (MST)
0. Exit
--------------------------------------------
"""
    while True:
        print(menu)
        choice = input("Choose an option: ").strip()

        if choice == "1":
            list_locations(graph)

        elif choice in ("2", "3", "4"):
            list_locations(graph)
            start = input("\nStart location: ").strip()
            end = input("Destination: ").strip()
            if choice == "2":
                path, cost = optimizer.shortest_route(start, end, "dijkstra")
            elif choice == "3":
                path, cost = optimizer.shortest_route(start, end, "astar")
            else:
                path, cost = optimizer.shortest_route(start, end, "bfs")
                print_path(path, cost, unit="stops" if choice == "4" else "m")
                continue
            print_path(path, cost)

        elif choice == "5":
            list_locations(graph)
            start = input("\nYour current location: ").strip()
            category = input(
                "Facility category (canteen/washroom/atm/medical/parking): "
            ).strip().lower()
            place, path, cost = optimizer.nearest_facility(start, category)
            if place:
                print(f"\nNearest '{category}': {place}")
                print_path(path, cost)
            else:
                print("  No matching facility found or unreachable.")

        elif choice == "6":
            list_locations(graph)
            start = input("\nStart location: ").strip()
            end = input("Destination: ").strip()
            primary, cost, alt, alt_cost = optimizer.alternate_route(start, end)
            print("\nPrimary route:")
            print_path(primary, cost)
            print("\nAlternate route:")
            if alt:
                print_path(alt, alt_cost)
            else:
                print("  No distinct alternate route available.")

        elif choice == "7":
            mst_edges, total = optimizer.campus_network_plan()
            print(f"\nMinimum road network (total length: {total} m):")
            for a, b, w in mst_edges:
                print(f"  {a}  --{w}m--  {b}")

        elif choice == "0":
            print("Goodbye!")
            break

        else:
            print("Invalid option, try again.")


if __name__ == "__main__":
    main()