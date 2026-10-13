"""
Optimal Path Detection for a Delivery Vehicle
---------------------------------------------
Finds the route between a source and a destination that minimises
distance, travel time, trip cost, or a balanced mix of time and cost.

Algorithms : Dijkstra's algorithm and A* search (both written from scratch)
Data       : an illustrative road network defined in this file (sample data)

Usage:  python optimal_path.py
        python optimal_path.py --source Depot --dest Destination --plot routes.png
"""
import argparse
import heapq
import itertools
import math

# --------------------------------------------------------------------------
# 1. ROAD NETWORK (sample data)
# --------------------------------------------------------------------------
# Node coordinates in km on a plane
NODES = {
    "Depot":       (0, 0),
    "Junction-1":  (9, 6),
    "Junction-2":  (10, -7),
    "Junction-3":  (21, 9),
    "Junction-4":  (20, -1),
    "Junction-5":  (19, -13),
    "Junction-6":  (31, 5),
    "Junction-7":  (30, -7),
    "Junction-8":  (39, -1),
    "Destination": (47, 3),
}

# Road types: free-flow speed (km/h) and winding factor
# (winding factor = road length / straight-line length)
ROAD_TYPES = {
    "highway": {"speed": 70, "winding": 1.05},
    "main":    {"speed": 50, "winding": 1.10},
    "local":   {"speed": 30, "winding": 1.20},
}

FUEL_COST_PER_KM = 7.5   # rupees per km (assumed)
MAX_SPEED = max(r["speed"] for r in ROAD_TYPES.values())

# (from, to, road type, congestion factor >= 1.0, toll in rupees)
ROADS = [
    ("Depot",      "Junction-1",  "main",    1.3,   0),
    ("Depot",      "Junction-2",  "local",   1.0,   0),
    ("Depot",      "Junction-4",  "highway", 1.1,  70),
    ("Junction-1", "Junction-3",  "main",    1.2,   0),
    ("Junction-1", "Junction-4",  "local",   1.0,   0),
    ("Junction-2", "Junction-4",  "local",   1.0,   0),
    ("Junction-2", "Junction-5",  "local",   1.0,   0),
    ("Junction-3", "Junction-4",  "local",   1.8,   0),
    ("Junction-3", "Junction-6",  "highway", 1.1,  40),
    ("Junction-4", "Junction-6",  "main",    1.4,   0),
    ("Junction-4", "Junction-7",  "main",    1.0,   0),
    ("Junction-5", "Junction-7",  "local",   1.0,   0),
    ("Junction-6", "Junction-8",  "main",    1.2,   0),
    ("Junction-6", "Destination", "highway", 1.0, 110),
    ("Junction-7", "Junction-8",  "highway", 1.0,  35),
    ("Junction-8", "Destination", "main",    1.0,   0),
]


def euclid(a, b):
    (x1, y1), (x2, y2) = NODES[a], NODES[b]
    return math.hypot(x1 - x2, y1 - y2)


def build_graph():
    """Return adjacency list: node -> list of edge dicts (undirected roads)."""
    graph = {n: [] for n in NODES}
    for u, v, rtype, congestion, toll in ROADS:
        info = ROAD_TYPES[rtype]
        dist = round(euclid(u, v) * info["winding"], 1)                 # km
        time = round(dist / info["speed"] * 60 * congestion, 1)         # minutes
        cost = round(dist * FUEL_COST_PER_KM + toll, 1)                 # rupees
        for a, b in ((u, v), (v, u)):
            graph[a].append({"to": b, "type": rtype, "distance": dist,
                             "time": time, "cost": cost})
    return graph


# --------------------------------------------------------------------------
# 2. PATH SEARCH: Dijkstra and A*
# --------------------------------------------------------------------------
def a_star(graph, source, dest, weight, heuristic=None):
    """
    Generic best-first search. With heuristic=None this is Dijkstra's algorithm.
    weight(edge)   -> non-negative cost of using an edge
    heuristic(node)-> admissible lower bound of the remaining cost to dest
    Returns (path, total_weight, nodes_expanded).
    """
    h = heuristic or (lambda n: 0.0)
    g = {source: 0.0}
    parent = {source: None}
    heap = [(h(source), 0, source)]
    counter = itertools.count(1)
    closed = set()
    expanded = 0
    while heap:
        _, _, node = heapq.heappop(heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if node == dest:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1], g[dest], expanded
        for edge in graph[node]:
            nxt = edge["to"]
            new_g = g[node] + weight(edge)
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                parent[nxt] = node
                heapq.heappush(heap, (new_g + h(nxt), next(counter), nxt))
    return None, math.inf, expanded


def route_totals(graph, path):
    """Total distance (km), time (min) and cost (Rs) of a path."""
    d = t = c = 0.0
    for a, b in zip(path, path[1:]):
        e = next(x for x in graph[a] if x["to"] == b)
        d += e["distance"]; t += e["time"]; c += e["cost"]
    return round(d, 1), round(t, 1), round(c, 1)


# --------------------------------------------------------------------------
# 3. OBJECTIVES: shortest, fastest, cheapest, balanced
# --------------------------------------------------------------------------
def make_objectives(dest, alpha=0.5):
    """
    Each objective = (edge weight function, heuristic function).
    The heuristics use the straight-line distance, which never over-estimates
    the true remaining distance, time or cost, so A* stays optimal.
    Balanced objective: alpha * time/T_REF + (1 - alpha) * cost/C_REF.
    """
    T_REF, C_REF = 60.0, 400.0          # reference scales (minutes, rupees)
    h_dist = lambda n: euclid(n, dest)
    h_time = lambda n: euclid(n, dest) / MAX_SPEED * 60
    h_cost = lambda n: euclid(n, dest) * FUEL_COST_PER_KM
    return {
        "Shortest distance": (lambda e: e["distance"], h_dist),
        "Fastest (min time)": (lambda e: e["time"], h_time),
        "Cheapest (min cost)": (lambda e: e["cost"], h_cost),
        "Balanced (time + cost)": (
            lambda e: alpha * e["time"] / T_REF + (1 - alpha) * e["cost"] / C_REF,
            lambda n: (alpha * h_time(n) / T_REF
                       + (1 - alpha) * h_cost(n) / C_REF)),
    }


# --------------------------------------------------------------------------
# 4. VERIFICATION: brute force over all simple paths
# --------------------------------------------------------------------------
def brute_force(graph, source, dest, weight):
    best = (None, math.inf)
    def dfs(node, visited, path, total):
        nonlocal best
        if total >= best[1]:
            return
        if node == dest:
            best = (list(path), total)
            return
        for e in graph[node]:
            if e["to"] not in visited:
                visited.add(e["to"]); path.append(e["to"])
                dfs(e["to"], visited, path, total + weight(e))
                path.pop(); visited.remove(e["to"])
    dfs(source, {source}, [source], 0.0)
    return best


# --------------------------------------------------------------------------
# 5. VISUALISATION
# --------------------------------------------------------------------------
def plot_routes(graph, results, source, dest, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, r) in zip(axes.flat, results.items()):
        for u, v, *_ in ROADS:
            (x1, y1), (x2, y2) = NODES[u], NODES[v]
            ax.plot([x1, x2], [y1, y2], color="0.75", lw=1.5, zorder=1)
        px = [NODES[n][0] for n in r["path"]]
        py = [NODES[n][1] for n in r["path"]]
        ax.plot(px, py, color="black", lw=3, zorder=2)
        for n, (x, y) in NODES.items():
            special = n in (source, dest)
            ax.scatter(x, y, s=140 if special else 70,
                       color="black" if special else "white",
                       edgecolor="black", zorder=3)
            label = n.replace("Junction-", "J") if not special else n
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8)
        d, t, c = r["totals"]
        ax.set_title(f"{name}\n{d} km | {t} min | Rs {c}", fontsize=10, pad=16)
        ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_network(graph, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"highway": ("-", 3.0), "main": ("--", 2.0), "local": (":", 2.0)}
    fig, ax = plt.subplots(figsize=(9, 5))
    for u, v, rtype, cong, toll in ROADS:
        (x1, y1), (x2, y2) = NODES[u], NODES[v]
        ls, lw = styles[rtype]
        ax.plot([x1, x2], [y1, y2], color="black", ls=ls, lw=lw, zorder=1)
        e = next(x for x in graph[u] if x["to"] == v)
        ax.annotate(f"{e['distance']} km", ((x1 + x2) / 2, (y1 + y2) / 2),
                    fontsize=7, ha="center",
                    bbox=dict(fc="white", ec="none", pad=0.5))
    for n, (x, y) in NODES.items():
        ax.scatter(x, y, s=90, color="white", edgecolor="black", zorder=3)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, weight="bold")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="black", ls=s[0], lw=s[1],
                      label=k.capitalize() + " road") for k, s in styles.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(filename, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
# 6. MAIN PROGRAM
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Optimal path detection for a vehicle")
    ap.add_argument("--source", default="Depot")
    ap.add_argument("--dest", default="Destination")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight of time in the balanced objective (0 to 1)")
    ap.add_argument("--plot", default=None, help="save a route comparison figure")
    ap.add_argument("--network-plot", default=None,
                    help="save a figure of the road network")
    args = ap.parse_args()

    graph = build_graph()
    if args.source not in graph or args.dest not in graph:
        raise SystemExit("Unknown node. Choose from: " + ", ".join(graph))

    print("=" * 74)
    print(f" OPTIMAL PATH DETECTION   {args.source}  ->  {args.dest}")
    print("=" * 74)
    print(f" Road network: {len(graph)} nodes, {len(ROADS)} roads "
          f"| fuel Rs {FUEL_COST_PER_KM}/km\n")

    objectives = make_objectives(args.dest, args.alpha)
    results = {}
    for name, (w, h) in objectives.items():
        S, D = args.source, args.dest
        path_d, val_d, exp_d = a_star(graph, S, D, w)       # Dijkstra
        path_a, val_a, exp_a = a_star(graph, S, D, w, h)    # A*
        path_b, val_b = brute_force(graph, S, D, w)         # check
        ok = (path_d == path_a and abs(val_d - val_b) < 1e-9
              and abs(val_a - val_b) < 1e-9)
        d, t, c = route_totals(graph, path_a)
        results[name] = {"path": path_a, "totals": (d, t, c),
                         "exp_d": exp_d, "exp_a": exp_a, "ok": ok}
        print(f"[{name}]")
        print("  Route    : " + " -> ".join(path_a))
        print(f"  Distance : {d} km   Time : {t} min   Cost : Rs {c}")
        print(f"  Nodes expanded  Dijkstra: {exp_d}   A*: {exp_a}   "
              f"Verified optimal: {'YES' if ok else 'NO'}\n")

    print("-" * 74)
    print(f" {'Objective':<24}{'Distance(km)':>13}{'Time(min)':>11}"
          f"{'Cost(Rs)':>10}{'Hops':>6}")
    print("-" * 74)
    for name, r in results.items():
        d, t, c = r["totals"]
        print(f" {name:<24}{d:>13}{t:>11}{c:>10}{len(r['path']) - 1:>6}")
    print("-" * 74)

    short = results["Shortest distance"]["totals"]
    fast = results["Fastest (min time)"]["totals"]
    cheap = results["Cheapest (min cost)"]["totals"]
    print(f"\n Fastest vs shortest-distance route : "
          f"time {fast[1] - short[1]:+.1f} min, cost {fast[2] - short[2]:+.1f} Rs")
    print(f" Cheapest vs fastest route          : "
          f"time {cheap[1] - fast[1]:+.1f} min, cost {cheap[2] - fast[2]:+.1f} Rs")

    # Sensitivity: how the balanced route changes with the weight given to time
    print("\n Sensitivity of the balanced route to the time weight (alpha)")
    print("-" * 74)
    print(f" {'alpha':<7}{'Time(min)':>10}{'Cost(Rs)':>10}  Route")
    print("-" * 74)
    for a in (0.0, 0.1, 0.3, 0.5, 0.7, 1.0):
        w, h = make_objectives(args.dest, a)["Balanced (time + cost)"]
        p, _, _ = a_star(graph, args.source, args.dest, w, h)
        d, t, c = route_totals(graph, p)
        short_names = " > ".join(n.replace("Junction-", "J") for n in p)
        print(f" {a:<7}{t:>10}{c:>10}  {short_names}")
    print("-" * 74)

    if args.plot:
        plot_routes(graph, results, args.source, args.dest, args.plot)
        print(f" Route comparison figure saved to {args.plot}")
    if args.network_plot:
        plot_network(graph, args.network_plot)
        print(f" Road network figure saved to {args.network_plot}")


if __name__ == "__main__":
    main()
