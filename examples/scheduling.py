import numpy as np
import networkx as nx

import nvxpy as nvx


NUM_NODES = 4
NUM_TIME_SLOTS = 3

MAX_TIME = 50.0
MIN_ALL = 0.5
MAX_WORK = 3.0
MAX_TRAVEL = 3.0

# Building random graph to emulate a scheduling task where a technician
# must travel to a location and begin work within a set of time ranges.
nx_G = nx.complete_graph(NUM_NODES)
slots = []
for n in nx_G.nodes():
    slot_times = np.random.uniform(0.0, MAX_TIME, (NUM_TIME_SLOTS * 2,))
    sorted_slot_times = np.sort(slot_times)

    nx_G.nodes[n]["slots"] = [
        (sorted_slot_times[i * 2], sorted_slot_times[i * 2 + 1])
        for i in range(NUM_TIME_SLOTS)
    ]
    nx_G.nodes[n]["work"] = np.random.uniform(MIN_ALL, MAX_WORK)

    print(f"node: {n} with slots: {nx_G.nodes[n]['slots']}")

nx_G.add_node("depot")
for n in nx_G.nodes():
    if n != "depot":
        nx_G.add_edge("depot", n)

for u, v in nx_G.edges():
    nx_G[u][v]["travel"] = np.random.uniform(MIN_ALL, MAX_TRAVEL)

nx_dir_G = nx_G.to_directed()

G = nvx.DiGraph(nx_dir_G)

phis = G.edge_vars(integer=True)
X_ij = G.edge_vars(binary=True)

arrival_times = {}
wait_times = {}
start_times = {}
departure_times = {}
constraints = []
for n in G.nodes:
    if n == "depot":
        arrival_times[n] = 0.0
        wait_times[n] = 0.0
        start_times[n] = 0.0
        departure_times[n] = 0.0
    else:
        arrival_times[n] = nvx.Variable(pos=True)
        wait_times[n] = nvx.Variable(pos=True)
        start_times[n] = arrival_times[n] + wait_times[n]
        departure_times[n] = start_times[n] + G.nodes[n]["work"]
        # Upper bounds for MIP solver
        constraints.append(arrival_times[n] <= MAX_TIME)
        constraints.append(wait_times[n] <= MAX_TIME)

    constraints.append(G.in_degree(X_ij, n) == 1)
    constraints.append(G.out_degree(X_ij, n) == 1)

    in_flow = sum(phis[i, n] for (i, j) in phis.keys() if j == n)
    out_flow = sum(phis[n, j] for (i, j) in phis.keys() if i == n)
    if n == "depot":
        constraints.append(out_flow - in_flow == NUM_NODES)
    else:
        constraints.append(in_flow - out_flow == 1)
        constraints.append(start_times[n] ^ G.nodes[n]["slots"])

for u, v in G.edges:
    e = (u, v)
    constraints.append(phis[e] >= 0.0)
    constraints.append(phis[e] <= NUM_NODES)  # Upper bound for MIP solver
    constraints.append(phis[e] <= NUM_NODES * X_ij[e])

    # Skip edges going TO depot (no arrival time constraint needed for return)
    if v != "depot":
        constraints.append(
            arrival_times[v]
            >= departure_times[u] + G.edges[e]["travel"] - (MAX_TIME * (1.0 - X_ij[e]))
        )
        constraints.append(
            arrival_times[v]
            <= departure_times[u]
            + G.edges[e]["travel"]
            + 1e-3
            + (MAX_TIME * (1.0 - X_ij[e]))
        )

travel_cost = sum(X_ij[e] * G.edges[e]["travel"] for e in G.edges())
wait_cost = sum(wait_times[n] for n in G.nodes)

obj = travel_cost + wait_cost

prob = nvx.Problem(nvx.Minimize(obj), constraints, compile=True)

# First solve with BnB to get a feasible initial solution
print("Finding feasible solution with BnB...")
result = prob.solve(
    solver=nvx.BNB,
    solver_options={
        "bb_max_nodes": 300,
        "bb_verbose": True,
        "bb_method": "hybrid",
    },
)

print(f"Result status: {result.status}")
print(f"Objective: {obj.value.item():.4f}")
print()


def safe_val(v):
    if isinstance(v, (int, float)):
        return v
    return v.value.item() if hasattr(v.value, "item") else float(v.value)


# Build tour from flow values
phis_sorted = sorted(
    [(e, phis[e].value.item()) for e in G.edges], key=lambda x: x[1], reverse=True
)
tour = ["depot"]
current = "depot"
visited = {"depot"}
for _ in range(NUM_NODES):
    for e, phi in phis_sorted:
        if e[0] == current and phi > 0.5 and e[1] not in visited:
            tour.append(e[1])
            visited.add(e[1])
            current = e[1]
            break
tour.append("depot")

print("=" * 70)
print("TOUR")
print("=" * 70)
print(" -> ".join(str(n) for n in tour))
print()

print("=" * 70)
print("SCHEDULE")
print("=" * 70)
print(
    f"{'Node':<8} {'Arrival':<10} {'Wait':<10} {'Start':<10} {'Work':<10} {'Depart':<10}"
)
print("-" * 70)

total_travel = 0.0
total_wait = 0.0
prev = None
for n in tour[:-1]:  # Skip final depot
    arr = safe_val(arrival_times[n])
    wait = safe_val(wait_times[n])
    start = safe_val(start_times[n])
    dep = safe_val(departure_times[n])

    if n == "depot":
        print(
            f"{str(n):<8} {arr:<10.2f} {wait:<10.2f} {start:<10.2f} {'-':<10} {dep:<10.2f}"
        )
    else:
        work = nx_dir_G.nodes[n]["work"]
        total_wait += wait
        print(
            f"{str(n):<8} {arr:<10.2f} {wait:<10.2f} {start:<10.2f} {work:<10.2f} {dep:<10.2f}"
        )

        # Show which slot was used
        slots = nx_dir_G.nodes[n]["slots"]
        for i, (s, e) in enumerate(slots):
            if s - 0.01 <= start <= e + 0.01:
                print(f"         ^ slot {i}: [{s:.2f}, {e:.2f}]")
                break

    if prev is not None:
        travel = nx_dir_G[prev][n]["travel"]
        total_travel += travel
    prev = n

# Add travel back to depot
if prev is not None and prev != "depot":
    total_travel += nx_dir_G[prev]["depot"]["travel"]

print("-" * 70)
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total travel time: {total_travel:.4f}")
print(f"Total wait time:   {total_wait:.4f}")
print(f"Objective value:   {safe_val(obj):.4f}")
