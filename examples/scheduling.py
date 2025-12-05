"""
TSP with Time Windows - Job Scheduling Problem

Given a set of locations (jobs) with:
- Multiple valid time slots when work can START at each location
- Fixed work duration at each location
- Travel times between locations

Find a path that visits each location exactly once, minimizing:
    total_time = travel_time + work_time + waiting_time

Where waiting_time is the slack between arrival and start of work
(we can arrive early and wait for a valid time slot).

This version uses DiscreteRanges for time slot constraints instead of
Big-M formulations. The time slot selection is handled via branching:
    start_time ^ [[slot1_start, slot1_end], [slot2_start, slot2_end], ...]
"""

import numpy as np
import nvxpy as nvx
from nvxpy.variable import Variable, reset_variable_ids
from nvxpy.problem import Problem, Minimize
import networkx as nx

# Seed for reproducibility
np.random.seed(43)

# Problem parameters
MIN_TIME = 0.0
MAX_TIME = 100.0

MIN_TRAVEL_COST = 0.5
MAX_TRAVEL_COST = 2.0

MIN_WORK_TIME = 1.0
MAX_WORK_TIME = 3.0

NUM_NODES = 6  # Keep small for testing
NUM_SLOTS_PER_NODE = 3  # Each node has multiple possible time slots

# Big-M still needed for MTZ subtour elimination and time sequencing
BIG_M = MAX_TIME

# Build a sparse random directed graph (not complete)
# Keep generating until we get one with a Hamiltonian cycle starting from node 0
def has_hamiltonian_cycle_from_depot(G, depot=0):
    """Check if directed graph has a Hamiltonian cycle starting/ending at depot."""
    from itertools import permutations
    nodes = list(G.nodes())
    other_nodes = [n for n in nodes if n != depot]
    # Check all orderings of other nodes, with depot at start and end
    for perm in permutations(other_nodes):
        path = [depot] + list(perm) + [depot]  # cycle: depot -> ... -> depot
        if all(G.has_edge(path[i], path[i+1]) for i in range(len(path)-1)):
            return True
    return False

edge_probability = 0.5
graph_seed = 124  # Fixed seed for reproducibility
while True:
    G = nx.gnp_random_graph(NUM_NODES, edge_probability, seed=graph_seed)
    # Remove self-loops if any
    G.remove_edges_from(nx.selfloop_edges(G))
    if has_hamiltonian_cycle_from_depot(G, depot=0):
        break
    # Increase probability if we're having trouble finding one
    edge_probability = min(0.9, edge_probability + 0.05)
    graph_seed += 1
print("=" * 60)
print("TSP WITH TIME WINDOWS - JOB SCHEDULING PROBLEM")
print("=" * 60)
print(f"\nGraph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (sparse, not complete)")

# Print graph structure visually
print("\nGraph structure (node -> neighbors):")
for n in sorted(G.nodes()):
    neighbors = sorted(G.neighbors(n))
    if neighbors:
        neighbor_str = " --> ".join(str(nb) for nb in neighbors)
        print(f"  {n} --> {neighbor_str}")
    else:
        print(f"  {n} (no outgoing edges)")

# Assign time slots and work times to each node
# Generate random time windows: each window is 2-3x longer than MAX_WORK_TIME
print("\nNode time slots:")
WINDOW_MULTIPLIER = 2.5  # Windows are 2.5x longer than max work time
WINDOW_SIZE = MAX_WORK_TIME * WINDOW_MULTIPLIER

for n in G.nodes():
    # Random work time for this node
    work_time = np.random.uniform(MIN_WORK_TIME, MAX_WORK_TIME)
    G.nodes[n]['work_time'] = work_time

    # Generate random non-overlapping time slots
    time_slots = []
    available_start = 0.0

    for slot_idx in range(NUM_SLOTS_PER_NODE):
        # Random gap before this slot (0 to 20 time units)
        gap = np.random.uniform(0 if slot_idx == 0 else 5, 25)
        slot_start = available_start + gap

        # Window size varies a bit (2x to 3x work time)
        window_size = np.random.uniform(2.0, 3.0) * MAX_WORK_TIME

        slot_end = min(slot_start + window_size, MAX_TIME)
        time_slots.append((slot_start, slot_end))

        # Next slot starts after this one
        available_start = slot_end

    G.nodes[n]['time_slots'] = time_slots
    slots_str = ", ".join(f"[{s:.1f}, {e:.1f}]" for s, e in time_slots)
    print(f"  Node {n}: work={work_time:.1f}, slots=[{slots_str}]")

# Assign random travel times to each edge
for i, j in G.edges():
    G[i][j]['weight'] = np.random.uniform(MIN_TRAVEL_COST, MAX_TRAVEL_COST)

print("\nTravel times:")
for i, j, d in G.edges(data=True):
    print(f"  {i} -> {j}: {d['weight']:.2f}")

G = G.to_directed()

# ============================================================
# OPTIMIZATION MODEL - Using DiscreteRanges for Time Slots
# ============================================================
reset_variable_ids()

# Wrap graph for nvxpy
nvx_G = nvx.DiGraph(G)

# --- Decision Variables ---

# Binary edge variables: x[i,j] = 1 if we travel from i to j
edge_vars = nvx_G.edge_vars(binary=True, name_prefix="x")

# Initialize edge variables
for e in edge_vars.values():
    e.value = 0.0

# Continuous time variables for each node
arrival_time = {}   # When we arrive at node i
start_time = {}     # When we start work at node i (must be in a time slot)
wait_time = {}      # Waiting time before starting work at node i

for n in G.nodes():
    arrival_time[n] = Variable(name=f"arrival_{n}")
    arrival_time[n].value = 0.0

    start_time[n] = Variable(name=f"start_{n}")
    # Initialize to beginning of first slot (to prefer earlier starts)
    first_slot = G.nodes[n]['time_slots'][0]
    start_time[n].value = first_slot[0]

    wait_time[n] = Variable(pos=True, name=f"wait_{n}")  # wait >= 0
    wait_time[n].value = 0.0

# NO binary slot selection variables needed - DiscreteRanges handles this!

# Flow variables for subtour elimination (Single-Commodity Flow formulation)
# Instead of MTZ position variables, we use a flow-based approach:
# - Node 0 (depot) supplies (n-1) units of commodity
# - Each other node consumes 1 unit
# - Flow only travels on active edges: f[i,j] <= (n-1) * x[i,j]
# This prevents subtours because a subtour not containing the depot has no supply.
flow_vars = nvx_G.edge_vars(integer=True, name_prefix="f")
for f in flow_vars.values():
    f.value = 0.0

# --- Constraints ---
constraints = []

# 1. TSP constraints: Hamiltonian cycle (visit each node exactly once and return)
#    - Each node has exactly one incoming edge
#    - Each node has exactly one outgoing edge
constraints.extend(nvx_G.in_degree(edge_vars) == 1)
constraints.extend(nvx_G.out_degree(edge_vars) == 1)

# 2. Subtour elimination (Single-Commodity Flow formulation)
#    Flow conservation: for each node, inflow - outflow = demand
#    - Node 0 (depot): supplies n-1 units, so outflow - inflow = n-1
#    - Other nodes: consume 1 unit, so inflow - outflow = 1
for n in G.nodes():
    inflow = sum(flow_vars[i, n] for (i, j) in flow_vars.keys() if j == n)
    outflow = sum(flow_vars[n, j] for (i, j) in flow_vars.keys() if i == n)
    if n == 0:
        constraints.append(outflow - inflow == NUM_NODES - 1)
    else:
        constraints.append(inflow - outflow == 1)

# Flow capacity: flow only on active edges, bounded by (n-1)
for (i, j), x_ij in edge_vars.items():
    constraints.append(flow_vars[i, j] >= 0)
    constraints.append(flow_vars[i, j] <= (NUM_NODES - 1) * x_ij)

# 3. START TIME IN TIME SLOTS - Using DiscreteRanges!
#    Instead of:
#      - Binary slot selection variables z[n][k]
#      - Constraint: sum(z[n]) == 1
#      - Big-M constraints for each slot
#    We simply say: start_time[n] must be in one of the slot ranges
print("\n=== Using DiscreteRanges for time slot constraints ===")
for n in G.nodes():
    slots = [list(slot) for slot in G.nodes[n]['time_slots']]  # [[lb, ub], ...]
    print(f"  Node {n}: start_time ^ {slots}")
    constraints.append(start_time[n] ^ slots)

# 4. Wait time links arrival and start: start_time = arrival_time + wait_time
for n in G.nodes():
    constraints.append(start_time[n] == arrival_time[n] + wait_time[n])

# Fix arrival time at start node to 0
constraints.append(arrival_time[0] == 0)


# 5. Time sequencing: if we go from i to j, arrival at j = departure from i + travel
#    departure_time[i] = start_time[i] + work_time[i]
#    When edge (i,j) is active: arrival_time[j] == departure_time[i] + travel_time[i,j]
#    This is enforced via Big-M from both sides:
#      arrival_time[j] >= departure_time[i] + travel_time[i,j] - BIG_M * (1 - x[i,j])
#      arrival_time[j] <= departure_time[i] + travel_time[i,j] + BIG_M * (1 - x[i,j])
#    Skip edges returning to node 0 (we've fixed arrival_time[0] = 0 as the start)
for (i, j), x_ij in edge_vars.items():
    if j == 0:  # Skip return edges - no time constraint for returning to depot
        continue
    work_i = G.nodes[i]['work_time']
    travel_ij = G[i][j]['weight']
    departure_i = start_time[i] + work_i

    # Lower bound: arrival >= departure + travel when edge is active
    constraints.append(arrival_time[j] >= departure_i + travel_ij - BIG_M * (1 - x_ij))
    # Upper bound: arrival <= departure + travel when edge is active
    constraints.append(arrival_time[j] <= departure_i + travel_ij + BIG_M * (1 - x_ij))

# 6. Bound constraints on times
for n in G.nodes():
    constraints.append(arrival_time[n] >= MIN_TIME)
    constraints.append(arrival_time[n] <= MAX_TIME)  # Allow some overflow for long paths

# --- Objective ---
# Minimize total time = travel_time + work_time + waiting_time
# travel_time = sum of selected edge weights
# work_time = sum of all work times (constant, but include for completeness)
# waiting_time[n] = start_time[n] - arrival_time[n]

# Total travel time
total_travel = nvx_G.total_weight(edge_vars)

# Total work time (constant)
total_work = sum(G.nodes[n]['work_time'] for n in G.nodes())

# Total waiting time (using explicit wait variables)
total_waiting = sum(w_time for w_time in wait_time.values())

# The total_waiting term already minimizes total wait.
# The problem is the gap between slots forces waiting somewhere.
# We add a larger penalty to prefer starting as early as possible.
# This pushes starts to the beginning of their selected time slots.
early_start_bonus = 0.1 * sum(s for s in start_time.values())

# Total objective
objective = total_travel + total_waiting + early_start_bonus

print(f"\nTotal work time (constant): {total_work:.2f}")
print("\nSolving optimization problem...")

# --- Solve ---
prob = Problem(Minimize(objective), constraints)
result = prob.solve(solver=nvx.BNB, compile=True, solver_options={
    # B&B options - more stringent for better solution quality
    "bb_verbose": True,
    "bb_max_time": 300,           # Allow more time
    "bb_max_nodes": 50000,        # Explore more nodes
    "bb_abs_gap": 1e-8,           # Tighter absolute gap
    "bb_rel_gap": 1e-6,           # Tighter relative gap
    "bb_int_tol": 1e-7,           # Stricter integer tolerance
    # Use IPOPT for tighter NLP solutions
    "nlp_backend": "ipopt",
    "nlp_options": {
        "tol": 1e-10,
        "acceptable_tol": 1e-9,
        "max_iter": 3000,
    },
})

print(f"\nResult status: {result.status}")

if result.status in [nvx.SolverStatus.OPTIMAL, nvx.SolverStatus.SUBOPTIMAL]:
    print("\n=== SOLUTION ===")

    # Extract path
    path = [0]  # Start at node 0
    current = 0
    visited = {0}

    for _ in range(NUM_NODES - 1):
        for (i, j), x_ij in edge_vars.items():
            if i == current and x_ij.value > 0.5 and j not in visited:
                path.append(j)
                visited.add(j)
                current = j
                break

    print(f"\nPath: {' -> '.join(map(str, path))}")

    # Print flow variables for active edges
    print("\nFlow variables (active edges only):")
    for (i, j), x_ij in edge_vars.items():
        if x_ij.value > 0.5:
            f_val = flow_vars[i, j].value
            print(f"  {i} -> {j}: x={x_ij.value:.2f}, flow={f_val:.2f}")

    print("\nSchedule:")
    print("-" * 90)
    print(f"{'Node':<6} {'Travel':<10} {'Arrival':<10} {'Start':<10} {'Wait':<10} {'Work':<10} {'Depart':<10} {'Slot'}")
    print("-" * 90)

    total_wait_actual = 0
    prev_node = None
    for n in path:
        arr = arrival_time[n].value
        st = start_time[n].value
        wait = max(0.0, wait_time[n].value)  # Use actual wait_time variable
        work = G.nodes[n]['work_time']
        depart = st + work
        total_wait_actual += wait

        # Travel time from previous node
        if prev_node is None:
            travel = 0.0
            travel_str = "-"
        else:
            travel = G[prev_node][n]['weight']
            travel_str = f"{travel:.2f}"

        # Find which slot the start_time falls in
        slot_info = "?"
        for k, (s, e) in enumerate(G.nodes[n]['time_slots']):
            if s - 0.01 <= st <= e + 0.01:
                slot_info = f"slot {k}: ({s:.1f}, {e:.1f})"
                break

        print(f"{n:<6} {travel_str:<10} {arr:<10.2f} {st:<10.2f} {wait:<10.2f} {work:<10.2f} {depart:<10.2f}  {slot_info}")
        prev_node = n

    print("-" * 90)

    # Compute actual totals
    total_travel_actual = total_travel.value

    print(f"\nTotal travel time: {total_travel_actual:.2f}")
    print(f"Total work time:   {total_work:.2f}")
    print(f"Total wait time:   {total_wait_actual:.2f}")
    print(f"Total time:        {total_travel_actual + total_work + total_wait_actual:.2f}")

    # Debug: compute objective breakdown
    total_start = sum(start_time[n].value for n in G.nodes())
    print("\nObjective breakdown:")
    print(f"  travel:           {total_travel_actual:.4f}")
    print(f"  wait:             {total_wait_actual:.4f}")
    print(f"  start_time:       {total_start:.4f}")
    print(f"  TOTAL:            {objective.value:.4f}")
