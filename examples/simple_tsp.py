"""
Simple TSP - Traveling Salesman Problem (cycle)

Find the shortest Hamiltonian cycle visiting all nodes.
"""

import numpy as np
import nvxpy as nvx
from nvxpy.variable import Variable, reset_variable_ids
from nvxpy.problem import Problem, Minimize
import networkx as nx

np.random.seed(42)

NUM_NODES = 5

# Build directed complete graph
G = nx.complete_graph(NUM_NODES, create_using=nx.DiGraph())

# Remove self-loops
for i in range(NUM_NODES):
    if G.has_edge(i, i):
        G.remove_edge(i, i)

# Assign random edge weights (travel costs)
for i in range(NUM_NODES):
    for j in range(NUM_NODES):
        if i != j:
            # Symmetric costs
            if i < j:
                weight = np.random.uniform(1.0, 5.0)
                G[i][j]["weight"] = weight
                G[j][i]["weight"] = weight

print("Edge weights:")
for i, j, d in G.edges(data=True):
    if i < j:
        print(f"  {i} <-> {j}: {d['weight']:.2f}")

# ============================================================
# OPTIMIZATION MODEL
# ============================================================
reset_variable_ids()

nvx_G = nvx.DiGraph(G)

# Binary edge variables
edge_vars = nvx_G.edge_vars(binary=True, name_prefix="x")
# for e in edge_vars.values():
#     e.value = 0.5

# Position variables for MTZ subtour elimination
position = {}
for n in G.nodes():
    position[n] = Variable(name=f"pos_{n}")
    position[n].value = np.array([float(n)])

# --- Constraints ---
constraints = []

# Each node has exactly one incoming and one outgoing edge
constraints.extend(nvx_G.in_degree(edge_vars) == 1)
constraints.extend(nvx_G.out_degree(edge_vars) == 1)

# MTZ subtour elimination
BIG_M = NUM_NODES + 1
for (i, j), x_ij in edge_vars.items():
    if i != 0 and j != 0:
        constraints.append(position[j] >= position[i] + 1 - BIG_M * (1 - x_ij))

# Position bounds
for n in G.nodes():
    constraints.append(position[n] >= 0)
    constraints.append(position[n] <= NUM_NODES - 1)

# Fix starting node position
constraints.append(position[0] == 0)

# --- Objective: minimize total travel ---
objective = nvx_G.total_weight(edge_vars)

print(f"\nProblem: {NUM_NODES} nodes, {len(edge_vars)} edge variables")
print("Solving...")

prob = Problem(Minimize(objective), constraints, compile=True)
result = prob.solve(solver=nvx.BNB, solver_options={"outlev": 6})

print(f"\nResult status: {result.status}")

if result.status in [nvx.SolverStatus.OPTIMAL, nvx.SolverStatus.SUBOPTIMAL]:
    # Extract tour
    tour = [0]
    current = 0
    visited = {0}

    for _ in range(NUM_NODES - 1):
        for (i, j), x_ij in edge_vars.items():
            if i == current and x_ij.value > 0.5 and j not in visited:
                tour.append(j)
                visited.add(j)
                current = j
                break

    # Find return edge
    for (i, j), x_ij in edge_vars.items():
        if i == current and j == 0 and x_ij.value > 0.5:
            tour.append(0)
            break

    print(f"Tour: {' -> '.join(map(str, tour))}")

    # Compute cost
    total_cost = sum(
        G[i][j]["weight"] * edge_vars[i, j].value for (i, j) in edge_vars.keys()
    )
    print(f"Total cost: {total_cost.item():.2f}")
