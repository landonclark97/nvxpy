"""Classic Mixed-Integer Programming (MIP) examples using nvxpy.

This module demonstrates how nvxpy's expressive syntax makes it easy to
formulate well-known combinatorial optimization problems. Each example
includes a clear mathematical formulation alongside the code.

Run this module directly to execute all examples, or import individual
functions to experiment interactively.
"""

from __future__ import annotations

import autograd.numpy as np
import networkx as nx

import nvxpy as nvx
from nvxpy import Variable, Problem, Minimize, Maximize, BNB, Graph, DiGraph


# =============================================================================
# Knapsack Problem
# =============================================================================

def knapsack_problem():
    """
    0/1 Knapsack Problem
    --------------------
    Given items with weights and values, select items to maximize
    total value without exceeding the knapsack capacity.

    Formulation:
        maximize    sum(v[i] * x[i] for all items i)
        subject to  sum(w[i] * x[i] for all items i) <= capacity
                    x[i] in {0, 1}  (binary: take or don't take)

    Example: A thief with a knapsack of capacity 15 kg chooses among
    items with different weights and values.
    """
    print("=" * 60)
    print("0/1 KNAPSACK PROBLEM")
    print("=" * 60)

    # Problem data
    items = ["Gold Bar", "Silver Coins", "Diamond", "Painting", "Watch"]
    values = [10, 6, 14, 7, 3]    # Value of each item
    weights = [5, 3, 7, 4, 2]     # Weight of each item
    capacity = 15                  # Knapsack capacity

    n = len(items)
    nvx.reset_variable_ids()

    # Decision variables: x[i] = 1 if we take item i
    x = [Variable(integer=True, name=f"x_{items[i]}") for i in range(n)]
    for var in x:
        var.value = 0.0  # Initial guess

    # Objective: maximize total value
    total_value = sum(values[i] * x[i] for i in range(n))

    # Constraint: total weight <= capacity
    total_weight = sum(weights[i] * x[i] for i in range(n))

    problem = Problem(
        Maximize(total_value),
        [
            total_weight <= capacity,
            *[xi ^ [0, 1] for xi in x],  # Binary constraints
        ]
    )

    result = problem.solve(solver=BNB)

    print("\nItems available (value, weight):")
    for i, item in enumerate(items):
        print(f"  {item}: value={values[i]}, weight={weights[i]}")
    print(f"\nKnapsack capacity: {capacity}")
    print("\nOptimal selection:")

    selected_value = 0
    selected_weight = 0
    for i, item in enumerate(items):
        if x[i].value.item() > 0.5:
            print(f"  [X] {item}")
            selected_value += values[i]
            selected_weight += weights[i]
        else:
            print(f"  [ ] {item}")

    print(f"\nTotal value: {selected_value}")
    print(f"Total weight: {selected_weight}/{capacity}")
    print(f"Status: {result.status}")

    return result


# =============================================================================
# Maximum Independent Set
# =============================================================================

def maximum_independent_set():
    """
    Maximum Independent Set Problem
    --------------------------------
    Find the largest set of vertices in a graph such that no two
    vertices in the set are adjacent (connected by an edge).

    Formulation:
        maximize    sum(x[i] for all vertices i)
        subject to  x[i] + x[j] <= 1   for all edges (i, j)
                    x[i] in {0, 1}

    Example: A social network where we want to find the largest group
    of people who don't know each other (for diverse focus groups).

    This example uses nvx.Graph for cleaner constraint generation.
    """
    print("\n" + "=" * 60)
    print("MAXIMUM INDEPENDENT SET")
    print("=" * 60)

    # Build social network graph
    people = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
    nxg = nx.Graph()
    nxg.add_nodes_from(range(len(people)))
    nxg.add_edges_from([
        (0, 1),  # Alice knows Bob
        (0, 2),  # Alice knows Carol
        (1, 2),  # Bob knows Carol
        (1, 3),  # Bob knows Dave
        (2, 4),  # Carol knows Eve
        (3, 4),  # Dave knows Eve
        (4, 5),  # Eve knows Frank
    ])

    nvx.reset_variable_ids()

    # Wrap with nvx.Graph for clean constraint generation
    G = Graph(nxg)
    y = G.node_vars(binary=True, name_prefix="x")

    # Objective: maximize the number of people selected
    set_size = nvx.sum([y[i] for i in G.nodes])

    # Constraints: no two adjacent vertices can both be selected
    # G.independent(y) generates: y[i] + y[j] <= 1 for all edges
    constraints = G.independent(y)
    for var in y.values():
        constraints.extend(var.constraints)

    problem = Problem(Maximize(set_size), constraints)
    result = problem.solve(solver=BNB)

    print("\nSocial network connections:")
    for (i, j) in G.edges:
        print(f"  {people[i]} -- {people[j]}")

    print("\nMaximum independent set (strangers who can form a focus group):")
    selected = [people[i] for i in G.nodes if y[i].value.item() > 0.5]
    print(f"  {', '.join(selected)}")
    print(f"\nSet size: {len(selected)}")
    print(f"Status: {result.status}")

    return result


# =============================================================================
# Minimum Vertex Cover
# =============================================================================

def minimum_vertex_cover():
    """
    Minimum Vertex Cover Problem
    ----------------------------
    Find the smallest set of vertices such that every edge in the
    graph has at least one endpoint in the set.

    Formulation:
        minimize    sum(x[i] for all vertices i)
        subject to  x[i] + x[j] >= 1   for all edges (i, j)
                    x[i] in {0, 1}

    Example: A network of roads where we want to place the minimum
    number of security cameras to monitor all roads.

    This example uses nvx.Graph for cleaner constraint generation.
    """
    print("\n" + "=" * 60)
    print("MINIMUM VERTEX COVER")
    print("=" * 60)

    # Build road network graph
    intersections = ["A", "B", "C", "D", "E"]
    nxg = nx.Graph()
    nxg.add_nodes_from(range(len(intersections)))
    nxg.add_edges_from([
        (0, 1),  # Road A-B
        (0, 2),  # Road A-C
        (1, 2),  # Road B-C
        (1, 3),  # Road B-D
        (2, 3),  # Road C-D
        (3, 4),  # Road D-E
    ])

    nvx.reset_variable_ids()

    # Wrap with nvx.Graph
    G = Graph(nxg)
    y = G.node_vars(binary=True, name_prefix="x")

    # Objective: minimize number of cameras
    num_cameras = nvx.sum([y[i] for i in G.nodes])

    # Constraints: every road must be covered by at least one camera
    # For vertex cover: y[i] + y[j] >= 1 for all edges (i, j)
    constraints = []
    for (i, j) in G.edges:
        constraints.append(y[i] + y[j] >= 1)

    for var in y.values():
        constraints.extend(var.constraints)

    problem = Problem(Minimize(num_cameras), constraints)
    result = problem.solve(solver=BNB)

    print("\nRoad network:")
    for (i, j) in G.edges:
        print(f"  {intersections[i]} -- {intersections[j]}")

    print("\nOptimal camera placement:")
    cameras = [intersections[i] for i in G.nodes if y[i].value.item() > 0.5]
    print(f"  Cameras at: {', '.join(cameras)}")
    print(f"\nNumber of cameras needed: {len(cameras)}")
    print(f"Status: {result.status}")

    return result


# =============================================================================
# Set Cover Problem
# =============================================================================

def set_cover_problem():
    """
    Set Cover Problem
    -----------------
    Given a universe of elements and a collection of sets, find the
    minimum number of sets that cover all elements.

    Formulation:
        minimize    sum(x[j] for all sets j)
        subject to  sum(x[j] for j where element i is in set j) >= 1
                    x[j] in {0, 1}

    Example: A city wants to place fire stations to cover all
    neighborhoods, where each station covers certain neighborhoods.
    """
    print("\n" + "=" * 60)
    print("SET COVER PROBLEM")
    print("=" * 60)

    # Neighborhoods to cover
    neighborhoods = ["Downtown", "Uptown", "Riverside", "Hills", "Garden", "Industrial"]

    # Possible fire station locations and which neighborhoods they cover
    stations = {
        "Station_Central": [0, 1, 2],        # Covers Downtown, Uptown, Riverside
        "Station_North": [1, 3],             # Covers Uptown, Hills
        "Station_East": [2, 4, 5],           # Covers Riverside, Garden, Industrial
        "Station_West": [0, 3, 4],           # Covers Downtown, Hills, Garden
        "Station_South": [4, 5],             # Covers Garden, Industrial
    }

    station_names = list(stations.keys())
    n_neighborhoods = len(neighborhoods)
    n_stations = len(stations)

    nvx.reset_variable_ids()

    # Decision variables: x[j] = 1 if we build station j
    x = [Variable(integer=True, name=f"x_{station_names[j]}") for j in range(n_stations)]
    for var in x:
        var.value = 0.0

    # Objective: minimize number of stations
    num_stations = sum(x[j] for j in range(n_stations))

    # Constraints: every neighborhood must be covered
    coverage_constraints = []
    for i in range(n_neighborhoods):
        # Sum of stations that cover neighborhood i must be >= 1
        covering_stations = sum(
            x[j] for j in range(n_stations)
            if i in stations[station_names[j]]
        )
        coverage_constraints.append(covering_stations >= 1)

    problem = Problem(
        Minimize(num_stations),
        [
            *coverage_constraints,
            *[xi ^ [0, 1] for xi in x],
        ]
    )

    result = problem.solve(solver=BNB)

    print(f"\nNeighborhoods: {', '.join(neighborhoods)}")
    print("\nStation coverage:")
    for name, covered in stations.items():
        covered_names = [neighborhoods[i] for i in covered]
        print(f"  {name}: {', '.join(covered_names)}")

    print("\nOptimal station placement:")
    selected = [station_names[j] for j in range(n_stations) if x[j].value.item() > 0.5]
    for station in selected:
        print(f"  [X] {station}")

    print(f"\nNumber of stations needed: {len(selected)}")
    print(f"Status: {result.status}")

    return result


# =============================================================================
# Graph Coloring (Chromatic Number)
# =============================================================================

def graph_coloring():
    """
    Graph Coloring Problem
    ----------------------
    Assign colors to vertices such that no two adjacent vertices
    have the same color, using the minimum number of colors.

    Formulation:
        minimize    k (number of colors used)
        subject to  sum(x[i,c] for c in colors) = 1     (each vertex gets one color)
                    x[i,c] + x[j,c] <= 1                (adjacent vertices differ)
                    x[i,c] <= y[c]                      (color c used if any vertex has it)
                    x[i,c], y[c] in {0, 1}

    Example: Scheduling exams where conflicting courses (shared students)
    must be at different times, minimizing time slots needed.
    """
    print("\n" + "=" * 60)
    print("GRAPH COLORING (EXAM SCHEDULING)")
    print("=" * 60)

    # Courses and conflicts (shared students)
    courses = ["Math", "Physics", "Chemistry", "Biology", "CompSci"]
    conflicts = [
        (0, 1),  # Math-Physics conflict
        (0, 4),  # Math-CompSci conflict
        (1, 2),  # Physics-Chemistry conflict
        (1, 4),  # Physics-CompSci conflict
        (2, 3),  # Chemistry-Biology conflict
    ]

    n = len(courses)
    max_colors = n  # Upper bound on colors needed

    nvx.reset_variable_ids()

    # x[i][c] = 1 if course i is scheduled in slot c
    x = [[Variable(integer=True, name=f"x_{courses[i]}_slot{c}")
          for c in range(max_colors)] for i in range(n)]

    # y[c] = 1 if slot c is used
    y = [Variable(integer=True, name=f"y_slot{c}") for c in range(max_colors)]

    for i in range(n):
        for c in range(max_colors):
            x[i][c].value = 1.0 if c == i else 0.0
    for c in range(max_colors):
        y[c].value = 1.0 if c < 3 else 0.0

    # Objective: minimize number of time slots
    num_slots = sum(y[c] for c in range(max_colors))

    constraints = []

    # Each course gets exactly one slot
    for i in range(n):
        constraints.append(sum(x[i][c] for c in range(max_colors)) == 1)

    # Conflicting courses cannot share a slot
    for (i, j) in conflicts:
        for c in range(max_colors):
            constraints.append(x[i][c] + x[j][c] <= 1)

    # Link x and y: if course i uses slot c, then slot c is used
    for i in range(n):
        for c in range(max_colors):
            constraints.append(x[i][c] <= y[c])

    # Symmetry breaking: use slots in order
    for c in range(max_colors - 1):
        constraints.append(y[c] >= y[c + 1])

    # Binary constraints
    for i in range(n):
        for c in range(max_colors):
            constraints.append(x[i][c] ^ [0, 1])
    for c in range(max_colors):
        constraints.append(y[c] ^ [0, 1])

    problem = Problem(Minimize(num_slots), constraints)
    result = problem.solve(solver=BNB)

    print(f"\nCourses: {', '.join(courses)}")
    print("\nConflicts (shared students):")
    for (i, j) in conflicts:
        print(f"  {courses[i]} -- {courses[j]}")

    print("\nOptimal exam schedule:")
    for c in range(max_colors):
        if y[c].value.item() > 0.5:
            slot_courses = [courses[i] for i in range(n) if x[i][c].value.item() > 0.5]
            if slot_courses:
                print(f"  Time Slot {c + 1}: {', '.join(slot_courses)}")

    num_used = sum(1 for c in range(max_colors) if y[c].value.item() > 0.5)
    print(f"\nMinimum time slots needed: {num_used}")
    print(f"Status: {result.status}")

    return result


# =============================================================================
# Facility Location Problem
# =============================================================================

def facility_location():
    """
    Uncapacitated Facility Location Problem
    ----------------------------------------
    Decide which facilities to open and assign customers to facilities
    to minimize total cost (fixed opening costs + transportation costs).

    Formulation:
        minimize    sum(f[j] * y[j]) + sum(c[i,j] * x[i,j])
        subject to  sum(x[i,j] for j) = 1           (each customer assigned once)
                    x[i,j] <= y[j]                   (can only use open facilities)
                    x[i,j], y[j] in {0, 1}

    Example: A company deciding where to build warehouses to serve
    retail stores, balancing construction costs vs shipping costs.
    """
    print("\n" + "=" * 60)
    print("FACILITY LOCATION PROBLEM")
    print("=" * 60)

    # Potential warehouse locations and fixed costs
    warehouses = ["Chicago", "Denver", "Atlanta"]
    fixed_costs = [100, 80, 90]  # Cost to open each warehouse

    # Retail stores
    stores = ["NYC", "LA", "Miami", "Seattle"]

    # Transportation costs from warehouse j to store i
    transport_costs = [
        [10, 40, 25, 45],  # Chicago to each store
        [30, 15, 35, 20],  # Denver to each store
        [20, 35, 10, 50],  # Atlanta to each store
    ]

    n_stores = len(stores)
    n_warehouses = len(warehouses)

    nvx.reset_variable_ids()

    # y[j] = 1 if warehouse j is opened
    y = [Variable(integer=True, name=f"y_{warehouses[j]}") for j in range(n_warehouses)]

    # x[i][j] = 1 if store i is served by warehouse j
    x = [[Variable(integer=True, name=f"x_{stores[i]}_{warehouses[j]}")
          for j in range(n_warehouses)] for i in range(n_stores)]

    # Initial values
    for j in range(n_warehouses):
        y[j].value = 1.0
    for i in range(n_stores):
        for j in range(n_warehouses):
            x[i][j].value = 1.0 / n_warehouses

    # Objective: minimize total cost
    fixed_cost = sum(fixed_costs[j] * y[j] for j in range(n_warehouses))
    transport_cost = sum(
        transport_costs[j][i] * x[i][j]
        for i in range(n_stores)
        for j in range(n_warehouses)
    )
    total_cost = fixed_cost + transport_cost

    constraints = []

    # Each store must be assigned to exactly one warehouse
    for i in range(n_stores):
        constraints.append(sum(x[i][j] for j in range(n_warehouses)) == 1)

    # Can only assign to open warehouses
    for i in range(n_stores):
        for j in range(n_warehouses):
            constraints.append(x[i][j] <= y[j])

    # Binary constraints
    for j in range(n_warehouses):
        constraints.append(y[j] ^ [0, 1])
    for i in range(n_stores):
        for j in range(n_warehouses):
            constraints.append(x[i][j] ^ [0, 1])

    problem = Problem(Minimize(total_cost), constraints)
    result = problem.solve(solver=BNB)

    print("\nWarehouses (fixed cost):")
    for j, wh in enumerate(warehouses):
        print(f"  {wh}: ${fixed_costs[j]}")

    print("\nTransportation costs (warehouse -> store):")
    print(f"  {'':12} " + "  ".join(f"{s:8}" for s in stores))
    for j, wh in enumerate(warehouses):
        costs = "  ".join(f"${c:<7}" for c in transport_costs[j])
        print(f"  {wh:12} {costs}")

    print("\nOptimal solution:")
    print("  Open warehouses:")
    for j, wh in enumerate(warehouses):
        if y[j].value.item() > 0.5:
            print(f"    [X] {wh}")
        else:
            print(f"    [ ] {wh}")

    print("\n  Store assignments:")
    for i, store in enumerate(stores):
        for j, wh in enumerate(warehouses):
            if x[i][j].value.item() > 0.5:
                print(f"    {store} <- {wh} (cost: ${transport_costs[j][i]})")

    # Calculate total cost
    total_fixed = sum(fixed_costs[j] for j in range(n_warehouses) if y[j].value.item() > 0.5)
    total_transport = sum(
        transport_costs[j][i]
        for i in range(n_stores)
        for j in range(n_warehouses)
        if x[i][j].value.item() > 0.5
    )
    print(f"\n  Fixed costs: ${total_fixed}")
    print(f"  Transport costs: ${total_transport}")
    print(f"  Total cost: ${total_fixed + total_transport}")
    print(f"Status: {result.status}")

    return result


# =============================================================================
# Traveling Salesman Problem (Small Instance)
# =============================================================================

def traveling_salesman():
    """
    Traveling Salesman Problem (TSP)
    --------------------------------
    Find the shortest tour visiting all cities exactly once and
    returning to the starting city.

    Formulation (MTZ formulation):
        minimize    sum(d[i,j] * x[i,j])
        subject to  sum(x[i,j] for j) = 1       (leave each city once)
                    sum(x[i,j] for i) = 1       (enter each city once)
                    u[i] - u[j] + n*x[i,j] <= n-1  (subtour elimination)
                    x[i,j] in {0, 1}

    Example: A delivery driver planning the shortest route through cities.

    This example uses nvx.DiGraph with in_degree/out_degree constraints.
    """
    print("\n" + "=" * 60)
    print("TRAVELING SALESMAN PROBLEM")
    print("=" * 60)

    # Cities and coordinates
    cities = ["Start", "A", "B", "C", "D"]
    coords = [(0, 0), (1, 5), (5, 2), (6, 6), (3, 3)]

    n = len(cities)

    # Distance matrix
    def euclidean_dist(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Build complete directed graph with distances as weights
    nxg = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = euclidean_dist(coords[i], coords[j])
                nxg.add_edge(i, j, weight=dist)

    nvx.reset_variable_ids()

    # Wrap with nvx.DiGraph
    G = DiGraph(nxg)
    x = G.edge_vars(binary=True, name_prefix="x")

    # Initialize with nearest-neighbor heuristic for better solver performance
    def nearest_neighbor_tour():
        """Generate a greedy tour starting from node 0."""
        visited = {0}
        tour = [0]
        current = 0
        while len(visited) < n:
            best_next = None
            best_dist = float('inf')
            for j in range(n):
                if j not in visited:
                    dist = nxg[current][j]['weight']
                    if dist < best_dist:
                        best_dist = dist
                        best_next = j
            tour.append(best_next)
            visited.add(best_next)
            current = best_next
        return tour

    nn_tour = nearest_neighbor_tour()
    for var in x.values():
        var.value = np.array([0.0])
    for i in range(len(nn_tour)):
        from_node = nn_tour[i]
        to_node = nn_tour[(i + 1) % n]
        x[(from_node, to_node)].value = np.array([1.0])

    # u[i] = position of city i in the tour (for subtour elimination)
    u = [Variable(name=f"u_{i}") for i in range(n)]
    for i, node in enumerate(nn_tour):
        u[node].value = float(i)

    # Objective: minimize total distance using G.total_weight
    total_distance = G.total_weight(x)

    constraints = []

    # Each city must be left exactly once: out_degree == 1
    constraints.extend(G.out_degree(x) == 1)

    # Each city must be entered exactly once: in_degree == 1
    constraints.extend(G.in_degree(x) == 1)

    # MTZ subtour elimination constraints
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                constraints.append(u[i] - u[j] + n * x[i, j] <= n - 1)

    # Position bounds
    constraints.append(u[0] == 0)  # Start city is position 0
    for i in range(1, n):
        constraints.append(u[i] >= 1)
        constraints.append(u[i] <= n - 1)

    # Binary constraints from edge_vars
    for var in x.values():
        constraints.extend(var.constraints)

    problem = Problem(Minimize(total_distance), constraints)
    result = problem.solve(solver=BNB, compile=True)

    # Build distance matrix for display
    distances = [[euclidean_dist(coords[i], coords[j]) for j in range(n)] for i in range(n)]

    print("\nCities and coordinates:")
    for i, city in enumerate(cities):
        print(f"  {city}: {coords[i]}")

    print("\nDistance matrix:")
    print(f"  {'':6}" + "".join(f"{c:>8}" for c in cities))
    for i, city in enumerate(cities):
        row = "".join(f"{distances[i][j]:8.2f}" for j in range(n))
        print(f"  {city:6}{row}")

    # Reconstruct tour
    print("\nOptimal tour:")
    tour = [0]  # Start at city 0
    current = 0
    for _ in range(n - 1):
        for j in range(n):
            if j != current and (current, j) in x and x[current, j].value.item() > 0.5:
                tour.append(j)
                current = j
                break
    tour.append(0)  # Return to start

    tour_str = " -> ".join(cities[i] for i in tour)
    print(f"  {tour_str}")

    total_dist = sum(distances[tour[i]][tour[i+1]] for i in range(len(tour) - 1))
    print(f"\nTotal distance: {total_dist:.2f}")
    print(f"Status: {result.status}")

    return result


# =============================================================================
# Bin Packing Problem
# =============================================================================

def bin_packing():
    """
    Bin Packing Problem
    -------------------
    Pack items of different sizes into the minimum number of bins,
    where each bin has a fixed capacity.

    Formulation:
        minimize    sum(y[j] for all bins j)
        subject to  sum(s[i] * x[i,j] for i) <= C * y[j]  (capacity)
                    sum(x[i,j] for j) = 1                  (each item in one bin)
                    x[i,j], y[j] in {0, 1}

    Example: Packing files onto disks or boxes into trucks.
    """
    print("\n" + "=" * 60)
    print("BIN PACKING PROBLEM")
    print("=" * 60)

    # Items and their sizes
    items = ["File_A", "File_B", "File_C", "File_D", "File_E", "File_F"]
    sizes = [4, 8, 5, 1, 7, 3]  # Size of each item
    capacity = 10               # Bin capacity

    n_items = len(items)
    n_bins = n_items  # Upper bound: one item per bin

    nvx.reset_variable_ids()

    # y[j] = 1 if bin j is used
    y = [Variable(integer=True, name=f"y_bin{j}") for j in range(n_bins)]

    # x[i][j] = 1 if item i is placed in bin j
    x = [[Variable(integer=True, name=f"x_{items[i]}_bin{j}")
          for j in range(n_bins)] for i in range(n_items)]

    # Initialize
    for j in range(n_bins):
        y[j].value = 1.0 if j < 3 else 0.0
    for i in range(n_items):
        for j in range(n_bins):
            x[i][j].value = 1.0 if i == j else 0.0

    # Objective: minimize number of bins used
    num_bins = sum(y[j] for j in range(n_bins))

    constraints = []

    # Each item must be placed in exactly one bin
    for i in range(n_items):
        constraints.append(sum(x[i][j] for j in range(n_bins)) == 1)

    # Bin capacity constraint
    for j in range(n_bins):
        constraints.append(sum(sizes[i] * x[i][j] for i in range(n_items)) <= capacity * y[j])

    # Symmetry breaking: use bins in order
    for j in range(n_bins - 1):
        constraints.append(y[j] >= y[j + 1])

    # Binary constraints
    for j in range(n_bins):
        constraints.append(y[j] ^ [0, 1])
    for i in range(n_items):
        for j in range(n_bins):
            constraints.append(x[i][j] ^ [0, 1])

    problem = Problem(Minimize(num_bins), constraints)
    result = problem.solve(solver=BNB)

    print("\nItems and sizes:")
    for i, item in enumerate(items):
        print(f"  {item}: size {sizes[i]}")
    print(f"\nBin capacity: {capacity}")
    print(f"Total item size: {sum(sizes)}")
    print(f"Lower bound on bins: {(sum(sizes) + capacity - 1) // capacity}")

    print("\nOptimal packing:")
    for j in range(n_bins):
        if y[j].value.item() > 0.5:
            bin_items = [items[i] for i in range(n_items) if x[i][j].value.item() > 0.5]
            bin_size = sum(sizes[i] for i in range(n_items) if x[i][j].value.item() > 0.5)
            print(f"  Bin {j + 1}: {', '.join(bin_items)} (total: {bin_size}/{capacity})")

    num_used = sum(1 for j in range(n_bins) if y[j].value.item() > 0.5)
    print(f"\nMinimum bins needed: {num_used}")
    print(f"Status: {result.status}")

    return result


# =============================================================================
# Assignment Problem
# =============================================================================

def assignment_problem():
    """
    Assignment Problem
    ------------------
    Assign n workers to n tasks to minimize total cost, where each
    worker is assigned exactly one task and each task to one worker.

    Formulation:
        minimize    sum(c[i,j] * x[i,j])
        subject to  sum(x[i,j] for j) = 1   (each worker gets one task)
                    sum(x[i,j] for i) = 1   (each task gets one worker)
                    x[i,j] in {0, 1}

    Example: Assigning employees to projects based on their skill costs.
    """
    print("\n" + "=" * 60)
    print("ASSIGNMENT PROBLEM")
    print("=" * 60)

    workers = ["Alice", "Bob", "Carol", "Dave"]
    tasks = ["Frontend", "Backend", "Database", "Testing"]

    # Cost matrix: cost for worker i to do task j (based on skill/time)
    costs = [
        [9, 2, 7, 8],   # Alice
        [6, 4, 3, 7],   # Bob
        [5, 8, 1, 8],   # Carol
        [7, 6, 9, 4],   # Dave
    ]

    n = len(workers)

    nvx.reset_variable_ids()

    # x[i][j] = 1 if worker i is assigned to task j
    x = [[Variable(integer=True, name=f"x_{workers[i]}_{tasks[j]}")
          for j in range(n)] for i in range(n)]

    for i in range(n):
        for j in range(n):
            x[i][j].value = 1.0 if i == j else 0.0

    # Objective: minimize total cost
    total_cost = sum(costs[i][j] * x[i][j] for i in range(n) for j in range(n))

    constraints = []

    # Each worker assigned to exactly one task
    for i in range(n):
        constraints.append(sum(x[i][j] for j in range(n)) == 1)

    # Each task assigned to exactly one worker
    for j in range(n):
        constraints.append(sum(x[i][j] for i in range(n)) == 1)

    # Binary constraints
    for i in range(n):
        for j in range(n):
            constraints.append(x[i][j] ^ [0, 1])

    problem = Problem(Minimize(total_cost), constraints)
    result = problem.solve(solver=BNB)

    print("\nCost matrix (worker -> task):")
    print(f"  {'':8}" + "".join(f"{t:>10}" for t in tasks))
    for i, worker in enumerate(workers):
        row = "".join(f"{costs[i][j]:>10}" for j in range(n))
        print(f"  {worker:8}{row}")

    print("\nOptimal assignment:")
    total = 0
    for i, worker in enumerate(workers):
        for j, task in enumerate(tasks):
            if x[i][j].value.item() > 0.5:
                print(f"  {worker} -> {task} (cost: {costs[i][j]})")
                total += costs[i][j]

    print(f"\nTotal cost: {total}")
    print(f"Status: {result.status}")

    return result


# =============================================================================
# Run All Examples
# =============================================================================

ALL_EXAMPLES = [
    knapsack_problem,
    maximum_independent_set,
    minimum_vertex_cover,
    set_cover_problem,
    graph_coloring,
    facility_location,
    traveling_salesman,
    bin_packing,
    assignment_problem,
]


def run_all_examples():
    """Run all MIP example problems."""
    print("\n" + "#" * 60)
    print("# CLASSIC MIP PROBLEMS WITH nvxpy")
    print("#" * 60)

    for example in ALL_EXAMPLES:
        try:
            example()
        except Exception as e:
            print(f"\nExample {example.__name__} failed: {e}")
        print()


if __name__ == "__main__":
    run_all_examples()
