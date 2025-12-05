"""Tests for Graph and DiGraph constructs."""

import networkx as nx
import autograd.numpy as np
import pytest

import nvxpy as nvx
from nvxpy.constructs.graph import EdgeVars, NodeVars, DegreeExpr


class TestGraphBasics:
    """Basic Graph creation and variable generation tests."""

    def test_graph_creation(self):
        G = nvx.Graph(nx.complete_graph(4))
        assert len(list(G.nodes)) == 4
        assert len(list(G.edges)) == 6  # C(4,2) = 6 edges

    def test_graph_rejects_digraph(self):
        with pytest.raises(TypeError):
            nvx.Graph(nx.DiGraph())

    def test_edge_vars_creation(self):
        G = nvx.Graph(nx.complete_graph(3))
        x = G.edge_vars(binary=True)

        assert isinstance(x, EdgeVars)
        assert len(x) == 3  # 3 edges in K3
        assert x.is_binary

        for (u, v), var in x.items():
            assert var.is_integer  # binary implies integer
            assert len(var.constraints) == 2  # >= 0 and <= 1

    def test_node_vars_creation(self):
        G = nvx.Graph(nx.path_graph(5))
        y = G.node_vars(binary=True)

        assert isinstance(y, NodeVars)
        assert len(y) == 5
        assert y.is_binary

        for node, var in y.items():
            assert var.is_integer

    def test_edge_vars_integer(self):
        G = nvx.Graph(nx.complete_graph(3))
        x = G.edge_vars(integer=True)

        assert len(x) == 3
        for (u, v), var in x.items():
            assert var.is_integer
            assert len(var.constraints) == 0  # no bounds for integer

    def test_edge_vars_continuous(self):
        G = nvx.Graph(nx.complete_graph(3))
        x = G.edge_vars()

        for (u, v), var in x.items():
            assert not var.is_integer


class TestDiGraphBasics:
    """Basic DiGraph creation and variable generation tests."""

    def test_digraph_creation(self):
        G = nvx.DiGraph(nx.DiGraph([(0, 1), (1, 2), (2, 0)]))
        assert len(list(G.nodes)) == 3
        assert len(list(G.edges)) == 3

    def test_digraph_rejects_graph(self):
        with pytest.raises(TypeError):
            nvx.DiGraph(nx.Graph())

    def test_in_degree(self):
        G = nvx.DiGraph(nx.DiGraph([(0, 1), (0, 2), (1, 2)]))
        x = G.edge_vars(binary=True)

        in_deg = G.in_degree(x)
        assert isinstance(in_deg, DegreeExpr)

        # in_degree constraints for all nodes
        constraints = in_deg == 1
        assert len(constraints) == 3

    def test_out_degree(self):
        G = nvx.DiGraph(nx.DiGraph([(0, 1), (0, 2), (1, 2)]))
        x = G.edge_vars(binary=True)

        out_deg = G.out_degree(x)
        constraints = out_deg >= 0
        assert len(constraints) == 3


class TestDegreeConstraints:
    """Test degree constraint generation."""

    def test_degree_all_nodes(self):
        G = nvx.Graph(nx.complete_graph(4))
        x = G.edge_vars(binary=True)

        constraints = G.degree(x) == 2
        assert len(constraints) == 4  # one per node

        for c in constraints:
            assert c.op == "=="

    def test_degree_single_node(self):
        G = nvx.Graph(nx.complete_graph(4))
        x = G.edge_vars(binary=True)

        constraints = G.degree(x)[0] >= 1
        assert len(constraints) == 1

    def test_degree_le(self):
        G = nvx.Graph(nx.path_graph(3))
        x = G.edge_vars(binary=True)

        constraints = G.degree(x) <= 2
        assert len(constraints) == 3
        for c in constraints:
            assert c.op == "<="


class TestGraphConstraintHelpers:
    """Test covers, independent, flow_conservation."""

    def test_independent_set(self):
        G = nvx.Graph(nx.complete_graph(4))
        y = G.node_vars(binary=True)

        constraints = G.independent(y)
        assert len(constraints) == 6  # one per edge in K4

        for c in constraints:
            assert c.op == "<="

    def test_vertex_cover(self):
        G = nvx.Graph(nx.path_graph(4))
        x = G.edge_vars(binary=True)
        y = G.node_vars(binary=True)

        # Set all edges to be selected
        for var in x.values():
            var.value = np.array([1.0])

        constraints = G.covers(x, y)
        assert len(constraints) == 3  # 3 edges in path graph

    def test_flow_conservation(self):
        # Simple directed path: 0 -> 1 -> 2
        G = nvx.DiGraph(nx.DiGraph([(0, 1), (1, 2)]))
        x = G.edge_vars()

        constraints = G.flow_conservation(x, source=0, sink=2, demand=1.0)
        assert len(constraints) == 3  # one per node


class TestTotalWeight:
    """Test total_weight objective."""

    def test_total_weight_default(self):
        nxg = nx.Graph()
        nxg.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 4.0)])
        G = nvx.Graph(nxg)

        x = G.edge_vars(binary=True)
        for var in x.values():
            var.value = np.array([1.0])

        obj = G.total_weight(x)
        assert obj.value == 12.0  # 5 + 3 + 4

    def test_total_weight_custom_attr(self):
        nxg = nx.Graph()
        nxg.add_edge(0, 1, cost=10.0)
        nxg.add_edge(1, 2, cost=20.0)
        G = nvx.Graph(nxg, weight_attr="cost")

        x = G.edge_vars(binary=True)
        for var in x.values():
            var.value = np.array([1.0])

        obj = G.total_weight(x)
        assert obj.value == 30.0


class TestGraphSolving:
    """Integration tests solving actual graph problems."""

    def test_minimum_vertex_cover(self):
        """Solve minimum vertex cover on a path graph."""
        nvx.reset_variable_ids()

        # Path graph: 0 - 1 - 2 - 3
        G = nvx.Graph(nx.path_graph(4))
        y = G.node_vars(binary=True, name_prefix="v")

        # Minimize number of vertices while covering all edges
        constraints = []

        # Each edge must be covered by at least one endpoint
        for u, v in G.edges:
            constraints.append(y[u] + y[v] >= 1)

        # Collect all variable constraints
        for var in y.values():
            constraints.extend(var.constraints)

        prob = nvx.Problem(
            nvx.Minimize(nvx.sum([var for var in y.values()])),
            constraints
        )

        result = prob.solve(solver=nvx.BNB)
        assert result.status == nvx.SolverStatus.OPTIMAL

        # Optimal vertex cover for path of 4: nodes 1 and 2 (or similar)
        selected = sum(1 for var in y.values() if var.value > 0.5)
        assert selected == 2

    def test_maximum_independent_set(self):
        """Solve maximum independent set on a small graph."""
        nvx.reset_variable_ids()

        # Star graph: center 0 connected to 1, 2, 3
        nxg = nx.star_graph(3)
        G = nvx.Graph(nxg)
        y = G.node_vars(binary=True, name_prefix="s")

        constraints = G.independent(y)
        for var in y.values():
            constraints.extend(var.constraints)

        prob = nvx.Problem(
            nvx.Maximize(nvx.sum([var for var in y.values()])),
            constraints
        )

        result = prob.solve(solver=nvx.BNB)
        assert result.status == nvx.SolverStatus.OPTIMAL

        # Max independent set in star graph: all leaves (3) or just center (1)
        # Leaves win: 3 > 1
        selected = sum(1 for var in y.values() if var.value > 0.5)
        assert selected == 3

    def test_degree_constrained_subgraph(self):
        """Find subgraph where each node has degree exactly 2 (cycle)."""
        nvx.reset_variable_ids()

        G = nvx.Graph(nx.complete_graph(4))
        x = G.edge_vars(binary=True, name_prefix="x")

        constraints = G.degree(x) == 2
        for var in x.values():
            constraints.extend(var.constraints)

        prob = nvx.Problem(
            nvx.Minimize(G.total_weight(x)),
            constraints
        )

        result = prob.solve(solver=nvx.BNB)
        assert result.status == nvx.SolverStatus.OPTIMAL

        # Should select exactly 4 edges (Hamiltonian cycle in K4)
        selected = sum(1 for var in x.values() if var.value > 0.5)
        assert selected == 4
