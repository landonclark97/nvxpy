"""Graph constructs for MIP formulations using networkx."""

from __future__ import annotations

from typing import Any, Iterator


from ..variable import Variable
from ..constraint import Constraint
from ..expression import Expr


class EdgeVars:
    """Dict-like container for edge decision variables."""

    def __init__(self, graph: "BaseGraph", variables: dict, binary: bool = False):
        self._graph = graph
        self._vars = variables
        self._binary = binary

    def __getitem__(self, key: tuple) -> Variable:
        return self._vars[key]

    def __iter__(self) -> Iterator[tuple]:
        return iter(self._vars)

    def __len__(self) -> int:
        return len(self._vars)

    def items(self):
        return self._vars.items()

    def keys(self):
        return self._vars.keys()

    def values(self):
        return self._vars.values()

    @property
    def is_binary(self) -> bool:
        return self._binary


class NodeVars:
    """Dict-like container for node decision variables."""

    def __init__(self, graph: "BaseGraph", variables: dict, binary: bool = False):
        self._graph = graph
        self._vars = variables
        self._binary = binary

    def __getitem__(self, key) -> Variable:
        return self._vars[key]

    def __iter__(self) -> Iterator:
        return iter(self._vars)

    def __len__(self) -> int:
        return len(self._vars)

    def items(self):
        return self._vars.items()

    def keys(self):
        return self._vars.keys()

    def values(self):
        return self._vars.values()

    @property
    def is_binary(self) -> bool:
        return self._binary


class DegreeExpr:
    """Expression representing degree constraints.

    Supports comparison operators to create constraints:
        G.degree(x) == 2
        G.degree(x) <= 3
        G.in_degree(x) >= 1
    """

    def __init__(self, graph: "BaseGraph", edge_vars: EdgeVars, node: Any = None,
                 degree_type: str = "degree"):
        self._graph = graph
        self._edge_vars = edge_vars
        self._node = node
        self._degree_type = degree_type

    def _get_degree_expr(self, node) -> Expr:
        """Build the sum expression for degree at a node."""
        terms = []

        if self._degree_type == "degree":
            # For undirected: sum of all edges incident to node
            for u, v in self._edge_vars.keys():
                if u == node or v == node:
                    terms.append(self._edge_vars[u, v])
        elif self._degree_type == "in_degree":
            # For directed: sum of edges coming into node
            for u, v in self._edge_vars.keys():
                if v == node:
                    terms.append(self._edge_vars[u, v])
        elif self._degree_type == "out_degree":
            # For directed: sum of edges going out of node
            for u, v in self._edge_vars.keys():
                if u == node:
                    terms.append(self._edge_vars[u, v])
        else:
            raise ValueError(f"Unknown degree type: {self._degree_type}")

        if not terms:
            return 0

        result = terms[0]
        for t in terms[1:]:
            result = result + t
        return result

    def _build_constraints(self, op: str, value) -> list[Constraint]:
        """Build constraints for all nodes or single node."""
        if self._node is not None:
            expr = self._get_degree_expr(self._node)
            return [Constraint(expr, op, value)]

        # Apply to all nodes
        constraints = []
        for node in self._graph._nx_graph.nodes():
            expr = self._get_degree_expr(node)
            constraints.append(Constraint(expr, op, value))
        return constraints

    def __eq__(self, other) -> list[Constraint]:
        return self._build_constraints("==", other)

    def __le__(self, other) -> list[Constraint]:
        return self._build_constraints("<=", other)

    def __ge__(self, other) -> list[Constraint]:
        return self._build_constraints(">=", other)

    def __getitem__(self, node) -> "DegreeExpr":
        """Get degree expression for a specific node."""
        return DegreeExpr(self._graph, self._edge_vars, node, self._degree_type)


class BaseGraph:
    """Base class for Graph and DiGraph."""

    def __init__(self, nx_graph, weight_attr: str = "weight"):
        """
        Args:
            nx_graph: A networkx graph (Graph or DiGraph)
            weight_attr: Edge attribute name for weights (default: "weight")
        """
        self._nx_graph = nx_graph
        self._weight_attr = weight_attr

    @property
    def nodes(self):
        return self._nx_graph.nodes()

    @property
    def edges(self):
        return self._nx_graph.edges()

    def edge_vars(self, binary: bool = False, integer: bool = False,
                  name_prefix: str = "e") -> EdgeVars:
        """Create decision variables for each edge.

        Args:
            binary: If True, variables are binary (0 or 1)
            integer: If True, variables are integer-valued
            name_prefix: Prefix for variable names

        Returns:
            EdgeVars dict-like mapping (u, v) -> Variable
        """
        variables = {}
        for i, (u, v) in enumerate(self._nx_graph.edges()):
            var = Variable(
                shape=(1,),
                name=f"{name_prefix}_{u}_{v}",
                integer=binary or integer,
            )
            if binary:
                var.constraints.append(Constraint(var, ">=", 0))
                var.constraints.append(Constraint(var, "<=", 1))
            variables[(u, v)] = var
        return EdgeVars(self, variables, binary=binary)

    def node_vars(self, binary: bool = False, integer: bool = False,
                  name_prefix: str = "n") -> NodeVars:
        """Create decision variables for each node.

        Args:
            binary: If True, variables are binary (0 or 1)
            integer: If True, variables are integer-valued
            name_prefix: Prefix for variable names

        Returns:
            NodeVars dict-like mapping node -> Variable
        """
        variables = {}
        for node in self._nx_graph.nodes():
            var = Variable(
                shape=(1,),
                name=f"{name_prefix}_{node}",
                integer=binary or integer,
            )
            if binary:
                var.constraints.append(Constraint(var, ">=", 0))
                var.constraints.append(Constraint(var, "<=", 1))
            variables[node] = var
        return NodeVars(self, variables, binary=binary)

    def degree(self, edge_vars: EdgeVars, node=None) -> DegreeExpr:
        """Create degree expression(s) for nodes.

        Usage:
            G.degree(x) == 2        # All nodes have degree 2
            G.degree(x)[node] >= 1  # Specific node has degree >= 1

        Args:
            edge_vars: Edge variables from edge_vars()
            node: Optional specific node (if None, applies to all nodes)

        Returns:
            DegreeExpr that supports ==, <=, >= operators
        """
        return DegreeExpr(self, edge_vars, node, "degree")

    def total_weight(self, edge_vars: EdgeVars, weight_attr: str = None) -> Expr:
        """Sum of weights for selected edges.

        Args:
            edge_vars: Edge variables (typically binary selection)
            weight_attr: Edge attribute for weight (default: self._weight_attr)

        Returns:
            Expression: sum of weight[e] * edge_vars[e] for all edges
        """
        attr = weight_attr or self._weight_attr

        terms = []
        for (u, v), var in edge_vars.items():
            edge_data = self._nx_graph.get_edge_data(u, v)
            weight = edge_data.get(attr, 1.0) if edge_data else 1.0
            terms.append(weight * var)

        if not terms:
            return 0

        result = terms[0]
        for t in terms[1:]:
            result = result + t
        return result

    def covers(self, edge_vars: EdgeVars, node_vars: NodeVars) -> list[Constraint]:
        """Vertex cover constraints: each edge must be covered by at least one endpoint.

        For each edge (u, v): node_vars[u] + node_vars[v] >= edge_vars[u, v]
        (If edge is selected, at least one endpoint must be in the cover)

        For standard vertex cover where all edges must be covered:
            For each edge (u, v): node_vars[u] + node_vars[v] >= 1
        """
        constraints = []
        for (u, v), evar in edge_vars.items():
            constraints.append(Constraint(
                node_vars[u] + node_vars[v],
                ">=",
                evar
            ))
        return constraints

    def independent(self, node_vars: NodeVars) -> list[Constraint]:
        """Independent set constraints: no two adjacent nodes can both be selected.

        For each edge (u, v): node_vars[u] + node_vars[v] <= 1
        """
        constraints = []
        for u, v in self._nx_graph.edges():
            constraints.append(Constraint(
                node_vars[u] + node_vars[v],
                "<=",
                1
            ))
        return constraints

    def flow_conservation(self, edge_vars: EdgeVars, source, sink,
                          demand: float = 1.0) -> list[Constraint]:
        """Flow conservation constraints for network flow problems.

        - At source: outflow - inflow = demand
        - At sink: inflow - outflow = demand
        - At other nodes: inflow = outflow

        Args:
            edge_vars: Edge flow variables
            source: Source node
            sink: Sink node
            demand: Flow demand (default 1.0)
        """
        constraints = []

        for node in self._nx_graph.nodes():
            # Calculate inflow and outflow
            inflow_terms = []
            outflow_terms = []

            for (u, v), var in edge_vars.items():
                if v == node:
                    inflow_terms.append(var)
                if u == node:
                    outflow_terms.append(var)

            def sum_terms(terms):
                if not terms:
                    return 0
                result = terms[0]
                for t in terms[1:]:
                    result = result + t
                return result

            inflow = sum_terms(inflow_terms)
            outflow = sum_terms(outflow_terms)

            if node == source:
                constraints.append(Constraint(outflow - inflow, "==", demand))
            elif node == sink:
                constraints.append(Constraint(inflow - outflow, "==", demand))
            else:
                constraints.append(Constraint(inflow, "==", outflow))

        return constraints


class Graph(BaseGraph):
    """Undirected graph wrapper for MIP formulations.

    Example:
        import networkx as nx
        import nvxpy as nvx

        G = nvx.Graph(nx.complete_graph(5))
        x = G.edge_vars(binary=True)
        y = G.node_vars(binary=True)

        constraints = [
            *G.degree(x) == 2,      # Each node has degree 2
            *G.independent(y),       # Selected nodes form independent set
        ]

        prob = nvx.Problem(nvx.Minimize(G.total_weight(x)), constraints)
    """

    def __init__(self, nx_graph, weight_attr: str = "weight"):
        import networkx as nx
        if isinstance(nx_graph, nx.DiGraph):
            raise TypeError("Use DiGraph for directed graphs")
        super().__init__(nx_graph, weight_attr)


class DiGraph(BaseGraph):
    """Directed graph wrapper for MIP formulations.

    Example:
        import networkx as nx
        import nvxpy as nvx

        G = nvx.DiGraph(nx.DiGraph())
        G._nx_graph.add_weighted_edges_from([(0, 1, 5), (1, 2, 3)])

        x = G.edge_vars(binary=True)

        constraints = [
            *G.in_degree(x) == 1,   # Each node has in-degree 1
            *G.out_degree(x) == 1,  # Each node has out-degree 1
        ]
    """

    def __init__(self, nx_graph, weight_attr: str = "weight"):
        import networkx as nx
        if not isinstance(nx_graph, nx.DiGraph):
            raise TypeError("Use Graph for undirected graphs")
        super().__init__(nx_graph, weight_attr)

    def in_degree(self, edge_vars: EdgeVars, node=None) -> DegreeExpr:
        """Create in-degree expression(s) for nodes.

        Usage:
            G.in_degree(x) == 1        # All nodes have in-degree 1
            G.in_degree(x)[node] >= 1  # Specific node has in-degree >= 1
        """
        return DegreeExpr(self, edge_vars, node, "in_degree")

    def out_degree(self, edge_vars: EdgeVars, node=None) -> DegreeExpr:
        """Create out-degree expression(s) for nodes.

        Usage:
            G.out_degree(x) == 1        # All nodes have out-degree 1
            G.out_degree(x)[node] >= 1  # Specific node has out-degree >= 1
        """
        return DegreeExpr(self, edge_vars, node, "out_degree")
