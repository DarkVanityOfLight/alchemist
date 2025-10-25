from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple
from expressions import (
    Argument, IRNode, Identifier, ProductDomain, VectorSpace, FiniteSet,
    UnionSet, IntersectionSet, ComplementSet,
    LinearScale, Shift, SetComprehension, SymbolicSet, Vector, UnionSpace
)
from collections import defaultdict, deque
from dataclasses import dataclass, replace
import logging

from guards import SetGuard, SimpleGuard

if TYPE_CHECKING:
    from scope_handler import ScopeHandler

logger = logging.getLogger(__name__)

# FIXME: Annotation information may be lost if a node is eliminated from the tree
# Consider pushing annotations to child nodes instead


# -----------------------------------------------------------------------------
# Core traversal utilities
# -----------------------------------------------------------------------------

def get_semantic_children(node: IRNode, scopes: ScopeHandler) -> Tuple[IRNode, ...]:
    """
    Get the semantic children of a node, resolving identifiers through scopes.
    
    For Identifiers, returns the resolved definition. For SetComprehensions,
    returns domain and optional guard set expression.
    
    Args:
        node: The IR node to get children from
        scopes: Scope handler for resolving identifiers
        
    Returns:
        Tuple of semantic child nodes
    """
    if isinstance(node, Identifier):
        resolved = scopes.lookup_by_id(node.ptr)
        return (resolved,) if resolved else ()
    
    if isinstance(node, SetComprehension):
        children: List[IRNode] = [node.domain]
        if isinstance(node.guard, SetGuard) and node.guard.set_expr is not None:  # type: ignore
            children.append(node.guard.set_expr)  # type: ignore
        return tuple(children)
    
    return node.children


def get_raw_children(node: IRNode) -> List[IRNode]:
    """
    Get the actual IR children that should be processed for replacement.
    
    Unlike get_semantic_children, this doesn't resolve identifiers.
    Used for transformations that need to replace child nodes.
    
    Args:
        node: The IR node to get children from
        
    Returns:
        List of child IR nodes
    """
    if isinstance(node, SetComprehension):
        children = [node.domain]
        if isinstance(node.guard, SetGuard) and node.guard.set_expr is not None:
            children.append(node.guard.set_expr)
        return children  # type: ignore
    return list(node.children)


def fold_ir(
    node: IRNode,
    scopes: ScopeHandler,
    fn: Callable[[IRNode, Tuple[Any, ...]], Any],
    memo: Optional[Dict[int, Any]] = None
) -> Any:
    """
    Fold (catamorphism) over the IR tree with memoization.
    
    Bottom-up traversal that applies a function to each node along with
    the results from its children.
    
    Args:
        node: Root node to fold over
        scopes: Scope handler for resolving identifiers
        fn: Function taking (node, child_results) and returning a result
        memo: Memoization dictionary to avoid reprocessing nodes
        
    Returns:
        Result of applying fn across the entire tree
    """
    memo = memo if memo is not None else {}
    key = id(node)
    
    if key in memo:
        return memo[key]
    
    # Recursively process children first
    child_vals = tuple(fold_ir(ch, scopes, fn, memo) for ch in get_semantic_children(node, scopes))
    
    # Apply function to this node with child results
    result = fn(node, child_vals)
    memo[key] = result
    return result


def map_ir(
    node: IRNode,
    scopes: ScopeHandler,
    fn: Callable[[IRNode, List[IRNode]], IRNode],
    memo: Optional[Dict[int, IRNode]] = None
) -> IRNode:
    """
    Map (functorial map) over the IR tree with memoization.
    
    Bottom-up traversal that transforms each node based on its transformed children.
    
    Args:
        node: Root node to map over
        scopes: Scope handler for resolving identifiers
        fn: Function taking (node, transformed_children) and returning new node
        memo: Memoization dictionary to avoid reprocessing nodes
        
    Returns:
        Transformed IR tree
    """
    memo = memo if memo is not None else {}
    key = id(node)
    
    if key in memo:
        return memo[key]
    
    # Get semantic children and recursively transform them
    kids = get_semantic_children(node, scopes)
    new_kids = [map_ir(ch, scopes, fn, memo) for ch in kids]
    
    # Apply function to create new node
    result = fn(node, new_kids)
    memo[key] = result
    return result


# -----------------------------------------------------------------------------
# Reconstruction logic
# -----------------------------------------------------------------------------

def reconstruct_node(node: IRNode, new_children: List[IRNode]) -> IRNode:
    """
    Reconstruct a node with new children, preserving node type and structure.
    
    Handles the dataclass replace operation for each node type, ensuring
    that the correct fields are updated with the new children.
    
    Args:
        node: Original node to reconstruct
        new_children: New child nodes to use
        
    Returns:
        New node with updated children
    """
    if not new_children:
        return node
    
    # Handle each node type's specific reconstruction
    if isinstance(node, SetComprehension):
        domain, *rest = new_children
        guard = node.guard
        if isinstance(guard, SetGuard) and guard.set_expr is not None and rest:
            guard = replace(guard, set_expr=rest[0])
        return replace(node, domain=domain, guard=guard)
    
    if isinstance(node, VectorSpace):
        return replace(node, basis=tuple(new_children))
    
    if isinstance(node, FiniteSet):
        return replace(node, members=frozenset(new_children))
    
    if isinstance(node, UnionSet):
        return replace(node, parts=tuple(new_children))
    
    if isinstance(node, IntersectionSet):
        return replace(node, parts=tuple(new_children))
    
    if isinstance(node, ComplementSet):
        return replace(node, complemented_set=new_children[0])
    
    if isinstance(node, SetGuard) and node.set_expr is not None:
        return replace(node, set_expr=new_children[0])
    
    if isinstance(node, Annotated):
        return replace(node, base=new_children[0])
    
    logger.debug("No reconstruction for %s", type(node).__name__)
    return node


# -----------------------------------------------------------------------------
# Identifier inlining
# -----------------------------------------------------------------------------

def build_dependency_graph(
    root: IRNode,
    scopes: ScopeHandler
) -> Tuple[Dict[int, IRNode], Dict[int, Set[int]]]:
    """
    Build a dependency graph for identifier definitions.
    
    Traverses the IR tree to find all identifier references, looks up their
    definitions, and builds a graph showing which definitions depend on which.
    
    Args:
        root: Root IR node to start traversal from
        scopes: Scope handler for looking up definitions
        
    Returns:
        Tuple of (definitions dict, dependency graph)
        - definitions: Maps identifier ptr to its definition node
        - graph: Maps identifier ptr to set of ptrs it depends on
    """
    definitions = {}
    graph = defaultdict(set)
    visited = set()
    
    def collect_definitions(node: IRNode):
        """Recursively collect all identifier definitions."""
        if id(node) in visited:
            return
        visited.add(id(node))
        
        if isinstance(node, Identifier):
            ptr = node.ptr
            if ptr not in definitions:
                def_node = scopes.lookup_by_id(ptr)
                if def_node:
                    definitions[ptr] = def_node
                    # Recursively collect definitions from the definition itself
                    collect_definitions(def_node)
        else:
            for child in node.children:
                collect_definitions(child)
    
    collect_definitions(root)
    
    # Build dependency edges: for each definition, find what it depends on
    for ptr, def_node in definitions.items():
        dependencies = collect_identifier_pointers(def_node)
        # Only include dependencies that are also definitions
        graph[ptr] = {dep for dep in dependencies if dep in definitions}
    
    return definitions, graph


def collect_identifier_pointers(node: IRNode) -> Set[int]:
    """
    Collect all identifier pointers referenced in a node tree.
    
    Uses raw traversal (doesn't follow semantic links) to find all
    Identifier nodes and extract their pointer values.
    
    Args:
        node: Root node to collect from
        
    Returns:
        Set of all identifier pointers found
    """
    pointers = set()
    
    if isinstance(node, Identifier):
        pointers.add(node.ptr)
    
    for child in node.children:
        pointers.update(collect_identifier_pointers(child))
    
    return pointers


def topological_sort(graph: Dict[int, Set[int]]) -> List[int]:
    """
    Perform topological sort on the dependency graph.
    
    Returns identifiers in an order where each identifier comes before
    any identifiers that depend on it. This ensures we can inline
    definitions in the correct order.
    
    Args:
        graph: Dependency graph mapping nodes to their dependencies
        
    Returns:
        List of node IDs in topological order
    """
    # Calculate in-degrees (number of dependencies)
    in_degree = {node: 0 for node in graph}
    for node, dependencies in graph.items():
        for dep in dependencies:
            if dep in in_degree:
                in_degree[node] += 1
    
    # Start with nodes that have no dependencies
    queue = deque([node for node, degree in in_degree.items() if degree == 0])
    result = []
    
    # Process nodes in order, removing edges as we go
    while queue:
        node = queue.popleft()
        result.append(node)
        
        # For each node that depends on this one, decrement its in-degree
        for other_node, dependencies in graph.items():
            if node in dependencies:
                in_degree[other_node] -= 1
                if in_degree[other_node] == 0:
                    queue.append(other_node)
    
    # If we didn't process all nodes, there's a cycle
    if len(result) != len(graph):
        logger.debug("Circular dependencies detected in identifier graph")
    
    return result


def inline_all_identifiers(root: IRNode, scopes: ScopeHandler) -> IRNode:
    """
    Inline all identifier references by replacing them with their definitions.
    
    Two-pass algorithm:
    1. Build complete substitution map in dependency order
    2. Apply all substitutions to root
    
    This ensures that when we inline a definition, all of its dependencies
    have already been inlined.
    
    Args:
        root: Root IR node to inline identifiers in
        scopes: Scope handler for looking up definitions
        
    Returns:
        New IR tree with all identifiers inlined
    """
    definitions, dependency_graph = build_dependency_graph(root, scopes)
    
    if not definitions:
        logger.debug("No definitions found for inlining")
        return root
    
    logger.debug("Found %d definitions to inline", len(definitions))
    for ptr, def_node in definitions.items():
        logger.debug("Definition %d: %s", ptr, type(def_node).__name__)
    
    # Sort definitions in dependency order
    sorted_identifiers = topological_sort(dependency_graph)
    logger.debug("Topological order: %s", sorted_identifiers)
    
    # Build substitution map by processing definitions in order
    substitution_map: Dict[int, IRNode] = {}
    
    for identifier_ptr in sorted_identifiers:
        if identifier_ptr in definitions:
            original_definition = definitions[identifier_ptr]
            # Inline any identifiers within this definition
            inlined_definition = replace_identifiers(original_definition, substitution_map)
            substitution_map[identifier_ptr] = inlined_definition
            logger.debug("Inlined %d -> %s", identifier_ptr, type(inlined_definition).__name__)
    
    logger.debug("Built substitution map with %d entries", len(substitution_map))
    
    # Apply all substitutions to the root
    result = replace_identifiers(root, substitution_map)
    
    # Check for any remaining unresolved identifiers
    remaining = collect_identifier_pointers(result)
    if remaining:
        logger.debug("Unresolved identifiers after inlining: %s", remaining)
    
    return result


def replace_identifiers(node: IRNode, substitution_map: Dict[int, IRNode]) -> IRNode:
    """
    Replace identifier nodes with their substitutions from the map.
    
    Recursively traverses the tree, replacing any Identifier whose ptr
    is in the substitution map with the corresponding node.
    
    Args:
        node: Node to perform replacements in
        substitution_map: Map from identifier ptr to replacement node
        
    Returns:
        New node with identifiers replaced
    """
    # Base case: replace this identifier if it's in the map
    if isinstance(node, Identifier) and node.ptr in substitution_map:
        return substitution_map[node.ptr]
    
    # Handle each node type explicitly to ensure correct field updates
    if isinstance(node, SetComprehension):
        new_domain = replace_identifiers(node.domain, substitution_map)
        new_guard = node.guard
        if isinstance(new_guard, SetGuard) and new_guard.set_expr is not None:
            new_guard = replace(new_guard, set_expr=replace_identifiers(new_guard.set_expr, substitution_map))
        return replace(node, domain=new_domain, guard=new_guard)
    
    if isinstance(node, SetGuard) and node.set_expr is not None:
        new_set_expr = replace_identifiers(node.set_expr, substitution_map)
        return replace(node, set_expr=new_set_expr)
    
    if isinstance(node, LinearScale):
        new_scaled_set = replace_identifiers(node.scaled_set, substitution_map)
        return replace(node, scaled_set=new_scaled_set)
    
    if isinstance(node, Shift):
        new_shifted_set = replace_identifiers(node.shifted_set, substitution_map)
        return replace(node, shifted_set=new_shifted_set)
    
    if isinstance(node, UnionSet):
        new_parts = tuple(replace_identifiers(part, substitution_map) for part in node.parts)
        return replace(node, parts=new_parts)
    
    if isinstance(node, IntersectionSet):
        new_parts = tuple(replace_identifiers(part, substitution_map) for part in node.parts)
        return replace(node, parts=new_parts)
    
    if isinstance(node, ComplementSet):
        new_complemented_set = replace_identifiers(node.complemented_set, substitution_map)
        return replace(node, complemented_set=new_complemented_set)
    
    if isinstance(node, FiniteSet):
        new_members = frozenset(replace_identifiers(member, substitution_map) for member in node.members)
        return replace(node, members=new_members)
    
    if isinstance(node, VectorSpace):
        new_basis = tuple(replace_identifiers(basis_vec, substitution_map) for basis_vec in node.basis)
        return replace(node, basis=new_basis)
    
    if isinstance(node, UnionSpace):
        new_parts = tuple(replace_identifiers(part, substitution_map) for part in node.parts)
        return replace(node, parts=new_parts)
    
    # For any other node types, try the generic approach
    try:
        children = get_raw_children(node)
        new_children = [replace_identifiers(child, substitution_map) for child in children]
        return reconstruct_node(node, new_children)
    except Exception as e:
        logger.debug("Failed to replace in %s: %s", type(node).__name__, e)
        return node


# -----------------------------------------------------------------------------
# LinearTransform pushdown
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearTransform(SymbolicSet):
    """
    Represents a linear transformation applied to a set.
    
    Transforms elements by: x' = scale * x + shift (component-wise)
    Used to push scaling and shifting operations down through the IR tree
    for optimization.
    """
    arguments: Tuple[Argument, ...]  # Variables being transformed
    scales: Tuple[int, ...]  # Multiplicative factors for each dimension
    shifts: Tuple[int, ...]  # Additive offsets for each dimension
    child: IRNode  # The set being transformed

    def __post_init__(self):
        """Validate that scales and shifts match the number of arguments."""
        if len(self.scales) != len(self.arguments) or len(self.shifts) != len(self.arguments):
            raise ValueError("Scale/shift lengths must match argument count")

    @classmethod
    def identity(cls, args: Tuple[Argument, ...], child: IRNode) -> LinearTransform:
        """
        Create an identity transform (scale=1, shift=0 for all dimensions).
        
        Args:
            args: Arguments to transform
            child: Child node
            
        Returns:
            Identity LinearTransform
        """
        n = len(args)
        return cls(args, (1,) * n, (0,) * n, child)

    def with_args_from(self, args: Tuple[Argument, ...], child: IRNode) -> LinearTransform:
        """
        Create a new LinearTransform with different arguments and child.
        
        Preserves the scale and shift values from this transform but
        applies them to different arguments and child node.
        
        Args:
            args: New arguments
            child: New child node
            
        Returns:
            New LinearTransform with same scales/shifts
        """
        return LinearTransform(args, self.scales, self.shifts, child)

    def apply_scale(self, factor: Tuple[int, ...]) -> LinearTransform:
        """
        Compose with additional scaling.
        
        When we have scale(s) followed by scale(f), the result is scale(s*f).
        Also scales the shifts: shift(sh) followed by scale(f) becomes shift(sh*f).
        
        Args:
            factor: Additional scale factors to apply
            
        Returns:
            New LinearTransform with composed scaling
        """
        new_scales = tuple(s * f for s, f in zip(self.scales, factor))
        new_shifts = tuple(sh * f for sh, f in zip(self.shifts, factor))
        return LinearTransform(self.arguments, new_scales, new_shifts, self.child)

    def apply_shift(self, delta: Tuple[int, ...]) -> LinearTransform:
        """
        Compose with additional shifting.
        
        When we have shift(sh) followed by shift(d), the result is shift(sh+d).
        
        Args:
            delta: Additional shift offsets to apply
            
        Returns:
            New LinearTransform with composed shifting
        """
        new_shifts = tuple(sh + d for sh, d in zip(self.shifts, delta))
        return LinearTransform(self.arguments, self.scales, new_shifts, self.child)


@dataclass(frozen=True)
class Annotated(IRNode):
    """
    Wrapper for IR nodes with additional metadata annotations.
    
    Allows attaching arbitrary metadata to nodes without modifying their type.
    Uses attribute delegation to make the wrapper transparent for most operations.
    """
    base: IRNode  # The wrapped node
    meta: dict[str, Any]  # Metadata dictionary
    
    def __getattribute__(self, name):
        """Delegate attribute access to the base node transparently."""
        # Always return our own base and meta
        if name in ('base', 'meta'):
            return object.__getattribute__(self, name)
        
        # Make isinstance checks work correctly
        if name == '__class__':
            return object.__getattribute__(self, 'base').__class__
            
        # Special handling for dataclass operations
        if name in ('__dataclass_fields__', '__dataclass_params__'):
            return getattr(object.__getattribute__(self, 'base'), name)
        
        # Delegate everything else to the base node
        try:
            base = object.__getattribute__(self, 'base')
            return getattr(base, name)
        except AttributeError:
            return object.__getattribute__(self, name)

    def __repr__(self) -> str:
        """String representation showing base and metadata."""
        if not self.meta:
            return f"@{self.base!r}"
        return f"@{self.base!r} {self.meta!r}"


def is_annotated(node: IRNode) -> bool:
    """
    Check if a node is an Annotated wrapper.
    
    Args:
        node: Node to check
        
    Returns:
        True if node is Annotated
    """
    return object.__getattribute__(node, '__class__') is Annotated


def get_annotations(node: IRNode) -> dict[str, Any]:
    """
    Extract metadata from an Annotated node.
    
    Args:
        node: Node to get annotations from
        
    Returns:
        Metadata dictionary, or empty dict if not annotated
    """
    if is_annotated(node):
        return object.__getattribute__(node, 'meta')
    return {}


def collect_existing_scales(node: IRNode) -> List[Tuple[int, ...]]:
    """
    Collect existing modulo guard scales from node annotations.
    
    Args:
        node: Node to extract scales from
        
    Returns:
        List of scale tuples from mod_guard annotation
    """
    annotations = get_annotations(node)
    return annotations.get("mod_guard", [])


def push_linear_transform(ltf: LinearTransform) -> IRNode:
    """
    Push linear transforms down through the IR tree.
    
    Optimization that moves scale and shift operations as close to the
    leaves as possible, potentially absorbing them into other operations
    or distributing them across set operations.
    
    Args:
        ltf: LinearTransform to push down
        
    Returns:
        Transformed IR node
        
    Raises:
        NotImplementedError: If pushdown not implemented for a node type
    """
    child = ltf.child
    
    # Base cases: cannot push further into these node types
    if isinstance(child, (Identifier, VectorSpace, FiniteSet, SimpleGuard, ProductDomain)):
        return child if isinstance(child, ProductDomain) else ltf
    
    # Absorption: combine consecutive linear operations
    if isinstance(child, LinearScale):
        # scale(s2) after scale(s1) becomes scale(s1*s2)
        new_ltf = ltf.apply_scale(child.factor.comps)
        # Preserve existing scale annotations
        existing_scales = collect_existing_scales(child)
        all_scales = existing_scales + [child.factor.comps]
        return push_linear_transform(
            replace(new_ltf, child=
                    Annotated(child.scaled_set, {"mod_guard": (all_scales, ltf.shifts)})))
    
    if isinstance(child, Shift):
        # shift(s2) after shift(s1) becomes shift(s1+s2)
        new_ltf = ltf.apply_shift(child.shift.comps)
        return push_linear_transform(replace(new_ltf, child=child.shifted_set))
    
    # Distribution: push transform into each part of set operations
    if isinstance(child, (UnionSet, UnionSpace, IntersectionSet)):
        parts = child.parts
        # Apply transform to each part independently
        transformed = [push_linear_transform(LinearTransform(ltf.arguments, ltf.scales, ltf.shifts, p)) for p in parts]
        
        if isinstance(child, UnionSet):
            return UnionSet(parts=tuple(transformed))  # type: ignore
        if isinstance(child, UnionSpace):
            return UnionSpace(parts=tuple(transformed))  # type: ignore
        return IntersectionSet(parts=tuple(transformed))  # type: ignore 

    if isinstance(child, ComplementSet):
        # Push transform into complemented set
        return ComplementSet(
            complemented_set=push_linear_transform(ltf.with_args_from(ltf.arguments, child.complemented_set))  # type: ignore
        )
    
    if isinstance(child, SetComprehension):
        # Push transform into both domain and guard
        return SetComprehension(
            child.arguments,
            push_linear_transform(ltf.with_args_from(ltf.arguments, child.domain)),  # type: ignore
            push_linear_transform(ltf.with_args_from(ltf.arguments, child.guard))  # type: ignore
        )
    
    if isinstance(child, SetGuard) and child.set_expr is not None:
        # Push transform into the guard's set expression
        return SetGuard(
            arguments=child.arguments,
            set_expr=push_linear_transform(ltf.with_args_from(ltf.arguments, child.set_expr))  # type: ignore
        )
    
    if isinstance(child, Vector):
        # Apply transform directly to vector components
        return Vector(tuple([ltf.scales[i] * child.comps[i] + ltf.shifts[i] for i in range(child.dimension)]))
    
    raise NotImplementedError(f"LT pushdown not implemented for {type(child).__name__}")


# -----------------------------------------------------------------------------
# Top-level optimization
# -----------------------------------------------------------------------------

def optimize(ir: IRNode, scopes: ScopeHandler) -> IRNode:
    """
    Apply optimization passes to the IR.
    
    Current optimizations:
    1. Inline all identifier references
    2. Push linear transforms down through the tree
    
    Args:
        ir: Root IR node to optimize
        scopes: Scope handler for identifier resolution
        
    Returns:
        Optimized IR tree
    """
    logger.debug("Starting optimization")
    
    # Pass 1: Inline all identifiers
    ir = inline_all_identifiers(ir, scopes)
    logger.debug("Inlined all identifiers")
    
    # Pass 2: Push linear transforms down
    assert isinstance(ir, SetComprehension)
    ltf = LinearTransform.identity(ir.arguments, ir)
    ir = push_linear_transform(ltf)
    logger.debug("Applied linear transform pushdown")
    
    logger.debug("Optimization complete")
    return ir
