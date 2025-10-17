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

# FIXME: The annotation of a node is lost if the node is eliminated from the tree
# instead push it on the child node.

# -----------------------------------------------------------------------------
# Core traversal utilities
# -----------------------------------------------------------------------------

def get_semantic_children(node: IRNode, scopes: ScopeHandler) -> Tuple[IRNode, ...]:
    if isinstance(node, Identifier):
        resolved = scopes.lookup_by_id(node.ptr)
        return (resolved,) if resolved else ()
    if isinstance(node, SetComprehension):
        children: List[IRNode] = [node.domain]
        if isinstance(node.guard, SetGuard) and node.guard.set_expr is not None: #type: ignore
            children.append(node.guard.set_expr) #type: ignore
        return tuple(children)
    return node.children


def get_raw_children(node: IRNode) -> List[IRNode]:
    """Get the actual children that should be processed for replacement."""
    if isinstance(node, SetComprehension):
        children = [node.domain]
        if isinstance(node.guard, SetGuard) and node.guard.set_expr is not None:
            children.append(node.guard.set_expr)
        return children #type: ignore
    return list(node.children)


def fold_ir(
    node: IRNode,
    scopes: ScopeHandler,
    fn: Callable[[IRNode, Tuple[Any, ...]], Any],
    memo: Optional[Dict[int, Any]] = None
) -> Any:
    memo = memo if memo is not None else {}
    key = id(node)
    if key in memo:
        return memo[key]
    child_vals = tuple(fold_ir(ch, scopes, fn, memo) for ch in get_semantic_children(node, scopes))
    result = fn(node, child_vals)
    memo[key] = result
    return result


def map_ir(
    node: IRNode,
    scopes: ScopeHandler,
    fn: Callable[[IRNode, List[IRNode]], IRNode],
    memo: Optional[Dict[int, IRNode]] = None
) -> IRNode:
    memo = memo if memo is not None else {}
    key = id(node)
    if key in memo:
        return memo[key]
    kids = get_semantic_children(node, scopes)
    new_kids = [map_ir(ch, scopes, fn, memo) for ch in kids]
    result = fn(node, new_kids)
    memo[key] = result
    return result

# -----------------------------------------------------------------------------
# Reconstruction logic
# -----------------------------------------------------------------------------

def reconstruct_node(node: IRNode, new_children: List[IRNode]) -> IRNode:
    if not new_children:
        return node
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
    """Build dependency graph with definitions and dependencies."""
    definitions = {}
    graph = defaultdict(set)
    visited = set()
    
    def collect_definitions(node: IRNode):
        if id(node) in visited:
            return
        visited.add(id(node))
        
        if isinstance(node, Identifier):
            ptr = node.ptr
            if ptr not in definitions:
                def_node = scopes.lookup_by_id(ptr)
                if def_node:
                    definitions[ptr] = def_node
                    collect_definitions(def_node)
        else:
            for child in node.children:
                collect_definitions(child)
    
    collect_definitions(root)
    
    for ptr, def_node in definitions.items():
        dependencies = collect_identifier_pointers(def_node)
        graph[ptr] = {dep for dep in dependencies if dep in definitions}
    
    return definitions, graph


def collect_identifier_pointers(node: IRNode) -> Set[int]:
    """Collect all identifier pointers in a node tree using raw traversal."""
    pointers = set()
    
    if isinstance(node, Identifier):
        pointers.add(node.ptr)
    
    for child in node.children:
        pointers.update(collect_identifier_pointers(child))
    
    return pointers


def topological_sort(graph: Dict[int, Set[int]]) -> List[int]:
    """Return topologically sorted list of identifiers."""
    in_degree = {node: 0 for node in graph}
    for node, dependencies in graph.items():
        for dep in dependencies:
            if dep in in_degree:
                in_degree[node] += 1
    
    queue = deque([node for node, degree in in_degree.items() if degree == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for other_node, dependencies in graph.items():
            if node in dependencies:
                in_degree[other_node] -= 1
                if in_degree[other_node] == 0:
                    queue.append(other_node)
    
    if len(result) != len(graph):
        logger.debug("Circular dependencies detected in identifier graph")
    
    return result


def inline_all_identifiers(root: IRNode, scopes: ScopeHandler) -> IRNode:
    """
    Two-pass inlining:
    1. Build complete substitution map in dependency order
    2. Apply all substitutions to root
    """
    definitions, dependency_graph = build_dependency_graph(root, scopes)
    
    if not definitions:
        logger.debug("No definitions found for inlining")
        return root
    
    logger.debug("Found %d definitions to inline", len(definitions))
    for ptr, def_node in definitions.items():
        logger.debug("Definition %d: %s", ptr, type(def_node).__name__)
    
    sorted_identifiers = topological_sort(dependency_graph)
    logger.debug("Topological order: %s", sorted_identifiers)
    
    substitution_map: Dict[int, IRNode] = {}
    
    for identifier_ptr in sorted_identifiers:
        if identifier_ptr in definitions:
            original_definition = definitions[identifier_ptr]
            inlined_definition = replace_identifiers(original_definition, substitution_map)
            substitution_map[identifier_ptr] = inlined_definition
            logger.debug("Inlined %d -> %s", identifier_ptr, type(inlined_definition).__name__)
    
    logger.debug("Built substitution map with %d entries", len(substitution_map))
    
    result = replace_identifiers(root, substitution_map)
    
    remaining = collect_identifier_pointers(result)
    if remaining:
        logger.debug("Unresolved identifiers after inlining: %s", remaining)
    
    return result


def replace_identifiers(node: IRNode, substitution_map: Dict[int, IRNode]) -> IRNode:
    if isinstance(node, Identifier) and node.ptr in substitution_map:
        return substitution_map[node.ptr]
    
    # Handle all node types explicitly
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
    arguments: Tuple[Argument, ...]
    scales: Tuple[int, ...]
    shifts: Tuple[int, ...]
    child: IRNode

    def __post_init__(self):
        if len(self.scales) != len(self.arguments) or len(self.shifts) != len(self.arguments):
            raise ValueError("Scale/shift lengths must match argument count")

    @classmethod
    def identity(cls, args: Tuple[Argument, ...], child: IRNode) -> LinearTransform:
        n = len(args)
        return cls(args, (1,) * n, (0,) * n, child)

    def with_args_from(self, args: Tuple[Argument, ...], child: IRNode) -> LinearTransform:
        """
        Create a new LinearTransform that preserves the scales/shifts
        from `base` but replaces the arguments and child.
        """
        return LinearTransform(args, self.scales, self.shifts, child)

    def apply_scale(self, factor: Tuple[int, ...]) -> LinearTransform:
        new_scales = tuple(s * f for s, f in zip(self.scales, factor))
        new_shifts = tuple(sh * f for sh, f in zip(self.shifts, factor))
        ltf = LinearTransform(self.arguments, new_scales, new_shifts, self.child)
        return ltf

    def apply_shift(self, delta: Tuple[int, ...]) -> LinearTransform:
        new_shifts = tuple(sh + d for sh, d in zip(self.shifts, delta))
        return LinearTransform(self.arguments, self.scales, new_shifts, self.child)


@dataclass(frozen=True)
class Annotated(IRNode):
    base: IRNode
    meta: dict[str, Any]
    
    def __getattribute__(self, name):
        if name in ('base', 'meta'):
            return object.__getattribute__(self, name)
        
        if name == '__class__':
            return object.__getattribute__(self, 'base').__class__
            
        # Special handling for dataclass replace operations
        if name in ('__dataclass_fields__', '__dataclass_params__'):
            return getattr(object.__getattribute__(self, 'base'), name)
        
        try:
            base = object.__getattribute__(self, 'base')
            return getattr(base, name)
        except AttributeError:
            return object.__getattribute__(self, name)

    def __repr__(self) -> str:
        if not self.meta:
            return f"@{self.base!r}"
        return f"@{self.base!r} {self.meta!r}"

def is_annotated(node: IRNode) -> bool:
    return object.__getattribute__(node, '__class__') is Annotated

def get_annotations(node: IRNode) -> dict[str, Any]:
    if is_annotated(node):
        return object.__getattribute__(node, 'meta')
    return {}

def collect_existing_scales(node: IRNode) -> List[Tuple[int, ...]]:
    """Collect existing mod_guard scales from annotations."""
    annotations = get_annotations(node)
    return annotations.get("mod_guard", [])

def push_linear_transform(ltf: LinearTransform) -> IRNode:
    child = ltf.child
    # Base cases
    if isinstance(child, (Identifier, VectorSpace, FiniteSet, SimpleGuard, ProductDomain)):
        return child if isinstance(child, ProductDomain) else ltf
    # Absorption
    if isinstance(child, LinearScale):
        new_ltf = ltf.apply_scale(child.factor.comps)
        existing_scales = collect_existing_scales(child)
        all_scales = existing_scales + [child.factor.comps]
        print(new_ltf.shifts)
        return push_linear_transform(
            replace(new_ltf, child=
                    Annotated(child.scaled_set, {"mod_guard": (all_scales, ltf.shifts)})))
    if isinstance(child, Shift):
        new_ltf = ltf.apply_shift(child.shift.comps)
        return push_linear_transform(replace(new_ltf, child=child.shifted_set))
    # Distribute into set operators
    if isinstance(child, UnionSet) or isinstance(child, UnionSpace) or isinstance(child, IntersectionSet):
        parts = child.parts
        transformed = [push_linear_transform(LinearTransform(ltf.arguments, ltf.scales, ltf.shifts, p)) for p in parts]
        if isinstance(child, UnionSet):
            return UnionSet(parts=tuple(transformed)) #type: ignore
        if isinstance(child, UnionSpace):
            return UnionSpace(parts=tuple(transformed)) #type: ignore
        return IntersectionSet(parts=tuple(transformed)) #type: ignore 

    if isinstance(child, ComplementSet):
        return ComplementSet(
            complemented_set=push_linear_transform(ltf.with_args_from(ltf.arguments, child.complemented_set)) #type: ignore
        )
    if isinstance(child, SetComprehension):
        s= SetComprehension(
            child.arguments,
            push_linear_transform(ltf.with_args_from(ltf.arguments, child.domain)), #type: ignore
            push_linear_transform(ltf.with_args_from(ltf.arguments, child.guard)) #type: ignore
        )
        return s
    if isinstance(child, SetGuard) and child.set_expr is not None:
        return SetGuard(
            arguments=child.arguments,
            set_expr=push_linear_transform(ltf.with_args_from(ltf.arguments, child.set_expr)) #type: ignore
        )
    if isinstance(child, Vector):
        return Vector(tuple([ltf.scales[i] * child.comps[i] + ltf.shifts[i] for i in range(child.dimension)]))
    raise NotImplementedError(f"LT pushdown not implemented for {type(child).__name__}")

# -----------------------------------------------------------------------------
# Top-level optimization
# -----------------------------------------------------------------------------
def optimize(ir: IRNode, scopes: ScopeHandler) -> IRNode:
    """Optimized version using two-pass inlining."""
    logger.debug("Starting optimization")
    ir = inline_all_identifiers(ir, scopes)
    logger.debug("Inlined all identifiers")
    assert isinstance(ir, SetComprehension)
    ltf = LinearTransform.identity(ir.arguments, ir)
    ir = push_linear_transform(ltf)
    logger.debug("Applied linear transform pushdown")
    logger.debug("Optimization complete")
    return ir
