from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple
from expressions import (
    Argument, IRNode, Identifier, ProductDomain, VectorSpace, FiniteSet,
    UnionSet, IntersectionSet, ComplementSet,
    LinearScale, Shift, SetComprehension, SymbolicSet, Vector, Scalar, UnionSpace
)
from collections import Counter
from dataclasses import dataclass, replace
import logging

from guards import SetGuard, SimpleGuard

if TYPE_CHECKING:
    from scope_handler import ScopeHandler

logger = logging.getLogger(__name__)

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
    logger.debug("No reconstruction for %s", type(node).__name__)
    return node

# -----------------------------------------------------------------------------
# Identifier inlining utilities
# -----------------------------------------------------------------------------

def count_identifiers(node: IRNode, child_counts: Tuple[Counter, ...]) -> Counter:
    total = Counter()
    for c in child_counts:
        total.update(c)
    if isinstance(node, Identifier):
        total[node.ptr] += 1
    return total


def find_once_used_ids(root: IRNode, scopes: ScopeHandler) -> Set[int]:
    counts = fold_ir(root, scopes, count_identifiers)
    return {id_ for id_, cnt in counts.items() if cnt == 1 and scopes.lookup_by_id(id_) is not None}


def inline_identifiers(
    node: IRNode,
    scopes: ScopeHandler,
    once_used: Set[int],
    memo: Optional[Dict[int, IRNode]] = None
) -> IRNode:
    memo = memo if memo is not None else {}
    key = id(node)
    if key in memo:
        return memo[key]
    if isinstance(node, Identifier) and node.ptr in once_used:
        definition = scopes.lookup_by_id(node.ptr)
        if definition:
            result = inline_identifiers(definition, scopes, once_used, memo)
            memo[key] = result
            return result
        memo[key] = node
        return node
    if isinstance(node, Identifier):
        memo[key] = node
        return node
    kids = get_semantic_children(node, scopes)
    new_kids = [inline_identifiers(ch, scopes, once_used, memo) for ch in kids]
    result = reconstruct_node(node, new_kids)
    memo[key] = result
    return result

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

    def apply_scale(self, factor: Tuple[int, ...]) -> LinearTransform:
        new_scales = tuple(s * f for s, f in zip(self.scales, factor))
        new_shifts = tuple(sh * f for sh, f in zip(self.shifts, factor))
        return LinearTransform(self.arguments, new_scales, new_shifts, self.child)

    def apply_shift(self, delta: Tuple[int, ...]) -> LinearTransform:
        new_shifts = tuple(sh + d for sh, d in zip(self.shifts, delta))
        return LinearTransform(self.arguments, self.scales, new_shifts, self.child)


def push_linear_transform(ltf: LinearTransform) -> IRNode:
    child = ltf.child
    # Base cases
    if isinstance(child, (Identifier, VectorSpace, FiniteSet, SimpleGuard, ProductDomain)):
        return child if isinstance(child, ProductDomain) else ltf
    # Absorption
    if isinstance(child, LinearScale):
        new_ltf = ltf.apply_scale(child.factor.comps)
        return push_linear_transform(replace(new_ltf, child=child.scaled_set))
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
            complemented_set=push_linear_transform(LinearTransform.identity(ltf.arguments, child.complemented_set)) #type: ignore
        )
    if isinstance(child, SetComprehension):
        return SetComprehension(
            child.arguments,
            push_linear_transform(LinearTransform.identity(ltf.arguments, child.domain)), #type: ignore
            push_linear_transform(LinearTransform.identity(ltf.arguments, child.guard)) #type: ignore
        )
    if isinstance(child, SetGuard) and child.set_expr is not None:
        return SetGuard(
            arguments=child.arguments,
            set_expr=push_linear_transform(LinearTransform.identity(ltf.arguments, child.set_expr)) #type: ignore
        )
    raise NotImplementedError(f"LT pushdown not implemented for {type(child).__name__}")

# -----------------------------------------------------------------------------
# Top-level optimization
# -----------------------------------------------------------------------------
def optimize(ir: IRNode, scopes: ScopeHandler) -> IRNode:
    logger.info("Starting optimization")
    total = fold_ir(ir, scopes, lambda n, cs: 1 + sum(cs))
    logger.debug("Total IR nodes: %d", total)
    once = find_once_used_ids(ir, scopes)
    if once:
        logger.debug("Inlining %d once-used identifiers", len(once))
        ir = inline_identifiers(ir, scopes, once)
    else:
        logger.debug("No identifiers to inline")
    if isinstance(ir, SetComprehension):
        ltf = LinearTransform.identity(ir.arguments, ir)
        ir = push_linear_transform(ltf)
        logger.info("Applied linear transform pushdown")
    logger.info("Optimization complete")
    return ir
