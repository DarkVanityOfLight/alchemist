from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple
from expressions import (
    Argument, IRNode, Identifier, ProductDomain, Scalar, UnionSpace, Vector, VectorSpace, FiniteSet, UnionSet, IntersectionSet,
    DifferenceSet, ComplementSet, LinearScale, Shift, SetComprehension, SymbolicSet
)
from collections import Counter
from dataclasses import dataclass, replace

from guards import Guard, SMTGuard, SetGuard, SimpleGuard
if TYPE_CHECKING:
    from scope_handler import ScopeHandler

_inlined = 0
_semantic_children_cache = {}

def semantic_children(node: IRNode, scopes: ScopeHandler) -> Tuple[IRNode, ...]:
    """Get the semantic children of a node, resolving identifiers through scopes."""
    # Use node id and a simple cache key
    cache_key = None
    if isinstance(node, Identifier):
        cache_key = (id(node), type(node).__name__, node.ptr)
    else:
        cache_key = (id(node), type(node).__name__)
    
    if cache_key in _semantic_children_cache:
        cached_result = _semantic_children_cache[cache_key]
        # For Identifier nodes, we still need to do the lookup since scope might change
        if not isinstance(node, Identifier):
            return cached_result
    
    if isinstance(node, Identifier):
        sbtree = scopes.lookup_by_id(node.ptr)
        result = (sbtree,) if sbtree else ()
    elif isinstance(node, SetComprehension):
        # Include domain and set expression from SetGuard if present
        children = [node.domain]
        if isinstance(node.guard, SetGuard) and node.guard.set_expr is not None:
            children.append(node.guard.set_expr)
        result = tuple(children)
    else:
        result = node.children
    
    # Cache non-Identifier results
    if not isinstance(node, Identifier):
        _semantic_children_cache[cache_key] = result
    
    return result


def fold_ir(node: IRNode,
            scopes: ScopeHandler,
            fn: Callable[[IRNode, Tuple[Any, ...]], Any],
            memo: Optional[Dict[int, Any]] = None
           ) -> Any:
    if memo is None:
        memo = {}
    key = id(node)
    if key in memo:
        return memo[key]

    kids = semantic_children(node, scopes)
    child_vals = tuple(fold_ir(ch, scopes, fn, memo) for ch in kids)
    result = fn(node, child_vals)
    memo[key] = result
    return result


def map_ir(node: IRNode,
           scopes: ScopeHandler,
           fn: Callable[[IRNode, List[IRNode]], IRNode],
           memo: Optional[Dict[int, IRNode]] = None
          ) -> IRNode:
    if memo is None:
        memo = {}
    key = id(node)
    if key in memo:
        return memo[key]

    kids: Tuple[IRNode, ...] = semantic_children(node, scopes)
    new_kids: List[IRNode] = [map_ir(ch, scopes, fn, memo) for ch in kids]

    # Apply the transformation function to get the result
    result = fn(node, new_kids)
    memo[key] = result
    
    # Return the transformed result (don't short-circuit based on identity)
    return result

def map_ir_pruned(
    node: IRNode,
    scopes: ScopeHandler,
    fn: Callable[[IRNode, List[IRNode]], IRNode],
    once_used_ids: Set[int],
    memo: Optional[Dict[int, IRNode]] = None
) -> IRNode:
    if memo is None:
        memo = {}
    key = id(node)
    if key in memo:
        return memo[key]

    # PRUNE: if this is an Identifier we won't inline, just return it
    if isinstance(node, Identifier) and node.ptr not in once_used_ids:
        memo[key] = node
        return node

    # Otherwise, behave just like your usual map_ir
    kids = semantic_children(node, scopes)
    new_kids = [map_ir_pruned(ch, scopes, fn, once_used_ids, memo) for ch in kids]
    result = fn(node, new_kids)
    memo[key] = result
    return result

def count_identifiers(node: IRNode, child_counts: Tuple[Counter, ...]) -> Counter:
    """
    Count identifier usage in the IR tree by ID.
    
    Args:
        node: Current IRNode
        child_counts: Counters from child nodes
    
    Returns:
        Counter with identifier ID usage counts
    """
    # merge all child counters
    total = Counter()
    for cnt in child_counts:
        total.update(cnt)
    
    # if this node is an Identifier, increment its ID
    if isinstance(node, Identifier):
        total[node.ptr] += 1
    
    return total

_NODE_RECONSTRUCTORS = {
    VectorSpace: lambda node, children: replace(node, basis=tuple(children)),
    FiniteSet: lambda node, children: replace(node, members=frozenset(children)),
    UnionSet: lambda node, children: replace(node, parts=tuple(children)),
    IntersectionSet: lambda node, children: replace(node, parts=tuple(children)),
    DifferenceSet: lambda node, children: replace(node, minuend=children[0], subtrahend=children[1]),
    ComplementSet: lambda node, children: replace(node, complemented_set=children[0]),
    LinearScale: lambda node, children: replace(node, scaled_set=children[0]),
    Shift: lambda node, children: replace(node, shifted_set=children[0]),
}

def inline_mapper(node: IRNode, new_children: List[IRNode], 
                  once_used_ids: set, scopes: ScopeHandler) -> IRNode:
    # Fast path: if no children changed and not an identifier to inline
    if (not new_children or tuple(new_children) == node.children) and \
       not (isinstance(node, Identifier) and node.ptr in once_used_ids):
        return node
    
    # Inline identifiers used exactly once
    if isinstance(node, Identifier) and node.ptr in once_used_ids:
        global _inlined
        _inlined += 1
        definition = scopes.lookup_by_id(node.ptr)
        if definition:
            def recursive_mapper(inner_node: IRNode, inner_children: List[IRNode]) -> IRNode:
                return inline_mapper(inner_node, inner_children, once_used_ids, scopes)
            return map_ir(definition, scopes, recursive_mapper)
        return node
    
    # Handle reconstruction for SetComprehension with updated domain AND guard
    if isinstance(node, SetComprehension):
        new_domain = new_children[0]
        new_guard = node.guard
        
        # Update guard expression if it exists
        if isinstance(new_guard, SetGuard) and node.guard.set_expr is not None: # type: ignore
            if len(new_children) > 1:
                new_guard = replace(new_guard, set_expr=new_children[1])
            else:
                # Maintain existing expression if no new child
                new_guard = replace(new_guard, set_expr=node.guard.set_expr) #type: ignore
                
        return replace(node, domain=new_domain, guard=new_guard)
    
    # Handle reconstruction for SetGuard nodes directly
    if isinstance(node, SetGuard) and node.set_expr is not None:
        if new_children:
            return replace(node, set_expr=new_children[0])
        return node

    # Use pre-compiled reconstructors for faster dispatch
    node_type = type(node)
    if node_type in _NODE_RECONSTRUCTORS:
        return _NODE_RECONSTRUCTORS[node_type](node, new_children)
    
    return node

@dataclass(frozen=True)
class LinearTransform(Guard, SymbolicSet):
    arguments: Tuple[Argument, ...]
    scales: Tuple[int, ...]
    shifts: Tuple[int, ...]
    child: IRNode

    def __post_init__(self):
        n_args = len(self.arguments)
        if len(self.scales) != n_args:
            raise ValueError(f"scales length {len(self.scales)} does not match arguments length {n_args}")
        if len(self.shifts) != n_args:
            raise ValueError(f"shifts length {len(self.shifts)} does not match arguments length {n_args}")

    @classmethod
    def identity(cls, arguments: Tuple[Argument, ...], child: IRNode) -> LinearTransform:
        n = len(arguments)
        return cls(
            arguments=arguments,
            scales=tuple(1 for _ in range(n)),
            shifts=tuple(0 for _ in range(n)),
            child=child
        )

    def apply_scale(self, scale_vec: Tuple[int, ...]) -> LinearTransform:
        if len(scale_vec) != len(self.scales):
            raise ValueError("scale vector must have same length as scales")
        new_scales = tuple(s * sc for s, sc in zip(self.scales, scale_vec))
        new_shifts = tuple(sh * sc for sh, sc in zip(self.shifts, scale_vec))
        return LinearTransform(self.arguments, new_scales, new_shifts, self.child)

    def apply_shift(self, shift_vec: Tuple[int, ...]) -> LinearTransform:
        if len(shift_vec) != len(self.shifts):
            raise ValueError("shift vector must have same length as shifts")
        new_shifts = tuple(sh + delta for sh, delta in zip(self.shifts, shift_vec))
        return LinearTransform(self.arguments, self.scales, new_shifts, self.child)

    @property
    def children(self) -> Tuple[IRNode, ...]:
        return (self.child,)


def push_linear_transform(ltf: LinearTransform) -> IRNode:
    child = ltf.child

    match child:
        # Base cases
        case Identifier():
            return ltf
        case VectorSpace():
            return ltf
        case UnionSpace():
            return ltf
        case FiniteSet():
            return ltf
        case SimpleGuard():
            return ltf

        # Absorption
        case LinearScale(factor=f, scaled_set=grand_child):
            return replace(ltf.apply_scale(f.comps), child=grand_child)
        case Shift(shift=s, shifted_set=grand_child):
            return replace(ltf.apply_shift(s.comps), child=grand_child)

        # Ignores LTF
        case ProductDomain() as p:
            return p

        # Distribute over set operators
        case UnionSet(parts=p):
            transformed_parts = [push_linear_transform(replace(ltf, child=part)) for part in p]
            return UnionSet(parts=tuple(transformed_parts)) #type: ignore
        case IntersectionSet(parts=p):
            transformed_parts = [push_linear_transform(replace(ltf, child=part)) for part in p]
            return IntersectionSet(parts=tuple(transformed_parts)) #type: ignore
        case DifferenceSet(minuend=m, subtrahend=s):
            m_trans = push_linear_transform(replace(ltf, child=m))
            s_trans = push_linear_transform(replace(ltf, child=s))
            return DifferenceSet(minuend=m_trans, subtrahend=s_trans) #type: ignore
        case ComplementSet(complemented_set=c):
            c_trans = push_linear_transform(replace(ltf, child=c))
            return ComplementSet(complemented_set=c_trans) #type: ignore

        case SetComprehension(arguments=args, domain=d, guard=g):
            d_trans = push_linear_transform(replace(ltf, child=d))
            g_trans = push_linear_transform(replace(ltf, child=g))
            return SetComprehension(args, d_trans, g_trans) #type: ignore
        case SetGuard(arguments=args, set_expr=n):
            n_trans = push_linear_transform(replace(ltf, child=n))
            return SetGuard(arguments=args, set_expr=n_trans)  #type: ignore

        # Unhandled/Shouldn't hit
        case Vector():
            raise ValueError("Cannot LTF a scalar")  # FIXME: You probably can
        case Scalar():
            raise ValueError("Cannot LTF a scalar")
        case SMTGuard():
            raise ValueError("Cannot LTF a SMTGuard")

    raise NotImplementedError(f"Did not implement LTF pushdown for node: {child}")


def preprocess_multiple_used_definition(
    node: IRNode,
    once_used_ids: Set[int],
    scopes: ScopeHandler,
) -> IRNode:

    def mapper(node: IRNode, new_children: List[IRNode]) -> IRNode:
        result = inline_mapper(node, new_children, once_used_ids, scopes)
        return result

    inlined_ir = map_ir(node, scopes, mapper)

    return inlined_ir


def optimize(ir: IRNode, scopes: ScopeHandler) -> IRNode:
    total_nodes = fold_ir(
        ir,
        scopes,
        lambda node, child_counts: 1 + sum(child_counts)
    )
    print(total_nodes)

    usages = fold_ir(ir, scopes, count_identifiers)
    once_used_ids = {id for id, count in usages.items() if count == 1 and scopes.lookup_by_id(id) is not None}

    print(f"Inlining {len(once_used_ids)} definitions")
    if not once_used_ids:
        return ir

    to_preprocess = {
            def_id: tree
            for def_id, tree in scopes.parsed_ids.items()
            if def_id not in once_used_ids
        }
    new_id_mapping: Dict[int, IRNode] = {
        def_id: preprocess_multiple_used_definition(tree, once_used_ids, scopes)
        for def_id, tree in to_preprocess.items()
    }

    # ——— patch lookup_by_id ———
    orig_lookup = scopes.lookup_by_id
    def lookup_by_id_patched(id: int) -> Optional[IRNode]:
        if id in new_id_mapping:
            return new_id_mapping[id]
        return orig_lookup(id)
    scopes.lookup_by_id = lookup_by_id_patched

    inlined_ir = map_ir_pruned(
        ir, scopes,
        lambda node, kids: inline_mapper(node, kids, once_used_ids, scopes),
        once_used_ids
    )

    print("Inline complete")
    print(inlined_ir)

    print("Pushing down LTF")
    assert isinstance(inlined_ir, SetComprehension)
    ltf = LinearTransform.identity(inlined_ir.arguments, inlined_ir)
    ltf_ir = push_linear_transform(ltf)


    # print(ltf_ir)
    return ltf_ir
