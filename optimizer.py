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

def semantic_children(node: IRNode, scopes: ScopeHandler) -> Tuple[IRNode, ...]:
    """Get the semantic children of a node, resolving identifiers through scopes."""
    if isinstance(node, Identifier):
        sbtree = scopes.lookup_by_id(node.ptr)
        return (sbtree,) if sbtree else ()
    elif isinstance(node, SetComprehension):
        # Always include domain, conditionally include guard's set_expr
        children = [node.domain]
        if isinstance(node.guard, SetGuard) and node.guard.set_expr is not None:
            children.append(node.guard.set_expr)
        return tuple(children)
    else:
        return node.children


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
    result = fn(node, new_kids)
    memo[key] = result
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

    # PRUNE: if this is an Identifier we won't inline, process it normally
    # but don't recurse into its definition
    if isinstance(node, Identifier) and node.ptr not in once_used_ids:
        memo[key] = node
        return node

    kids = semantic_children(node, scopes)
    new_kids = [map_ir_pruned(ch, scopes, fn, once_used_ids, memo) for ch in kids]
    result = fn(node, new_kids)
    memo[key] = result
    return result

def count_identifiers(node: IRNode, child_counts: Tuple[Counter, ...]) -> Counter:
    """Count identifier usage in the IR tree by ID."""
    total = Counter()
    for cnt in child_counts:
        total.update(cnt)
    
    if isinstance(node, Identifier):
        total[node.ptr] += 1
    
    return total

def reconstruct_node(node: IRNode, new_children: List[IRNode]) -> IRNode:
    """Generic node reconstruction using dataclass replace."""
    if not new_children:
        return node
    
    # Handle special cases that don't follow standard children pattern
    if isinstance(node, SetComprehension):
        new_domain = new_children[0]
        new_guard = node.guard
        
        # Update guard expression if it exists and we have a second child
        if isinstance(new_guard, SetGuard) and node.guard.set_expr is not None: #type: ignore
            if len(new_children) > 1:
                new_guard = replace(new_guard, set_expr=new_children[1])
                
        return replace(node, domain=new_domain, guard=new_guard)
    
    elif isinstance(node, VectorSpace):
        return replace(node, basis=tuple(new_children))
    elif isinstance(node, FiniteSet):
        return replace(node, members=frozenset(new_children))
    elif isinstance(node, UnionSet):
        return replace(node, parts=tuple(new_children))
    elif isinstance(node, IntersectionSet):
        return replace(node, parts=tuple(new_children))
    elif isinstance(node, DifferenceSet):
        return replace(node, minuend=new_children[0], subtrahend=new_children[1])
    elif isinstance(node, ComplementSet):
        return replace(node, complemented_set=new_children[0])
    elif isinstance(node, LinearScale):
        return replace(node, scaled_set=new_children[0])
    elif isinstance(node, Shift):
        return replace(node, shifted_set=new_children[0])
    elif isinstance(node, SetGuard) and node.set_expr is not None:
        return replace(node, set_expr=new_children[0])
    
    # For other nodes, assume they follow the standard children pattern
    # This is a fallback - ideally all node types should be handled explicitly
    print(f"{node} was not reconstructed")
    return node

def inline_mapper(node: IRNode, new_children: List[IRNode], 
                  once_used_ids: set, scopes: ScopeHandler,
                  inline_memo: Optional[Dict[int, IRNode]] = None) -> IRNode:
    if inline_memo is None:
        inline_memo = {}
    
    # Inline identifiers used exactly once
    if isinstance(node, Identifier) and node.ptr in once_used_ids:
        if id(node) in inline_memo:
            return inline_memo[id(node)]
            
        global _inlined
        _inlined += 1
        definition = scopes.lookup_by_id(node.ptr)
        if definition:
            def recursive_mapper(inner_node: IRNode, inner_children: List[IRNode]) -> IRNode:
                return inline_mapper(inner_node, inner_children, once_used_ids, scopes, inline_memo)
            
            result = map_ir(definition, scopes, recursive_mapper)
            inline_memo[id(node)] = result
            return result
        return node
    
    # Check if reconstruction is needed
    current_children = semantic_children(node, scopes)
    if new_children and len(new_children) == len(current_children):
        # Compare actual children, not just length
        if all(new_child is old_child for new_child, old_child in zip(new_children, current_children)):
            return node
    
    return reconstruct_node(node, new_children)

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

    # Base cases - cannot push further
    if isinstance(child, (Identifier, VectorSpace, FiniteSet, SimpleGuard)):
        return ltf
    
    # Special case: ProductDomain ignores LTF
    if isinstance(child, ProductDomain):
        return child
    
    # Absorption cases
    if isinstance(child, LinearScale):
        new_ltf = ltf.apply_scale(child.factor.comps)
        return push_linear_transform(LinearTransform(new_ltf.arguments, new_ltf.scales, new_ltf.shifts, child.scaled_set))
    
    if isinstance(child, Shift):
        new_ltf = ltf.apply_shift(child.shift.comps)
        return push_linear_transform(LinearTransform(new_ltf.arguments, new_ltf.scales, new_ltf.shifts, child.shifted_set))
    
    # Distribute over set operators
    if isinstance(child, UnionSet):
        transformed_parts = [push_linear_transform(LinearTransform(ltf.arguments, ltf.scales, ltf.shifts, part)) 
                           for part in child.parts]
        return UnionSet(parts=tuple(transformed_parts)) #type: ignore
    
    if isinstance(child, UnionSpace):
        # Push transform into each part of the union space
        transformed_parts = [push_linear_transform(LinearTransform(ltf.arguments, ltf.scales, ltf.shifts, part)) 
                           for part in child.parts]
        return UnionSpace(parts=tuple(transformed_parts))#type: ignore
    
    if isinstance(child, IntersectionSet):
        transformed_parts = [push_linear_transform(LinearTransform(ltf.arguments, ltf.scales, ltf.shifts, part)) 
                           for part in child.parts]
        return IntersectionSet(parts=tuple(transformed_parts))#type: ignore
    
    if isinstance(child, DifferenceSet):
        m_trans = push_linear_transform(LinearTransform(ltf.arguments, ltf.scales, ltf.shifts, child.minuend))
        s_trans = push_linear_transform(LinearTransform(ltf.arguments, ltf.scales, ltf.shifts, child.subtrahend))
        return DifferenceSet(minuend=m_trans, subtrahend=s_trans)#type: ignore
    
    if isinstance(child, ComplementSet):
        c_trans = push_linear_transform(LinearTransform(ltf.arguments, ltf.scales, ltf.shifts, child.complemented_set))
        return ComplementSet(complemented_set=c_trans)#type: ignore
    
    if isinstance(child, SetComprehension):
        d_trans = push_linear_transform(LinearTransform(ltf.arguments, ltf.scales, ltf.shifts, child.domain))
        g_trans = push_linear_transform(LinearTransform(ltf.arguments, ltf.scales, ltf.shifts, child.guard))
        return SetComprehension(child.arguments, d_trans, g_trans)#type: ignore
    
    if isinstance(child, SetGuard):
        if child.set_expr is not None:
            n_trans = push_linear_transform(LinearTransform(ltf.arguments, ltf.scales, ltf.shifts, child.set_expr))
            return SetGuard(arguments=child.arguments, set_expr=n_trans)#type: ignore
        return child
    
    # Cases that should not occur or are errors
    if isinstance(child, (Vector, Scalar)):
        raise ValueError(f"Cannot apply linear transform to {type(child).__name__}")
    
    if isinstance(child, SMTGuard):
        raise ValueError("Cannot apply linear transform to SMTGuard")
    
    # Fallback for unhandled cases
    raise NotImplementedError(f"Linear transform pushdown not implemented for {type(child).__name__}")


def preprocess_multiple_used_definition(
    node: IRNode,
    once_used_ids: Set[int],
    scopes: ScopeHandler,
) -> IRNode:
    def mapper(node: IRNode, new_children: List[IRNode]) -> IRNode:
        return inline_mapper(node, new_children, once_used_ids, scopes)

    return map_ir(node, scopes, mapper)


def optimize(ir: IRNode, scopes: ScopeHandler) -> IRNode:
    total_nodes = fold_ir(
        ir,
        scopes,
        lambda node, child_counts: 1 + sum(child_counts)
    )
    print(f"Total nodes: {total_nodes}")

    usages = fold_ir(ir, scopes, count_identifiers)
    once_used_ids = {id for id, count in usages.items() if count == 1 and scopes.lookup_by_id(id) is not None}

    print(f"Inlining {len(once_used_ids)} definitions")
    if not once_used_ids:
        return ir

    # Preprocess definitions that are used multiple times
    to_preprocess = {
        def_id: tree
        for def_id, tree in scopes.parsed_ids.items()
        if def_id not in once_used_ids
    }
    
    new_id_mapping: Dict[int, IRNode] = {
        def_id: preprocess_multiple_used_definition(tree, once_used_ids, scopes)
        for def_id, tree in to_preprocess.items()
    }

    # Patch lookup_by_id to use preprocessed definitions
    orig_lookup = scopes.lookup_by_id
    def lookup_by_id_patched(id: int) -> Optional[IRNode]:
        if id in new_id_mapping:
            return new_id_mapping[id]
        return orig_lookup(id)
    scopes.lookup_by_id = lookup_by_id_patched

    # Perform inlining with pruning
    inlined_ir = map_ir_pruned(
        ir, scopes,
        lambda node, kids: inline_mapper(node, kids, once_used_ids, scopes),
        once_used_ids
    )

    print("Inline complete")
    print(inlined_ir)

    # Apply linear transform pushdown
    print("Pushing down linear transformations")
    if isinstance(inlined_ir, SetComprehension):
        ltf = LinearTransform.identity(inlined_ir.arguments, inlined_ir)
        ltf_ir = push_linear_transform(ltf)
        return ltf_ir
    
    return inlined_ir
