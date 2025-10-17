from __future__ import annotations
from functools import reduce
import logging
from typing import Dict, List, Set, Tuple
from math import lcm, gcd

from arm_ast import ASTNode, NodeType
from expressions import (
    BaseDomain, ComplementSet, FiniteSet, IRNode, IntersectionSet, 
    ProductDomain, SetComprehension, UnionSet, UnionSpace, Vector, 
    VectorSpace, smt2_domain_from_base_domain
)
from guards import SimpleGuard
from optimizer import LinearTransform, get_annotations, is_annotated


SumOfTerms = Dict[str, int]


def ast_to_terms(node: ASTNode) -> Tuple[SumOfTerms, int]:
    """
    Convert an AST representing a linear integer expression into a mapping
    of symbolic terms to coefficients and a constant term.

    Args:
        node: Root of the AST to process.

    Returns:
        Tuple containing (terms, constant) where `terms` maps each symbolic 
        node to its integer coefficient, and `constant` is the summed integer constant.
    """
    def process(n: ASTNode) -> Tuple[Dict[str, int], int]:
        match n.type:
            case NodeType.INTEGER:
                return ({}, int(n.value))

            case NodeType.IDENTIFIER:
                return ({n.value: 1}, 0)

            case NodeType.NEG:
                if len(n.children) != 1:
                    raise ValueError("Negation operator expects exactly one operand.")
                
                child_terms, child_const = process(n.children[0])
                negated_terms = {sym: -coeff for sym, coeff in child_terms.items()}
                return (negated_terms, -child_const)

            case NodeType.PLUS:
                combined_terms: Dict[str, int] = {}
                combined_const: int = 0

                for term in n.children:
                    term_terms, term_const = process(term)
                    combined_const += term_const
                    for sym, coeff in term_terms.items():
                        combined_terms[sym] = combined_terms.get(sym, 0) + coeff
                return (combined_terms, combined_const)

            case NodeType.MINUS:
                if len(n.children) != 2:
                    raise ValueError("Minus operator expects exactly two operands.")
                
                left_terms, left_const = process(n.children[0])
                right_terms, right_const = process(n.children[1])

                combined_terms = dict(left_terms)
                for sym, coeff in right_terms.items():
                    combined_terms[sym] = combined_terms.get(sym, 0) - coeff
                
                return (combined_terms, left_const - right_const)

            case NodeType.MUL:
                product_terms: Dict[str, int] = {}
                product_const: int = 1

                for term in n.children:
                    term_terms, term_const = process(term)
                    
                    # Disallow multiplication of two non-constant expressions
                    if term_terms and product_terms:
                        raise ValueError(
                            "Invalid multiplication of two non-constant expressions"
                        )
                    
                    # Scale existing symbolic terms by the new constant
                    new_terms = {sym: coeff * term_const for sym, coeff in product_terms.items()}
                    
                    # Introduce new symbolic terms scaled by the accumulated constant
                    for sym, coeff in term_terms.items():
                        new_coeff = coeff * product_const
                        new_terms[sym] = new_terms.get(sym, 0) + new_coeff

                    product_const *= term_const
                    product_terms = new_terms
                    
                return (product_terms, product_const)

            case _:
                raise ValueError(f"Unknown node type: {n.type}")

    terms, const = process(node)
    # Clean zero coefficients
    return {s: c for s, c in terms.items() if c != 0}, const


def arithmetic_solver(left_terms: SumOfTerms, left_const: int,
                      right_terms: SumOfTerms, right_const: int,
                      vars: Set[str]) -> Tuple[SumOfTerms, SumOfTerms, int]:
    """
    Solve an equation for a list of variables.
    Returns the left side containing vars and their coefficients,
    and the right side with remaining vars, coefficients and a constant integer part.
    """
    # Separate variables to solve for vs. others
    left_vars = {k: v for k, v in left_terms.items() if k in vars}
    left_others = {k: -v for k, v in left_terms.items() if k not in vars}
    
    right_vars = {k: -v for k, v in right_terms.items() if k in vars}
    right_others = {k: v for k, v in right_terms.items() if k not in vars}

    # Move all target variables to the left
    new_left = {}
    for k, v in left_vars.items():
        new_left[k] = v + right_vars.get(k, 0)
    for k, v in right_vars.items():
        if k not in new_left:
            new_left[k] = v

    # Move all other variables to the right
    new_right = {}
    for k, v in right_others.items():
        new_right[k] = v + left_others.get(k, 0)
    for k, v in left_others.items():
        if k not in new_right:
            new_right[k] = v

    const = right_const - left_const

    # Clean up zero coefficients
    new_left = {s: c for s, c in new_left.items() if c != 0}
    new_right = {s: c for s, c in new_right.items() if c != 0}

    return (new_left, new_right, const)


def emit_annotations(node: IRNode, args: Tuple[str, ...]) -> List[str]:
    """Emit modulus guard annotations for a node."""
    guards = []
    if is_annotated(node):
        ann = get_annotations(node)
        if 'mod_guard' in ann:
            mod_guards: List[Tuple[int, ...]] = ann['mod_guard'][0]
            shifts = ann['mod_guard'][1]
            for scales in mod_guards:
                for i, scale in enumerate(scales):
                    if i < len(args) and scale > 1:
                        guards.append(f"(= (mod {args[i]} {scale}) {shifts[i]%scale})")
    return guards


def emit_node(node: IRNode, args: Tuple[str, ...]) -> str:
    """Recursively emit SMT for different IR node types, with modulus guards if annotated."""
    guards = emit_annotations(node, args)

    match node:
        case SetComprehension():
            body = emit_set_comprehension(node, args)
        case LinearTransform():
            body = emit_linear_transform(node, args)
        case UnionSet(parts=parts):
            parts_smt = [emit_node(p, args) for p in parts]
            body = f"(or {' '.join(parts_smt)})"
        case IntersectionSet(parts=parts):
            parts_smt = [emit_node(p, args) for p in parts]
            body = f"(and {' '.join(parts_smt)})"
        case ComplementSet(complemented_set=inner):
            inner_smt = emit_node(inner, args)
            body = f"(not {inner_smt})"
        case UnionSpace():
            raise NotImplementedError("UnionSpace emission not yet implemented")
        case VectorSpace():
            raise NotImplementedError("VectorSpace emission not yet implemented")
        case FiniteSet(elements=els):
            if not els:
                body = "false"
            else:
                elems_smt = [emit_node(e, args) for e in els]
                body = f"(or {' '.join(elems_smt)})"
        case ProductDomain():
            body = emit_product_domain(node, args)
        case Vector(comps=coords):
            if coords:
                eqs = [f"(= {args[i]} {val})" for i, val in enumerate(coords)]
                body = f"(and {' '.join(eqs)})"
            else:
                body = "true"
        case _:
            raise ValueError(f"Unsupported IR node type: {type(node).__name__}")

    # Combine guards and body
    if guards:
        return f"(and {' '.join(guards)} {body})"
    else:
        return body


def emit_set_comprehension(node: SetComprehension, args: Tuple[str, ...]) -> str:
    """Emit SMT for a set comprehension."""
    domain_smt = emit_node(node.domain, args)
    
    from guards import SetGuard
    
    if isinstance(node.guard, SimpleGuard):
        raise ValueError("Found guard without linear transform")
    elif isinstance(node.guard, SetGuard):
        set_expr_smt = emit_node(node.guard.set_expr, args)
        return f"(and {domain_smt} {set_expr_smt})"
    elif isinstance(node.guard, LinearTransform):
        guard_smt = emit_node(node.guard, args)
        return f"(and {domain_smt} {guard_smt})"
    else:
        raise ValueError(f"Unsupported guard type: {type(node.guard)}")


def emit_linear_transform(node: LinearTransform, args: Tuple[str, ...]) -> str:
    """Handle LinearTransform nodes."""
    guards = emit_annotations(node, args)

    # Handle different child types
    if isinstance(node.child, VectorSpace):
        body = emit_vector_space_transform(node, args)
    elif isinstance(node.child, SimpleGuard):
        body = emit_guard_condition(node.child, node, args)
    else:
        logging.warning(f"Unhandled linear transformation {node.child}")
        body = emit_node(node.child, args)

    # Combine guards and body
    if guards:
        return f"(and {' '.join(guards)} {body})"
    else:
        return body


def emit_vector_space_transform(node: LinearTransform, args: Tuple[str, ...]) -> str:
    """Emit SMT for a LinearTransform applied to a VectorSpace."""
    vector_space = node.child
    assert isinstance(vector_space, VectorSpace), "Child must be a VectorSpace"
    
    dim = vector_space.dimension
    if len(args) != dim:
        raise ValueError(f"Args length {len(args)} != vector space dimension {dim}")

    # Find the smallest step size per dimension
    basis_scales = []
    for i in range(dim):
        entries = [abs(vec[i]) for vec in vector_space.basis if vec[i] != 0]
        step = reduce(gcd, entries) if entries else 1
        basis_scales.append(step)

    unconstrained_dims = {
        i for i in range(dim)
        if all(vec[i] == 0 for vec in vector_space.basis)
    }

    conditions = []
    for i in range(dim):
        name = args[i]
        shift = node.shifts[i]
        scale = basis_scales[i] * node.scales[i]

        if i in unconstrained_dims:
            conditions.append(f"(= {name} {shift})")
        elif scale != 1:
            if shift == 0:
                conditions.append(f"(= (mod {name} {scale}) 0)")
            else:
                conditions.append(f"(= (mod (- {name} {shift}) {scale}) 0)")

    conditions.extend(emit_annotations(node.child, args))
    base_set_constraints = emit_product_domain(vector_space.domain, args)
    return f"(and {' '.join(conditions)} {base_set_constraints})"


def emit_guard_condition(guard: SimpleGuard, lt: LinearTransform | None, args: Tuple[str, ...]) -> str:
    """Emit SMT for a guard condition, with optional linear transformation."""
    if lt is not None:
        return emit_guard_ast(guard.node, guard, lt, args)
    else:
        return emit_guard_ast_simple(guard.node, guard, args)


def emit_guard_ast_simple(node: ASTNode, guard: SimpleGuard, args: Tuple[str, ...]) -> str:
    """Emit SMT for guard AST without linear transformation."""
    match node.type:
        case NodeType.AND:
            children_smt = [emit_guard_ast_simple(child, guard, args) for child in node.children]
            return f"(and {' '.join(children_smt)})"
        
        case NodeType.OR:
            children_smt = [emit_guard_ast_simple(child, guard, args) for child in node.children]
            return f"(or {' '.join(children_smt)})"
        
        case NodeType.NOT:
            assert len(node.children) == 1
            child_smt = emit_guard_ast_simple(node.children[0], guard, args)
            return f"(not {child_smt})"
        
        case NodeType.IMPLY:
            assert len(node.children) == 2
            antecedent = emit_guard_ast_simple(node.children[0], guard, args)
            consequent = emit_guard_ast_simple(node.children[1], guard, args)
            return f"(=> {antecedent} {consequent})"
        
        case NodeType.EQUIV:
            assert len(node.children) == 2
            left = emit_guard_ast_simple(node.children[0], guard, args)
            right = emit_guard_ast_simple(node.children[1], guard, args)
            return f"(= {left} {right})"
        
        case NodeType.EQ | NodeType.NEQ | NodeType.LT | NodeType.GT | NodeType.GEQ | NodeType.LEQ:
            return emit_relation_simple(node, guard, args)
        
        case NodeType.TRUE:
            return "true"
        
        case NodeType.FALSE:
            return "false"
        
        case _:
            raise ValueError(f"Unsupported node type in guard: {node.type}")


def emit_relation_simple(node: ASTNode, guard: SimpleGuard, args: Tuple[str, ...]) -> str:
    """Emit SMT for a relation without linear transformation."""
    assert len(node.children) == 2
    lhs_node, rhs_node = node.children
    
    lhs_smt = ast_node_to_smt(lhs_node, guard, args)
    rhs_smt = ast_node_to_smt(rhs_node, guard, args)
    
    relation_map = {
        NodeType.EQ: "=",
        NodeType.NEQ: "distinct",
        NodeType.LT: "<",
        NodeType.GT: ">",
        NodeType.GEQ: ">=",
        NodeType.LEQ: "<="
    }
    
    smt_relation = relation_map[node.type]
    
    if node.type == NodeType.NEQ:
        return f"(not (= {lhs_smt} {rhs_smt}))"
    else:
        return f"({smt_relation} {lhs_smt} {rhs_smt})"


def ast_node_to_smt(node: ASTNode, guard: SimpleGuard, args: Tuple[str, ...]) -> str:
    """Convert an AST node to SMT expression."""
    match node.type:
        case NodeType.IDENTIFIER:
            # Map identifier to corresponding argument
            if hasattr(guard, 'var_names_by_pos'):
                pos = next(i for i, name in guard.var_names_by_pos.items() if name == node.value)
            else:
                var_names = [var.name for var in guard.variables]
                pos = var_names.index(node.value)
            return args[pos]
        
        case NodeType.INTEGER:
            return str(node.value)
        
        case NodeType.PLUS:
            operands = [ast_node_to_smt(child, guard, args) for child in node.children]
            return f"(+ {' '.join(operands)})"
        
        case NodeType.MINUS:
            operands = [ast_node_to_smt(child, guard, args) for child in node.children]
            if len(operands) == 1:
                return f"(- {operands[0]})"
            else:
                return f"(- {' '.join(operands)})"
        
        case NodeType.MUL:
            operands = [ast_node_to_smt(child, guard, args) for child in node.children]
            return f"(* {' '.join(operands)})"
        
        case NodeType.DIV:
            operands = [ast_node_to_smt(child, guard, args) for child in node.children]
            return f"(div {' '.join(operands)})"
        
        case NodeType.MOD:
            operands = [ast_node_to_smt(child, guard, args) for child in node.children]
            return f"(mod {' '.join(operands)})"
        
        case NodeType.POWER:
            operands = [ast_node_to_smt(child, guard, args) for child in node.children]
            return f"(^ {' '.join(operands)})"
        
        case _:
            raise ValueError(f"Unsupported AST node type: {node.type}")


def emit_guard_ast(node: ASTNode, guard: SimpleGuard, lt: LinearTransform, args: Tuple[str, ...]) -> str:
    """Recursively walk the guard AST and emit SMT for logical connectives and relations."""
    match node.type:
        case NodeType.AND:
            children_smt = [emit_guard_ast(child, guard, lt, args) for child in node.children]
            return f"(and {' '.join(children_smt)})"
        
        case NodeType.OR:
            children_smt = [emit_guard_ast(child, guard, lt, args) for child in node.children]
            return f"(or {' '.join(children_smt)})"
        
        case NodeType.NOT:
            assert len(node.children) == 1
            child_smt = emit_guard_ast(node.children[0], guard, lt, args)
            return f"(not {child_smt})"
        
        case NodeType.IMPLY:
            assert len(node.children) == 2
            antecedent = emit_guard_ast(node.children[0], guard, lt, args)
            consequent = emit_guard_ast(node.children[1], guard, lt, args)
            return f"(=> {antecedent} {consequent})"
        
        case NodeType.EQUIV:
            assert len(node.children) == 2
            left = emit_guard_ast(node.children[0], guard, lt, args)
            right = emit_guard_ast(node.children[1], guard, lt, args)
            return f"(= {left} {right})"
        
        case NodeType.EQ | NodeType.NEQ | NodeType.LT | NodeType.GT | NodeType.GEQ | NodeType.LEQ:
            return emit_relation(node, guard, lt, args)
        
        case NodeType.TRUE:
            return "true"
        
        case NodeType.FALSE:
            return "false"
        
        case _:
            raise ValueError(f"Unsupported node type in guard: {node.type}")


def emit_relation(node: ASTNode, guard: SimpleGuard, lt: LinearTransform, args: Tuple[str, ...]) -> str:
    """Emit SMT for a single relational operator, applying the linear transformation."""
    l = lcm(*lt.scales)
    args_to_insert = [
        f"(- (* {l//lt.scales[i]} {args[i]}) {l*lt.shifts[i]})" 
        for i in range(len(args))
    ]
    guard_relation = node.type
    
    # Handle sign flip for negative lcm
    if l < 0:
        relation_flip = {
            NodeType.LT: NodeType.GT,
            NodeType.GT: NodeType.LT,
            NodeType.GEQ: NodeType.LEQ,
            NodeType.LEQ: NodeType.GEQ
        }
        guard_relation = relation_flip.get(guard_relation, guard_relation)
    
    assert len(node.children) == 2
    lhs_node, rhs_node = node.children
    
    sym_lhs, const_lhs = ast_to_terms(lhs_node)
    sym_rhs, const_rhs = ast_to_terms(rhs_node)
    solved_lhs, _, solved_const = arithmetic_solver(
        sym_lhs, const_lhs, sym_rhs, const_rhs, 
        {var.name for var in guard.variables}
    )
    
    ordered_name_keys = [guard.var_names_by_pos[i] for i in range(len(args))]
    
    # Build terms for variables present in solved_lhs
    terms = []
    for i, name_key in enumerate(ordered_name_keys):
        if name_key in solved_lhs:
            terms.append(f"(* {solved_lhs[name_key]} {args_to_insert[i]})")
    
    sum_of_terms = "0" if not terms else (
        terms[0] if len(terms) == 1 else f"(+ {' '.join(terms)})"
    )
    
    s_rhs = f"(* {l} {solved_const})"
    
    relation_map = {
        NodeType.EQ: "=",
        NodeType.NEQ: "distinct",
        NodeType.LT: "<",
        NodeType.GT: ">",
        NodeType.GEQ: ">=",
        NodeType.LEQ: "<="
    }
    
    smt_relation = relation_map[guard_relation]
    
    if guard_relation == NodeType.NEQ:
        return f"(not (= {sum_of_terms} {s_rhs}))"
    else:
        return f"({smt_relation} {sum_of_terms} {s_rhs})"


def emit_product_domain(node: ProductDomain, args: Tuple[str, ...]) -> str:
    """Emit SMT constraints for a ProductDomain."""
    constraints = []
    
    for i, domain_type in enumerate(node.types):
        arg_name = args[i]
        
        match domain_type:
            case BaseDomain.INT | BaseDomain.REAL:
                continue
            case BaseDomain.NAT:
                constraints.append(f"(>= {arg_name} 0)")
            case BaseDomain.POS:
                constraints.append(f"(> {arg_name} 0)")
    
    if not constraints:
        return "true"
    elif len(constraints) == 1:
        return constraints[0]
    else:
        return f"(and {' '.join(constraints)})"


def emit(node: IRNode, relation_name: str) -> str:
    """
    Main emit function - entry point for the IR tree.
    Emits SMT-LIB code for the given IR node.
    """
    assert isinstance(node, SetComprehension), f"Expected SetComprehension as root, got {type(node)}"
    assert isinstance(node.domain, ProductDomain), f"Expected ProductDomain as domain in root, got {type(node.domain)}"

    # Extract argument names and types
    arg_names = tuple(arg.name for arg in node.arguments)
    arg_types = tuple(smt2_domain_from_base_domain(t) for t in node.domain.types)

    # Combine names and types for SMT argument declarations
    arg_decls = ' '.join(f'({name} {typ})' for name, typ in zip(arg_names, arg_types))

    # Emit the body of the relation
    relation = emit_node(node, arg_names)

    return f"(define-fun {relation_name} ({arg_decls}) Bool {relation})"
