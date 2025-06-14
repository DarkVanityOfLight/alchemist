from __future__ import annotations
from typing import Dict, Set, Tuple
from arm_ast import ASTNode, NodeType
from expressions import BaseDomain, ComplementSet, FiniteSet, IRNode, IntersectionSet, ProductDomain, SetComprehension, UnionSet, UnionSpace, Vector, VectorSpace, smt2_domain_from_base_domain
from guards import SimpleGuard
from optimizer import LinearTransform, get_annotations, is_annotated

from math import lcm


SumOfTerms = Dict[str, int]
def ast_to_terms(node: ASTNode) -> Tuple[SumOfTerms, int]:
    """
    Convert an AST representing a linear integer expression into a mapping
    of symbolic terms to coefficients and a constant term.

    Args:
        node (ASTNode): Root of the AST to process.

    Returns:
        Tuple[Dict[ASTNode, int], int]: A pair (terms, constant) where `terms` maps
        each symbolic node to its integer coefficient, and `constant` is the summed
        integer constant.
    """
    def process(n: ASTNode) -> Tuple[Dict[str, int], int]:
        T = n.type
        match T:
            case NodeType.INTEGER:
                # For integer constants, no symbolic terms, just return its value
                return ({}, int(n.value))

            case NodeType.IDENTIFIER:
                # Symbols represent variables; Return a single-term map: this symbol has coefficient 1
                return ({n.value: 1}, 0)

            case NodeType.NEG:
                # Handle negation: multiply all terms and constant by -1
                if len(n.children) != 1:
                    raise ValueError("Negation operator expects exactly one operand.")
                
                child_terms, child_const = process(n.children[0])
                negated_terms = {sym: -coeff for sym, coeff in child_terms.items()}
                return (negated_terms, -child_const)

            case NodeType.PLUS:
                # Handle addition: sum up terms and constants from all operands
                combined_terms: Dict[str, int] = {}
                combined_const: int = 0

                for term in n.children: # Use n.children to get all direct children
                    term_terms, term_const = process(term)
                    # Accumulate constant parts
                    combined_const += term_const
                    # Merge symbolic terms, adding coefficients
                    for sym, coeff in term_terms.items():
                        combined_terms[sym] = combined_terms.get(sym, 0) + coeff
                return (combined_terms, combined_const)

            case NodeType.MINUS:
                # Assuming binary minus for simplicity based on original code's arg(0) and arg(1)
                # The first child is the left operand, the second is the right.
                children_list = n.children
                if len(children_list) != 2:
                    raise ValueError("Minus operator expects exactly two operands.")
                
                left_terms, left_const = process(children_list[0])
                right_terms, right_const = process(children_list[1])

                combined_terms = {}

                for sym, coeff in left_terms.items():
                    combined_terms[sym] = coeff

                for sym, coeff in right_terms.items():
                    if sym in combined_terms:
                        combined_terms[sym] -= coeff
                    else:
                        combined_terms[sym] = -coeff
                return (combined_terms, left_const - right_const)

            case NodeType.MUL:
                # Handle multiplication: only allow one symbolic factor
                product_terms: Dict[str, int] = {}
                product_const: int = 1

                for term in n.children: # Use n.children to get all direct children
                    term_terms, term_const = process(term)
                    # Disallow multiplication of two non-constant expressions
                    if term_terms and product_terms:
                        raise ValueError(
                            "Invalid multiplication of two non-constant expressions"
                        )
                    # Scale existing symbolic terms by the new constant
                    new_terms: Dict[str, int] = {}
                    for sym, coeff in product_terms.items():
                        new_terms[sym] = coeff * term_const

                    # Introduce new symbolic terms scaled by the accumulated constant
                    for sym, coeff in term_terms.items():
                        new_coeff = coeff * product_const
                        new_terms[sym] = new_terms.get(sym, 0) + new_coeff

                    # Update constant multiplier
                    product_const *= term_const
                    product_terms = new_terms
                return (product_terms, product_const)

            case _:
                # Catch-all for any unrecognized node types
                raise ValueError(f"Unknown node type: {T}")

    terms, const = process(node)
    # Clean 0 coeffs
    return {s: c for s, c in terms.items() if c != 0}, const

def arithmetic_solver(left_terms: SumOfTerms, left_const: int,
                      right_terms: SumOfTerms, right_const: int,
                      vars: Set[str]) -> Tuple[SumOfTerms, SumOfTerms, int]:
    """
    Solve an sum of products for a list of variables.
    Returns the left side only containing vars and their coefficients,
    and the right side with vars, coefficients and a constant integer part.
    """

    assert isinstance(vars, set) # Sets speed up the process alot so make sure we get a set

    Lw, Lo = {}, {}
    for k, v in left_terms.items():
        if k in vars:
            Lw[k] = v
        else:
            Lo[k] = -v # Is moved to the other side so substracted

    Rw, Ro = {}, {}
    for k, v in right_terms.items():
        if k in vars:
            Rw[k] = -v # Is moved to the other side so substracted
        else:
            Ro[k] = v

    # Move all variables with vars to the left
    new_left = {}
    # Combine Lw and Rw, handling common keys by adding coefficients
    for k, v in Lw.items():
        new_left[k] = v + Rw.get(k, 0)
    for k, v in Rw.items():
        if k not in new_left: # Add keys from Lw that were not in Rw
            new_left[k] = v

    # Move all variables without vars to the right
    new_right = {}
    # Combine Ro and Lo, handling common keys by adding coefficients
    for k, v in Ro.items():
        new_right[k] = v + Lo.get(k, 0)
    for k, v in Lo.items():
        if k not in new_right: # Add keys from Lo that were not in Ro
            new_right[k] = v

    const = right_const - left_const

    # Clean up zero coefficients
    new_left = {s: c for s, c in new_left.items() if c != 0}
    new_right = {s: c for s, c in new_right.items() if c != 0}

    return (new_left, new_right, const)


def emit_node(node: IRNode, args: Tuple[str, ...]) -> str:
    """
    Recursively emit SMT for different IR node types, with modulus guards if annotated.
    """
    # check for mod_guard annotations and prepare guards
    guards = []
    if is_annotated(node):
        ann = get_annotations(node)
        if 'mod_guard' in ann:
            scales = ann['mod_guard']  # expected to be a tuple of int scales
            for i, scale in enumerate(scales):
                # only emit if scale > 0 and within args
                if i < len(args) and scale > 0:
                    guards.append(f"(= (mod {args[i]} {scale}) (mod 0 {scale}))")

    # emit main body
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
            eqs = []
            for i, val in enumerate(coords):
                eqs.append(f"(= {args[i]} {val})")
            body = "true" if not eqs else f"(and {' '.join(eqs)})"
        case _:
            raise ValueError(f"Unsupported IR node type: {type(node).__name__}")

    # combine guards and body
    if guards:
        return f"(and {' '.join(guards)} {body})"
    else:
        return body


def emit_set_comprehension(node: SetComprehension, args: Tuple[str, ...]) -> str:
    """
    Emit SMT for a set comprehension.
    For the root set comprehension, we emit the domain and guard conditions.
    """
    # For nested set comprehensions, we need to handle the domain and guard
    domain_smt = emit_node(node.domain, args)
    
    # Handle different guard types
    from guards import SimpleGuard, SetGuard
    
    if isinstance(node.guard, SimpleGuard):
        raise ValueError("Found guard without linear transform")
    elif isinstance(node.guard, SetGuard):
        # Handle set membership guards
        set_expr_smt = emit_node(node.guard.set_expr, args)
        return f"(and {domain_smt} {set_expr_smt})"
    elif isinstance(node.guard, LinearTransform):
        # Handle LinearTransform as a guard
        guard_smt = emit_node(node.guard, args)
        return f"(and {domain_smt} {guard_smt})"
    else:
        raise ValueError(f"Unsupported guard type: {type(node.guard)}")


def emit_vector_space_transform(node: LinearTransform, args: Tuple[str, ...]) -> str:
    """
    Generates the condition for a transformed VectorSpace.
    """
    vector_space = node.child
    assert isinstance(vector_space, VectorSpace)
    scales = node.scales
    shifts = node.shifts
    
    num_dims = vector_space.dimension
    unconstrained_dims = set()
    for basis_vec in vector_space.basis:
        for i in range(num_dims):
            if basis_vec[i] != 0:
                unconstrained_dims.add(i)

    conditions = []
    for i in range(num_dims):
        arg = args[i]
        scale = scales[i]
        shift = shifts[i]

        if i in unconstrained_dims:
            if shift == 0:
                conditions.append(f"(= (mod {arg} {scale}) 0)")
            else:
                conditions.append(f"(= (mod (- {arg} {shift}) {scale}) 0)")
        else:
            conditions.append(f"(= {arg} {shift})")

    return f"(and {' '.join(conditions)})"


def emit_linear_transform(node: LinearTransform, args: Tuple[str, ...]) -> str:
    """
    Handle LinearTransform nodes.
    """
    # Check if child is SetComprehension with SimpleGuard
    if isinstance(node.child, SetComprehension):
        assert False, "Hit weird case"
        set_comp = node.child
        if isinstance(set_comp.guard, SimpleGuard):
            return emit_linear_transformed_set_comprehension(node, args)
    if isinstance(node.child, SimpleGuard):
        return emit_guard_condition(node.child, node, args)
    
    # For other cases, recurse down

    if isinstance(node.child, VectorSpace):
        return emit_vector_space_transform(node, args)
    assert False, f"Can not linear transform {node.child}"

    return emit_node(node.child, args)

def emit_guard_condition(guard: SimpleGuard, lt: LinearTransform | None, args: Tuple[str, ...]) -> str:
    """
    Emit SMT for a guard condition, with optional linear transformation.
    """
    if lt is not None:
        return emit_guard_ast(guard.node, guard, lt, args)
    else:
        # No linear transformation - emit the guard as-is
        print("This shouldn't happen")
        return emit_guard_ast_simple(guard.node, guard, args)

def emit_guard_ast_simple(node: ASTNode, guard: SimpleGuard, args: Tuple[str, ...]) -> str:
    """
    Emit SMT for guard AST without linear transformation.
    """
    match node.type:
        # Logical connectives
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
        
        # Relational operators - emit without transformation
        case NodeType.EQ | NodeType.NEQ | NodeType.LT | NodeType.GT | NodeType.GEQ | NodeType.LEQ:
            return emit_relation_simple(node, guard, args)
        
        case NodeType.TRUE:
            return "true"
        
        case NodeType.FALSE:
            return "false"
        
        case _:
            raise ValueError(f"Unsupported node type in guard: {node.type}")


def emit_relation_simple(node: ASTNode, guard: SimpleGuard, args: Tuple[str, ...]) -> str:
    """
    Emit SMT for a relation without linear transformation.
    """
    assert len(node.children) == 2
    lhs_node = node.children[0]
    rhs_node = node.children[1]
    
    # Convert AST nodes to SMT expressions
    lhs_smt = ast_node_to_smt(lhs_node, guard, args)
    rhs_smt = ast_node_to_smt(rhs_node, guard, args)
    
    # Map NodeType to SMT relation symbols
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
    """
    Convert an AST node to SMT expression.
    This handles variables, constants, and arithmetic operations.
    """
    match node.type:
        case NodeType.IDENTIFIER:
            # Map identifier to corresponding argument
            if hasattr(guard, 'var_names_by_pos'):
                pos = next(i for i, name in guard.var_names_by_pos.items() if name == node.value)
                return args[pos]
            else:
                # Fallback - find position by name
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
                return f"(- {operands[0]})"  # Unary minus
            else:
                return f"(- {' '.join(operands)})"  # Binary minus
        
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
            # Note: SMT-LIB doesn't have a standard power operator
            # This might need special handling depending on the solver
            return f"(^ {' '.join(operands)})"
        
        case _:
            raise ValueError(f"Unsupported AST node type: {node.type}")


def emit_linear_transformed_set_comprehension(node: IRNode, args: Tuple[str, ...]):
    assert isinstance(node, LinearTransform)
    lt = node
    set_comp = node.child
    assert isinstance(set_comp, SetComprehension)
    guard = set_comp.guard
    
    # Handle boolean combinations of (in)equalities
    if isinstance(guard, SimpleGuard):
        return emit_guard_condition(guard, lt, args)
    else:
        raise ValueError(f"Unsupported guard type: {type(guard)}")


def emit_guard_ast(node: ASTNode, guard: SimpleGuard, lt: LinearTransform, args: Tuple[str, ...]) -> str:
    """
    Recursively walk the guard AST and emit SMT for logical connectives and relations.
    """
    match node.type:
        # Logical connectives
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
        
        # Relational operators - this is where we apply the linear transformation
        case NodeType.EQ | NodeType.NEQ | NodeType.LT | NodeType.GT | NodeType.GEQ | NodeType.LEQ:
            return emit_relation(node, guard, lt, args)
        
        case NodeType.TRUE:
            return "true"
        
        case NodeType.FALSE:
            return "false"
        
        case _:
            raise ValueError(f"Unsupported node type in guard: {node.type}")


def emit_relation(node: ASTNode, guard: SimpleGuard, lt: LinearTransform, args: Tuple[str, ...]) -> str:
    """
    Emit SMT for a single relational operator, applying the linear transformation.
    """
    l = lcm(*lt.scales)
    args_to_insert = [f"(+ (* {l//lt.scales[i]} {args[i]}) {l*lt.shifts[i]})" for i in range(len(args))]
    guard_relation = node.type
    
    # Handle sign flip for negative lcm
    if l < 0:
        match guard_relation:
            case NodeType.EQ: pass
            case NodeType.NEQ: pass
            case NodeType.LT: guard_relation = NodeType.GT
            case NodeType.GT: guard_relation = NodeType.LT
            case NodeType.GEQ: guard_relation = NodeType.LEQ
            case NodeType.LEQ: guard_relation = NodeType.GEQ
    
    assert len(node.children) == 2
    lhs_node = node.children[0]
    rhs_node = node.children[1]
    
    sym_lhs, const_lhs = ast_to_terms(lhs_node)
    sym_rhs, const_rhs = ast_to_terms(rhs_node)
    solved_lhs, _, solved_const = arithmetic_solver(sym_lhs, const_lhs, sym_rhs, const_rhs, {var.name for var in guard.variables})
    
    ordered_name_keys = [guard.var_names_by_pos[i] for i in range(len(args))]
    
    # Only include terms for variables that are present in solved_lhs
    terms = []
    for i, name_key in enumerate(ordered_name_keys):
        if name_key in solved_lhs:
            terms.append(f"(* {solved_lhs[name_key]} {args_to_insert[i]})")
    
    # Handle case where no terms are present
    if not terms:
        sum_of_terms = "0"
    else:
        sum_of_terms = f"(+ {' '.join(terms)})" if len(terms) > 1 else terms[0]
    
    s_rhs = f"(* {l} {solved_const})"
    
    # Map NodeType to SMT relation symbols
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
    """
    Emit SMT constraints for a ProductDomain.
    Each component type has specific constraints:
    - INT/REAL: no constraints (true)
    - NAT: >= 0 
    - POS: > 0
    """
    constraints = []
    
    for i, domain_type in enumerate(node.types):
        arg_name = args[i]
        
        match domain_type:
            case BaseDomain.INT | BaseDomain.REAL:
                # No constraints for integers and reals
                continue
            case BaseDomain.NAT:
                # Natural numbers: >= 0
                constraints.append(f"(>= {arg_name} 0)")
            case BaseDomain.POS:
                # Positive naturals: > 0
                constraints.append(f"(> {arg_name} 0)")
            case _:
                raise ValueError(f"Unknown domain type: {domain_type}")
    
    if not constraints:
        # No constraints needed
        return "true"
    elif len(constraints) == 1:
        # Single constraint
        return constraints[0]
    else:
        # Multiple constraints - combine with AND
        return f"(and {' '.join(constraints)})"


def emit(node: IRNode, relation_name: str) -> str:
    """
    Main emit function - entry point for the IR tree.
    Emits SMT-LIB code for the given IR node.
    """
    # First node must be a SetComprehension
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
