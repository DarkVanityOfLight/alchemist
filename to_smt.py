from __future__ import annotations
from string import Template
from typing import TYPE_CHECKING, Tuple, cast

from arm_ast import ARITHMETIC_OPERATIONS, BASE_SET_TYPES, SET_OPERATION_TYPES, ASTNode, NodeType, BaseSetType, SetOperationType
from scope_handler import ScopeHandler

from common import Variable, fresh_variable, Namespace, _smt_sort
from guards import SMTGuard, SimpleGuard, SetGuard
from sets import (
    BaseSet,
    ConstantScalar,
    SetExpression,
    TupleDomain,
    FiniteSet,
    SetComprehension,
    ConstantVector
)

if TYPE_CHECKING:
    from guards import Guard


def assert_node_type(node: ASTNode, expected: NodeType) -> None:
    if node.type != expected:
        raise AssertionError(
            f"Expected node of type {expected}, got {node.type} at line {node.line}"
        )


# ─── Guard Processing ────────────────────────────────────────────────────────────

def _process_guard(node: ASTNode, variables: Tuple[Variable, ...], scopes: ScopeHandler) -> Guard:
    assert node is not None and node.child is not None and node.child.next is not None

    if node.type == NodeType.IN:
        vec_node = node.child
        assert_node_type(vec_node, NodeType.VECTOR)
        assert vec_node.next is not None

        set_expr = _process_set_expression(vec_node.next, scopes)
        return SetGuard(variables, set_expr)

    return SimpleGuard(node, variables)


# ─── Set Expression Dispatch ─────────────────────────────────────────────────────

def _process_set_expression(node: ASTNode, scopes: ScopeHandler) -> SetExpression:
    assert node is not None

    if node.type in BASE_SET_TYPES:
        return BaseSet(BaseSetType(node.type))

    if node.type in SET_OPERATION_TYPES:
        operands = [_process_set_expression(child, scopes) for child in node]
        return _handle_set_operation(SetOperationType(node.type), tuple(operands))

    if node.type == NodeType.IDENTIFIER:
        return scopes[node.value]

    if node.type == NodeType.SET:
        return _process_set_comprehension(node, scopes)

    if node.type == NodeType.PAREN:

        if all(child.type in BASE_SET_TYPES for child in node.children):
            return _process_tuple_domain(node)

        if all(child.type == NodeType.INTEGER for child in node.children):
            return _process_constant_vector(node)

    if node.type in ARITHMETIC_OPERATIONS:
        return _process_arithmetic(node, scopes)

    if node.type == NodeType.VECTOR:
        # Process vector components
        components = []
        for child in node.children:
            if child.type != NodeType.INTEGER:
                raise NotImplementedError(f"Non-integer vector components not supported at line {child.line}")
            components.append(int(child.value))
        return ConstantVector(tuple(components))

    if node.type == NodeType.INTEGER:
        return ConstantScalar(node.value)

    node.print_tree()
    raise NotImplementedError(f"Unsupported set expression type: {node.type} at line {node.line}")


def _handle_set_operation(operation: SetOperationType, operands: Tuple[SetExpression, ...]) -> SetComprehension:
    assert operands, "No operands given"
    dims = [op.dim for op in operands]
    assert all(d == dims[0] for d in dims), "Operands have differing dimensions"
    dim = dims[0]

    # Create fresh result variables (outer variables)
    result_var_names = [fresh_variable(Namespace.ARGUMENT) for _ in range(dim)]
    result_vars = tuple(Variable(name, BaseSetType.INTEGERS) for name in result_var_names)

    # Create fresh bound variables for internal quantification
    bound_var_names = [fresh_variable(Namespace.BOUND_ARGUMENT) for _ in range(dim)]
    bound_vars = tuple(Variable(name, BaseSetType.INTEGERS) for name in bound_var_names)

    # Realize each operand’s membership constraints in terms of bound vars
    realized = [
        op.realize_constraints(tuple(bound_var_names))
        for op in operands
    ]

    # Add equality constraints linking result_vars to bound_vars (arg_i = bound_i)
    equality_constraints = [
        f"(= ${result_vars[i].name} {bound_vars[i].name})" for i in range(dim)
    ]

    # Combine constraints for the set operation
    match operation:
        case SetOperationType.UNION:
            op_clause = "(or " + " ".join(r for r in realized if r is not None) + ")"
        case SetOperationType.INTERSECTION:
            op_clause = "(and " + " ".join(r for r in realized if r is not None) + ")"
        case SetOperationType.DIFFERENCE:
            assert len(realized) == 2, "Difference expects exactly two operands"
            a, b = realized
            op_clause = f"(and {a} (not {b}))"
        case SetOperationType.XOR:
            assert len(realized) == 2, "XOR expects exactly two operands"
            a, b = realized
            op_clause = f"(or (and {a} (not {b})) (and {b} (not {a})))"
        case SetOperationType.COMPLEMENT:
            assert len(realized) == 1, "Complement expects exactly one operand"
            (a,) = realized
            op_clause = f"(not {a})"
        case SetOperationType.CARTESIAN_PRODUCT:
            raise NotImplementedError("Cartesian product is not implemented")

    # Combine operation clause and equality constraints
    all_constraints = "(and " + op_clause + " " + " ".join(equality_constraints) + ")"

    # Existential quantification over bound vars
    bound_decls = " ".join(f"({var.name} {_smt_sort(var.domain)})" for var in bound_vars)
    quantified = f"(exists ({bound_decls}) {all_constraints})"

    guard = SMTGuard(result_vars, Template(quantified))
    domain = TupleDomain(tuple(arg.domain for arg in result_vars))

    return SetComprehension(result_vars, domain, guard)


def _process_tuple_domain(node: ASTNode) -> TupleDomain:
    assert all(child.type in BASE_SET_TYPES for child in node.children)
    children = list(node)
    types = tuple(BaseSetType(child.type) for child in children)
    return TupleDomain(types)

def _process_constant_vector(node: ASTNode) -> ConstantVector:
    assert all(child.type == NodeType.INTEGER for child in node.children)
    return ConstantVector(tuple(child.value for child in node.children))


def _process_arithmetic(node: ASTNode, scopes: ScopeHandler) -> SetExpression:
    match node.type:
        case NodeType.MUL:
            return _process_mul(node, scopes)
        case NodeType.PLUS:
            return _process_plus(node, scopes)

    node.print_tree()
    raise NotImplementedError(f"Unimplemented arithmetic operator {node.type}")


# ─── Multiplication Cases ────────────────────────────────────────────────────────

def _process_mul(node: ASTNode, scopes: ScopeHandler) -> SetExpression:
    left_expr  = _process_set_expression(node.children[0], scopes)
    right_expr = _process_set_expression(node.children[1], scopes)
    comb = _combine_mul(left_expr, right_expr, scopes)
    return comb

def _combine_mul(left: SetExpression, right: SetExpression, scopes: ScopeHandler) -> SetExpression:
    # BaseSet × Vector
    if isinstance(left, BaseSet) and isinstance(right, ConstantVector):
        return _handle_base_mul_vector(left, right)
    if isinstance(right, BaseSet) and isinstance(left, ConstantVector):
        return _handle_base_mul_vector(right, left)

    # Vector × Arithmetic (not implemented)
    if isinstance(left, ConstantVector) or isinstance(right, ConstantVector):
        pass

    if isinstance(left, ConstantVector) and isinstance(right, ConstantScalar):
        return ConstantVector(tuple([right.value * comp for comp in left.components]))
    if isinstance(right, ConstantVector) and isinstance(left, ConstantScalar):
        return ConstantVector(tuple([left.value * comp for comp in right.components]))

    # BaseSet × Arithmetic
    if isinstance(left, BaseSet) and isinstance(right, SetComprehension):
        return _handle_base_mul_arithmetic(left, right)
    if isinstance(right, BaseSet) and isinstance(left, SetComprehension):
        return _handle_base_mul_arithmetic(right, left)

    # Scalar × Arithmetic
    if isinstance(left, ConstantScalar) and isinstance(right, SetComprehension):
        return _handle_scalar_mul_arithmetic(left, right)
    if isinstance(right, ConstantScalar) and isinstance(left, SetComprehension):
        return _handle_scalar_mul_arithmetic(right, left)
    raise NotImplementedError(f"Unhandled MUL types: {type(left)} * {type(right)}")

def _handle_scalar_mul_arithmetic(
    scalar: ConstantScalar, arithmetic: SetComprehension
) -> SetComprehension:
    # Create fresh result‐arguments matching arithmetic’s dimension
    arguments = [
        Variable(fresh_variable(Namespace.ARGUMENT), var.domain)
        for var in arithmetic.members
    ]

    # Create fresh bound vars that will satisfy the original arithmetic constraints
    bound_args = [
        Variable(fresh_variable(Namespace.BOUND_ARGUMENT), var.domain)
        for var in arithmetic.members
    ]

    # Realize the arithmetic’s own constraints over its bound vars
    arithmetic_constraint = arithmetic.realize_constraints(
        tuple(var.name for var in bound_args)
    )

    # Build multiplication constraints: (arg_i = (* scalar.value bound_i))
    mul_constraints = [
        f"(= ${arguments[i].name} (* {scalar.value} {bound_args[i].name}))"
        for i in range(arithmetic.dim)
    ]

    # Combine original and multiply constraints, filtering out any None
    parts = [arithmetic_constraint] + mul_constraints
    parts_filtered = [p for p in parts if p is not None]
    all_constraints = "(and " + " ".join(parts_filtered) + ")"

    # Existentially quantify over all bound vars
    bound_decls = " ".join(f"({var.name} {_smt_sort(var.domain)})" for var in bound_args)
    quantified = f"(exists ({bound_decls}) {all_constraints})"

    guard = SMTGuard(tuple(arguments), Template(quantified))
    domain = TupleDomain(tuple(arg.domain for arg in arguments))

    return SetComprehension(tuple(arguments), domain, guard)


def _handle_base_mul_vector(base_set: BaseSet, vector: ConstantVector) -> SetComprehension:
    existential_scalar = fresh_variable(Namespace.SCALAR)
    argument_names = [fresh_variable(Namespace.ARGUMENT) for _ in vector.components]
    argument_vars = tuple(Variable(name, base_set.set_type) for name in argument_names)
    
    # Create equality constraints using the argument names directly
    guard_clauses = []
    for i, coeff in enumerate(vector.components):
        arg_name = argument_names[i]
        guard_clauses.append(f"(= (* {existential_scalar} {coeff}) ${arg_name})")
    
    guard_body = "(and " + " ".join(guard_clauses) + ")"
    smt_guard = _build_smt_exists(base_set.set_type, existential_scalar, guard_body)
    
    # No Template needed since we're using concrete names
    guard = SMTGuard(argument_vars, Template(smt_guard))
    
    domain_types = tuple(base_set.set_type for _ in vector.components)
    domain = TupleDomain(domain_types)
    
    return SetComprehension(argument_vars, domain, guard)

def _handle_base_mul_arithmetic(base: BaseSet, arithmetic: SetComprehension) -> SetComprehension:
    raise NotImplementedError()


# ─── Addition Cases ─────────────────────────────────────────────────────────────

def _process_plus(node: ASTNode, scopes: ScopeHandler) -> SetExpression:
    left_expr = _process_set_expression(node.children[0], scopes)
    right_expr = _process_set_expression(node.children[1], scopes)
    return _combine_plus(left_expr, right_expr, scopes)

def _combine_plus(left: SetExpression, right: SetExpression, scopes: ScopeHandler) -> SetExpression:
    # BaseSet + Vector
    if isinstance(left, BaseSet) and isinstance(right, ConstantVector):
        return _handle_base_plus_vector(left, right)
    if isinstance(right, BaseSet) and isinstance(left, ConstantVector):
        return _handle_base_plus_vector(right, left)

    # Vector + Arithmetic
    if isinstance(left, ConstantVector) and isinstance(right, SetComprehension):
        return _handle_vector_plus_arithmetic(left, right)
    if isinstance(right, ConstantVector) and isinstance(left, SetComprehension):
        return _handle_vector_plus_arithmetic(right, left)

    # BaseSet + Arithmetic (not implemented)
    if isinstance(left, BaseSet) or isinstance(right, BaseSet):
        pass

    # Arithmetic + Arithmetic
    if isinstance(left, SetComprehension) and isinstance(right, SetComprehension):
        return _handle_arithmetic_plus_arithmetic(left, right)


    raise NotImplementedError(f"Unhandled PLUS types: {left} + {right}")

def _handle_base_plus_vector(base_set: BaseSet, vector: ConstantVector) -> SetComprehension:
    existential_scalar = fresh_variable(Namespace.SCALAR)
    argument_vars = tuple(Variable(fresh_variable(Namespace.ARGUMENT), base_set.set_type) for _ in vector.components)
    
    guard_clauses = []
    for i, coeff in enumerate(vector.components):
        arg_var = argument_vars[i]
        guard_clauses.append(f"(= (+ {existential_scalar} {coeff}) ${arg_var.name})")
    
    guard_body = "(and " + " ".join(guard_clauses) + ")"
    smt_guard = _build_smt_exists(base_set.set_type, existential_scalar, guard_body)
    smt_template = Template(smt_guard)
    guard = SMTGuard(argument_vars, smt_template)
    
    domain_types = tuple(base_set.set_type for _ in vector.components)
    domain = TupleDomain(domain_types)
    
    return SetComprehension(argument_vars, domain, guard)

def _handle_vector_plus_arithmetic(vector: ConstantVector, arithmetic: SetComprehension) -> SetComprehension:
    assert vector.dim == arithmetic.dim

    # Create new argument vars representing the result of the vector + arithmetic expression
    arguments = [Variable(fresh_variable(Namespace.ARGUMENT), arg.domain) for arg in arithmetic.members]

    # Create fresh bound vars that will satisfy the original arithmetic constraint
    bound_arguments = [Variable(fresh_variable(Namespace.BOUND_ARGUMENT), arg.domain) for arg in arithmetic.members]

    # Realize original constraints using the fresh bound vars
    constraint = arithmetic.realize_constraints(tuple(var.name for var in bound_arguments))

    # Construct vector offset constraints: (arg_i = (+ vector_i bound_i))
    offset_constraints = [
        f"(= ${arg.name} (+ {vector.components[i]} {bound_arguments[i].name}))"
        for i, arg in enumerate(arguments)
    ]

    # Combine constraints into a single guard
    parts = [constraint] + offset_constraints
    parts_filtered = [p for p in parts if p is not None]
    full_constraint = "(and " + " ".join(parts_filtered) + ")"

    # Wrap in existential quantifier over bound variables
    bound_decls = " ".join(f"({var.name} {_smt_sort(arg.domain)})" for var, arg in zip(bound_arguments, arithmetic.members))
    quantified = f"(exists ({bound_decls}) {full_constraint})"

    guard = SMTGuard(tuple(arguments), Template(quantified))
    domain = TupleDomain(tuple(arg.domain for arg in arguments))
    
    return SetComprehension(tuple(arguments), domain, guard)

def _handle_arithmetic_plus_arithmetic(
    left: SetComprehension, right: SetComprehension
) -> SetComprehension:
    assert left.dim == right.dim

    # Create fresh result‐arguments
    arguments = [
        Variable(fresh_variable(Namespace.ARGUMENT), var.domain)
        for var in left.members
    ]

    # Create fresh bound vars for left and right
    bound_args1 = [
        Variable(fresh_variable(Namespace.BOUND_ARGUMENT), var.domain)
        for var in left.members
    ]
    bound_args2 = [
        Variable(fresh_variable(Namespace.BOUND_ARGUMENT), var.domain)
        for var in right.members
    ]

    # Realize each set‐comprehension’s own constraints over its bound vars
    left_constraints = left.realize_constraints(tuple(arg.name for arg in bound_args1))
    right_constraints = right.realize_constraints(tuple(arg.name for arg in bound_args2))

    # Build sum‐constraints: arg_i = (+ bound1_i bound2_i)
    sum_constraints = [
        f"(= ${arguments[i].name} "
        f"(+ {bound_args1[i].name} {bound_args2[i].name}))"
        for i in range(left.dim)
    ]

    # Combine all constraints under one conjunction
    constraints = [left_constraints, right_constraints] + sum_constraints
    constraints_filtered = [c for c in constraints if c is not None]

    all_constraints = "(and " + " ".join(constraints_filtered) + ")"

    # Quantify existentially over both bound‐var lists
    bound_decls1 = " ".join(f"({var.name} {_smt_sort(var.domain)})" for var in bound_args1)
    bound_decls2 = " ".join(f"({var.name} {_smt_sort(var.domain)})" for var in bound_args2)
    quantified = f"(exists ({bound_decls1} {bound_decls2}) {all_constraints})"

    guard = SMTGuard(tuple(arguments), Template(quantified))
    domain = TupleDomain(tuple(arg.domain for arg in arguments))

    return SetComprehension(tuple(arguments), domain, guard)

# ─── SMT Helpers ────────────────────────────────────────────────────────────────

def _build_smt_exists(base_type: BaseSetType, scalar: str, body: str) -> str:
    match base_type:
        case BaseSetType.INTEGERS:
            return f"(exists (({scalar} Int)) {body})"
        case BaseSetType.NATURALS:
            return f"(exists (({scalar} Int)) (and (>= {scalar} 0) {body}))"
        case BaseSetType.POSITIVES:
            return f"(exists (({scalar} Int)) (and (> {scalar} 0) {body}))"
        case BaseSetType.REALS:
            return f"(exists (({scalar} Real)) {body})"
        case _:
            raise ValueError(f"No domain for type: {base_type}")


# ─── Set Comprehension ────────────────────────────────────────────────────────────

def _process_set_comprehension(node: ASTNode, scopes: ScopeHandler) -> SetComprehension:
    assert_node_type(node, NodeType.SET)

    members_in_domain = node.child
    assert members_in_domain and members_in_domain.type == NodeType.IN
    assert members_in_domain.child and members_in_domain.child.next
    assert members_in_domain.next

    member_names = _extract_vector_members(members_in_domain.child)
    domain_expr = _process_set_expression(members_in_domain.child.next, scopes)

    if isinstance(domain_expr, TupleDomain):
        members = tuple(
            Variable(name, typ) for name, typ in zip(member_names, domain_expr.types)
        )
    elif isinstance(domain_expr, SetComprehension):
        members = domain_expr.members
    else:
        members = tuple(Variable(name, BaseSetType.INTEGERS) for name in member_names)

    guard_node = members_in_domain.next
    guard = _process_guard(guard_node, members, scopes)
    return SetComprehension(members, domain_expr, guard)


def _extract_vector_members(vector_node: ASTNode) -> Tuple[str, ...]:
    assert_node_type(vector_node, NodeType.VECTOR)
    return tuple(child.value for child in vector_node.children)


# ─── Predicate and Definition ────────────────────────────────────────────────────

def _process_definition(node: ASTNode, scopes: ScopeHandler) -> None:
    assert_node_type(node, NodeType.DEFINITION)
    assert node.child and node.child.next

    ident = node.child.value
    value = _process_predicate(node.child.next, scopes)
    scopes.add_definition(ident, value)


def _process_predicate_context(node: ASTNode, scopes: ScopeHandler) -> None:
    assert_node_type(node, NodeType.PREDICATE_CONTEXT)
    for child in node:
        assert_node_type(child, NodeType.DEFINITION)
        _process_definition(child, scopes)


def _process_predicate(node: ASTNode, scopes: ScopeHandler) -> SetExpression:
    assert_node_type(node, NodeType.PREDICATE)
    assert node.child

    has_context = node.child.type == NodeType.PREDICATE_CONTEXT
    if has_context:
        scopes.enter_scope()
        _process_predicate_context(node.child, scopes)

    content_node = cast(ASTNode, node.child.next if has_context else node.child)
    predicate = _process_set_expression(content_node, scopes)

    if has_context:
        scopes.exit_scope()

    return predicate


def process_typed_tuple(node: ASTNode) -> TupleDomain:
    assert_node_type(node, NodeType.PAREN)
    types = []
    for child in node.children:
        match child.type:
            case NodeType.INTEGERS:
                types.append("INTEGERS")
            case NodeType.NATURALS:
                types.append("NATURALS")
            case NodeType.REALS:
                types.append("REALS")
            case _:
                raise Exception(f"Unknown type: {child.type}")
    return TupleDomain(tuple(types))


def convert(ast: ASTNode) -> SetComprehension:
    assert ast.child
    scopes = ScopeHandler()

    assert_node_type(ast, NodeType.PREDICATE)
    assert_node_type(ast.child, NodeType.PREDICATE_CONTEXT)

    result = _process_predicate(ast, scopes)
    assert isinstance(result, SetComprehension)
    return result
