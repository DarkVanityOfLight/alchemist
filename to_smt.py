from __future__ import annotations
from string import Template
from typing import TYPE_CHECKING, Tuple, cast

from arm_ast import ARITHMETIC_OPERATIONS, BASE_SET_TYPES, ASTNode, NodeType, BaseSetType 
from scope_handler import ScopeHandler

from common import Variable, get_fresh_variable
from guards import SMTGuard, SimpleGuard, SetGuard
from sets import (
    BaseSet,
    SetExpression,
    SetOperation,
    TupleDomain,
    FiniteSet,
    SetComprehension,
    Predicate,
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

    if _is_set_operation(node):
        operands = [_process_set_expression(child, scopes) for child in node]
        return SetOperation(node.type, operands)

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

    node.print_tree()
    raise NotImplementedError(f"Unsupported set expression type: {node.type} at line {node.line}")


def _is_set_operation(node: ASTNode) -> bool:
    return node.type in {
        NodeType.UNION,
        NodeType.INTERSECTION,
        NodeType.DIFFERENCE,
        NodeType.XOR,
        NodeType.COMPLEMENT,
        NodeType.CARTESIAN_PRODUCT,
    }


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
    return _combine_mul(left_expr, right_expr, scopes)

def _combine_mul(left: SetExpression, right: SetExpression, scopes: ScopeHandler) -> SetExpression:
    # BaseSet × Vector
    if isinstance(left, BaseSet) and isinstance(right, ConstantVector):
        return _handle_base_mul_vector(left, right)
    if isinstance(right, BaseSet) and isinstance(left, ConstantVector):
        return _handle_base_mul_vector(right, left)

    # Vector × Arithmetic (not implemented)
    if isinstance(left, ConstantVector) or isinstance(right, ConstantVector):
        raise NotImplementedError(f"Vector multiplication not fully implemented at line")

    # BaseSet × Arithmetic (not implemented)
    if isinstance(left, BaseSet) or isinstance(right, BaseSet):
        raise NotImplementedError(f"Base set multiplication not fully implemented")

    raise NotImplementedError(f"Unhandled MUL types: {type(left)} * {type(right)}")

def _handle_base_mul_vector(base_set: BaseSet, vector: ConstantVector) -> SetComprehension:
    existential_scalar = get_fresh_variable("scalar")
    argument_vars = tuple(Variable(get_fresh_variable("arg"), base_set.set_type) for _ in vector.components)
    
    guard_clauses = []
    for i, coeff in enumerate(vector.components):
        arg_var = argument_vars[i]
        guard_clauses.append(f"(= (* {existential_scalar} {coeff}) ${arg_var.name})")
    
    guard_body = "(and " + " ".join(guard_clauses) + ")"
    smt_guard = _build_smt_exists(base_set.set_type, existential_scalar, guard_body)
    smt_template = Template(smt_guard)
    guard = SMTGuard(argument_vars, smt_template)
    
    domain_types = tuple(base_set.set_type for _ in vector.components)
    domain = TupleDomain(domain_types)
    
    return SetComprehension(argument_vars, domain, guard)


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

    # Vector + Arithmetic (not implemented)
    if isinstance(left, ConstantVector) and isinstance(right, SetComprehension):
        return _handle_vector_plus_arithmetic(left, right)
    if isinstance(right, ConstantVector) and isinstance(left, SetComprehension):
        return _handle_vector_plus_arithmetic(right, left)

    # BaseSet + Arithmetic (not implemented)
    if isinstance(left, BaseSet) or isinstance(right, BaseSet):
        raise NotImplementedError(f"Base set addition not fully implemented")

    raise NotImplementedError(f"Unhandled PLUS types: {type(left)} + {type(right)}")

def _handle_base_plus_vector(base_set: BaseSet, vector: ConstantVector) -> SetComprehension:
    existential_scalar = get_fresh_variable("scalar")
    argument_vars = tuple(Variable(get_fresh_variable("arg"), base_set.set_type) for _ in vector.components)
    
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
    arguments = [Variable(get_fresh_variable("arg"), arg.domain) for arg in arithmetic.members]

    # Create fresh bound vars that will satisfy the original arithmetic constraint
    bound_arguments = [Variable(get_fresh_variable("bound"), arg.domain) for arg in arithmetic.members]

    # Realize original constraints using the fresh bound vars
    constraint = arithmetic.realize_constraints(tuple(var.name for var in bound_arguments))

    # Construct vector offset constraints: (arg_i = (+ vector_i bound_i))
    offset_constraints = [
        f"(= ${arg.name} (+ {vector.components[i]} ${bound_arguments[i].name}))"
        for i, arg in enumerate(arguments)
    ]

    # Combine constraints into a single guard
    full_constraint = "(and " + " ".join([constraint] + offset_constraints) + ")"

    # Wrap in existential quantifier over bound variables
    bound_decls = " ".join(f"({arg.domain} {var.name})" for var, arg in zip(bound_arguments, arithmetic.members))
    quantified = f"(exists ({bound_decls}) {full_constraint})"

    guard = SMTGuard(tuple(arguments), Template(quantified))
    domain = TupleDomain(tuple(arg.domain for arg in arguments))
    
    return SetComprehension(tuple(arguments), domain, guard)

# ─── SMT Helpers ────────────────────────────────────────────────────────────────

def _build_smt_exists(base_type: BaseSetType, scalar: str, body: str) -> str:
    match base_type:
        case BaseSetType.INTEGERS:
            return f"(exists ((Int {scalar})) {body})"
        case BaseSetType.NATURALS:
            return f"(exists ((Int {scalar})) (and (>= {scalar} 0) {body}))"
        case BaseSetType.POSITIVES:
            return f"(exists ((Int {scalar})) (and (> {scalar} 0) {body}))"
        case BaseSetType.REALS:
            return f"(exists ((Real {scalar})) {body})"
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


def _process_predicate(node: ASTNode, scopes: ScopeHandler) -> Predicate:
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
