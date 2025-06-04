from __future__ import annotations
from string import Template
from typing import TYPE_CHECKING, Tuple, cast

from arm_ast import ARITHMETIC_OPERATIONS, BASE_SET_TYPES, ASTNode, NodeType
from scope_handler import ScopeHandler

from common import Variable, SetType, get_fresh_variable, node_type_to_set_type
from guards import SMTGuard, SimpleGuard, SetGuard
from sets import (
    BaseSet,
    SetExpression,
    SetOperation,
    TupleDomain,
    FiniteSet,
    SetComprehension,
    Predicate,
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

    if _is_base_set(node):
        return BaseSet(node_type_to_set_type(node.type))

    if _is_set_operation(node):
        operands = [_process_set_expression(child, scopes) for child in node]
        return SetOperation(node.type, operands)

    if node.type == NodeType.IDENTIFIER:
        return scopes[node.value]

    if node.type == NodeType.SET:
        return _process_set_comprehension(node, scopes)

    if node.type == NodeType.PAREN:
        tuple_domain = _try_tuple_domain(node)
        if tuple_domain is not None:
            return tuple_domain

    if node.type == NodeType.MUL:
        return _process_mul(node, scopes)

    if node.type == NodeType.PLUS:
        # Not yet implemented, keep placeholder
        pass

    if node.type == NodeType.VECTOR:
        # Not yet implemented, keep placeholder
        pass

    node.print_tree()
    raise NotImplementedError(f"Unsupported set expression type: {node.type} at line {node.line}")


def _is_base_set(node: ASTNode) -> bool:
    return node.type in {
        NodeType.NATURALS,
        NodeType.INTEGERS,
        NodeType.POSITIVES,
        NodeType.REALS,
        NodeType.EMPTY,
    }


def _is_set_operation(node: ASTNode) -> bool:
    return node.type in {
        NodeType.UNION,
        NodeType.INTERSECTION,
        NodeType.DIFFERENCE,
        NodeType.XOR,
        NodeType.COMPLEMENT,
        NodeType.CARTESIAN_PRODUCT,
    }


def _try_tuple_domain(node: ASTNode) -> TupleDomain | None:
    children = list(node)
    if all(child.type in {NodeType.NATURALS, NodeType.INTEGERS, NodeType.POSITIVES, NodeType.REALS} for child in children):
        types = tuple(
            "Nat" if c.type == NodeType.NATURALS else
            "Int" if c.type == NodeType.INTEGERS else
            "Real"
            for c in children
        )
        return TupleDomain(cast(Tuple[SetType, ...], types))
    return None


# ─── Multiplication Cases ────────────────────────────────────────────────────────

def _process_mul(node: ASTNode, scopes: ScopeHandler) -> SetExpression:
    a, b = node.children
    types = (a.type, b.type)

    if _is_base_times_vector(types):
        base_node, vec_node = (a, b) if a.type in BASE_SET_TYPES else (b, a)
        return _handle_base_mul_vector(base_node, vec_node)

    if _is_identifier_times_integer(types):
        # Placeholder: arithmetic on a set by a scalar (not implemented)
        pass

    # Other MUL cases (e.g., Vector * Identifier) similarly unhandled => raise
    raise NotImplementedError(f"Unhandled MUL case: {a.type} * {b.type}")


def _is_base_times_vector(types: Tuple[NodeType, NodeType]) -> bool:
    t1, t2 = types
    return (
        (t1 in BASE_SET_TYPES and t2 == NodeType.PAREN)
        or (t1 == NodeType.PAREN and t2 in BASE_SET_TYPES)
    )


def _is_identifier_times_integer(types: Tuple[NodeType, NodeType]) -> bool:
    t1, t2 = types
    return (
        (t1 == NodeType.IDENTIFIER and t2 == NodeType.INTEGER)
        or (t1 == NodeType.INTEGER and t2 == NodeType.IDENTIFIER)
    )


def _handle_base_mul_vector(base_node: ASTNode, vec_node: ASTNode) -> SetComprehension:
    assert all(child.type == NodeType.INTEGER for child in vec_node.children), "Vector must be integers"

    # Fresh existential scalar
    existential_scalar = get_fresh_variable("scalar")

    # Create argument variables matching vector dimensions
    argument_vars = tuple(Variable(get_fresh_variable("argument"), None) for _ in vec_node.children)

    # Build SMT guard clauses "(= (* scalar coeff) $argument)"
    guard_clauses = [
        f"(= (* {existential_scalar} {coeff.value}) ${argument.name})"
        for coeff, argument in zip(vec_node.children, argument_vars)
    ]
    guard_body = f"(and {' '.join(guard_clauses)})"

    smt_guard = _build_smt_exists(base_node.type, existential_scalar, guard_body)
    smt_template = Template(smt_guard)
    guard = SMTGuard(argument_vars, smt_template)

    # Debug print remains exactly as before
    print(SetComprehension(argument_vars, BaseSet(node_type_to_set_type(base_node.type)), guard))

    return SetComprehension(
        argument_vars,
        BaseSet(node_type_to_set_type(base_node.type)),
        guard,
    )


def _build_smt_exists(base_type: NodeType, scalar: str, body: str) -> str:
    match base_type:
        case NodeType.INTEGERS:
            return f"(exists (Int {scalar}) {body})"
        case NodeType.NATURALS:
            return f"(exists (Int {scalar}) (and {body} (>= {scalar} 0)))"
        case NodeType.POSITIVES:
            return f"(exists (Int {scalar}) (and {body} (> {scalar} 0)))"
        case NodeType.REALS:
            return f"(exists (Real {scalar}) (and {body}))"
        case _:
            raise ValueError(f"No domain for type: {base_type}")


# ─── Set Comprehension ────────────────────────────────────────────────────────────

def _process_set_comprehension(node: ASTNode, scopes: ScopeHandler) -> SetComprehension:
    assert_node_type(node, NodeType.SET)

    # “members_in_domain” is the child of SET holding “IN <vector> <domain>” clause
    members_in_domain = node.child
    assert members_in_domain and members_in_domain.type == NodeType.IN
    assert members_in_domain.child and members_in_domain.child.next
    assert members_in_domain.next

    member_names = _extract_vector_members(members_in_domain.child)
    domain_expr = _process_set_expression(members_in_domain.child.next, scopes)

    # Determine member Variables (typed if TupleDomain, else default to INTEGERS)
    if isinstance(domain_expr, TupleDomain):
        members = tuple(
            Variable(name, typ) for name, typ in zip(member_names, domain_expr.types)
        )
    elif isinstance(domain_expr, SetComprehension):
        members = domain_expr.members
    else:
        members = tuple(Variable(name, "INTEGERS") for name in member_names)

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
