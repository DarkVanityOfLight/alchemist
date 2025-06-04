from __future__ import annotations
from string import Template
from types import NotImplementedType
from typing import Tuple, cast, List
from arm_ast import ARITHMETIC_OPERATIONS, BASE_SET_TYPES, ASTNode, NodeType
from scope_handler import ScopeHandler

from common import Variable, SetType, get_fresh_variable, node_type_to_set_type
from guards import SMTGuard, SimpleGuard, SetGuard
from sets import BaseSet, SetExpression, SetOperation, TupleDomain, FiniteSet, SetComprehension, Predicate, SetType


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from guards import Guard

def assert_node_type(node: ASTNode, expected_type):
    if node.type != expected_type:
        raise AssertionError(
            f"Expected node of type {expected_type}, got {node.type} "
            f"at line {node.line}"
        )

def _process_guard(node: ASTNode, variables: Tuple[Variable, ...], scopes: ScopeHandler) -> Guard:
    assert node is not None
    assert node.child is not None
    assert node.child.next is not None

    # Set guard
    if node.type == NodeType.IN:
        arguments = node.child
        assert arguments is not None
        assert_node_type(arguments, NodeType.VECTOR)
        set_expr = _process_set_expression(node.child.next, scopes)
        return SetGuard(variables, set_expr)

    # Formula guard
    return SimpleGuard(node, variables)

def _process_members(node: ASTNode) -> Tuple[str, ...]:
    assert_node_type(node, NodeType.VECTOR)
    return tuple(child.value for child in node)

def _process_set_expression(node: ASTNode, scopes: ScopeHandler) -> SetExpression:
    # ─── 1) Base sets ───────────────────────────────────────────────────────────
    if node.type in [
        NodeType.NATURALS, NodeType.INTEGERS,
        NodeType.POSITIVES, NodeType.REALS, NodeType.EMPTY
    ]:
        return BaseSet(node_type_to_set_type(node.type))

    # ─── 2) Set operations ─────────────────────────────────────────────────────
    if node.type in [
        NodeType.UNION, NodeType.INTERSECTION, NodeType.DIFFERENCE,
        NodeType.XOR, NodeType.COMPLEMENT, NodeType.CARTESIAN_PRODUCT
    ]:
        operands = [_process_set_expression(child, scopes) for child in node]
        return SetOperation(node.type, operands)

    # ─── 3) Identifiers ───────────────────────────────────────────────────────
    if node.type == NodeType.IDENTIFIER:
        return scopes[node.value]

    # ─── 4) Set comprehensions ─────────────────────────────────────────────────
    if node.type == NodeType.SET:
        return _process_set_comprehension(node, scopes)

    # ─── 5) PAREN: either TupleDomain or a single‐vector FiniteSet ────────────
    if node.type == NodeType.PAREN:
        # Check if all children are base sets (tuple domain)
        if all(child.type in [NodeType.NATURALS, NodeType.INTEGERS, 
                            NodeType.POSITIVES, NodeType.REALS] 
               for child in node):
            types = tuple("Nat" if c.type == NodeType.NATURALS else 
                         "Int" if c.type == NodeType.INTEGERS else
                         "Real" for c in node)
            return TupleDomain(cast(Tuple[SetType, ...], types))

    # ─── 6) Multiplication (MUL) ─────────────────────────────────────────────────
    if node.type == NodeType.MUL:
        a, b = node.children
        match (a.type, b.type):
            # Set * Vector => Hadamard
            case (NodeType.IDENTIFIER, NodeType.PAREN) | (NodeType.PAREN, NodeType.IDENTIFIER):
                set_node = a if a.type is NodeType.IDENTIFIER else b
                vec_node = b if a.type is NodeType.IDENTIFIER else a

                pass

            # BaseSet * Vector => SetComprehension
            case (t, NodeType.PAREN) if t in BASE_SET_TYPES:
                base_node = a
                vec_node = b # Vector of integers

                assert all(child.type == NodeType.INTEGER for child in vec_node.children)

                # Make variables from vector
                existential_scalar = get_fresh_variable("scalar")

                # Create a argument vector of these dimensions
                argument_vector = tuple([Variable(get_fresh_variable("argument"), None) for _ in vec_node])
                guards = [f"(= (* {existential_scalar} {coeff}) ${argument.name})"
                    for coeff, argument in zip(vec_node, argument_vector)]

                guard_expr = f"(and {' '.join(guards)})"
                smt_guard = ""
                match base_node.type: 
                    case NodeType.INTEGERS: 
                        smt_guard = f"(exists (Int {existential_scalar}) {guard_expr})"
                    case NodeType.NATURALS:
                        smt_guard = f"(exists (Int {existential_scalar}) (and {guard_expr} (>= {existential_scalar} 0))"
                    case NodeType.POSITIVES:
                        smt_guard = f"(exists (Int {existential_scalar}) (and {guard_expr} (> {existential_scalar} 0))"
                    case NodeType.REALS:
                        smt_guard = f"(exists (Real {existential_scalar}) (and {guard_expr})"
                    case _:
                        raise ValueError(f"No domain of type: {base_node.type}")

                template = Template(smt_guard)
                guard = SMTGuard(argument_vector, template)
                print(SetComprehension(argument_vector, BaseSet(node_type_to_set_type(base_node.type)), guard))


                return SetComprehension(argument_vector,
                                        BaseSet(node_type_to_set_type(base_node.type)),
                                        guard)

            case (NodeType.PAREN, t) if t in BASE_SET_TYPES:
                base_node = b
                vec_node = a

                pass

            # Set * Scalar => Arithmetic
            case (NodeType.IDENTIFIER, NodeType.INTEGER) | (NodeType.INTEGER, NodeType.IDENTIFIER):
                set_node = a if a.type is NodeType.IDENTIFIER else b
                scalar_node = b if a.type is NodeType.IDENTIFIER else a

                pass

            case _:
                raise NotImplementedError(f"Unhandled MUL case: {a.type} * {b.type}")

    # ─── 7) Sum (PLUS) ──────────────────────────────────────────────────────────
    if node.type == NodeType.PLUS:
        pass #TODO: Not Implemented

    # ─── 8) Vector literal ──────────────────────────────────────────────────────
    if node.type == NodeType.VECTOR:
        pass #TODO: Not Implemented

    # ─── 9) Anything else is unhandled ──────────────────────────────────────────
    node.print_tree()
    raise NotImplementedError(f"Unsupported set expression type: {node.type} in line: {node.line}")

def _process_set_comprehension(node: ASTNode, scopes: ScopeHandler) -> SetComprehension:
    assert_node_type(node, NodeType.SET)
    members_in_domain = node.child

    assert members_in_domain is not None
    assert_node_type(members_in_domain, NodeType.IN)
    assert members_in_domain.child is not None
    assert members_in_domain.child.next is not None
    assert members_in_domain.next is not None

    member_names = _process_members(members_in_domain.child)
    domain_expr = _process_set_expression(members_in_domain.child.next, scopes)
    
    # Create members with appropriate domains
    members: Tuple[Variable, ...]
    if isinstance(domain_expr, TupleDomain):
        members = tuple(Variable(name, typ) for name, typ in zip(member_names, domain_expr.types))
    elif isinstance(domain_expr, SetComprehension):
        members = domain_expr.members
    else:
        # For other set expressions, assume all integers
        members = tuple(Variable(name, "INTEGERS") for name in member_names)
    
    guard = _process_guard(members_in_domain.next, members, scopes)
    return SetComprehension(members, domain_expr, guard)

def _process_definition(node: ASTNode, scopes: ScopeHandler):
    """Process the definition into scope"""
    assert node.child is not None
    assert node.child.next is not None
    assert_node_type(node, NodeType.DEFINITION)
    
    ident = node.child.value
    value = _process_predicate(node.child.next, scopes)

    scopes.add_definition(ident, value)

def _process_predicate_context(node: ASTNode, scopes: ScopeHandler):
    """Process the predicate context into the current scope"""
    assert_node_type(node, NodeType.PREDICATE_CONTEXT)

    for child in node:
        assert_node_type(child, NodeType.DEFINITION)
        _process_definition(child, scopes)

def process_typed_tuple(node: ASTNode) -> TupleDomain:
    assert_node_type(node, NodeType.PAREN)
    children = [child for child in node]

    def _map(node: ASTNode) -> SetType:
        match node.type:
            case NodeType.INTEGERS: return "INTEGERS"
            case NodeType.NATURALS: return "NATURALS"
            case NodeType.REALS: return "REALS"
        raise Exception("Unknown type")

    mapped_children = cast(Tuple[SetType, ...], tuple(map(_map, children)))
    return TupleDomain(mapped_children)
            
def _process_predicate(node: ASTNode, scopes: ScopeHandler) -> Predicate:
    assert_node_type(node, NodeType.PREDICATE)
    assert node.child is not None
    assert node.child.next is not None

    has_context = node.child.type == NodeType.PREDICATE_CONTEXT

    if has_context:
        scopes.enter_scope()
        _process_predicate_context(node.child, scopes)

    content: ASTNode = node.child.next if has_context else node.child
    predicate = _process_set_expression(content, scopes)

    if has_context:
        scopes.exit_scope()

    return predicate

def convert(ast: ASTNode) -> SetComprehension:
    assert ast.child is not None
    scopes = ScopeHandler()

    assert_node_type(ast, NodeType.PREDICATE)
    assert_node_type(ast.child, NodeType.PREDICATE_CONTEXT)
    assert ast.child.type == NodeType.PREDICATE_CONTEXT

    res = _process_predicate(ast, scopes)
    assert isinstance(res, SetComprehension)
    return res
