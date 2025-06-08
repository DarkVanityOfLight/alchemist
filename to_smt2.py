from __future__ import annotations
from typing import Callable, List, Optional, TypeVar, Iterable

from arm_ast import BASE_SET_TYPES, ASTNode, NodeType
from common import assert_children_types, assert_node_type
from expressions import LinearScale, ProductDomain, Scalar, Shift, SymbolicSet, UnionSet, Vector, VectorSpace, domain_from_node_type
from scope_handler import ScopeHandler
import inspect

R = TypeVar("R")

def try_parse(node: ASTNode, scopes: ScopeHandler, func: Callable[..., R]) -> Optional[R]:
    try:
        sig = inspect.signature(func)
        params = sig.parameters
        if len(params) == 1:
            return func(node)
        elif len(params) == 2:
            return func(node, scopes)
        else:
            raise TypeError(f"Parser {func.__name__} must take 1 or 2 arguments, got {len(params)}")
    except AssertionError as e:
        print(f"Assertion failed in {func.__name__}: {e}")
        return None

def try_parsers(node: ASTNode, scopes: ScopeHandler, parsers: Iterable[Callable[..., R]]) -> Optional[R]:
    for parser in parsers:
        result = try_parse(node, scopes, parser)
        if result is not None:
            return result
    return None

def flatten(node: ASTNode) -> List[ASTNode]:
    typ = node.type
    current = node
    res = []
    while current is not None and current.type == typ:
        res.append(current)
        if current.child is None:
            break
        current = current.child
    return res


def as_vector(node: ASTNode) -> Vector:
    assert_node_type(node, NodeType.PAREN)
    assert all(child.type == NodeType.INTEGER for child in node.children)
    ints = [child.value for child in node]
    return Vector(ints)


def as_vector_space(node: ASTNode) -> VectorSpace:
    assert_node_type(node, NodeType.PREDICATE)
    assert node.child is not None
    assert_node_type(node.child, NodeType.PLUS)

    sums: List[ASTNode] = flatten(node.child)
    basis_vecs = []
    domains: List[NodeType] = []
    for s in sums:
        # Filter away PLUS node from children
        fil = tuple(filter(lambda child: child.type == NodeType.MUL, s.children))
        assert len(fil) >= 1
        mul = fil[0]
        assert mul.child
        basis, domain = (mul.child, mul.child.next) if mul.child.type == NodeType.PAREN else (mul.child.next, mul.child)
        assert basis and domain
        vec = as_vector(basis)
        basis_vecs.append(vec)
        domains.append(domain.type)

    assert all(domain == domains[0] for domain in domains[1:]), "Domains are not all the same"

    dim = basis_vecs[0].dimension
    assert all(v.dimension == dim for v in basis_vecs[1:]), "Basis vectors must have the same dimension"

    domain = ProductDomain(tuple(map(domain_from_node_type, [domains[0] for _ in range(dim)])))
    return VectorSpace(domain, basis_vecs)

def as_domain(node: ASTNode):
    assert_node_type(node, NodeType.PREDICATE)
    assert node.child
    assert_node_type(node.child, NodeType.PAREN)
    assert_children_types(node.child, tuple(BASE_SET_TYPES))

    types = map(domain_from_node_type, (child.type for child in node.child.children))
    return ProductDomain(tuple(types))

def as_scalar(node: ASTNode) -> Scalar:
    assert_node_type(node, NodeType.INTEGER)
    return Scalar(node.value)

def as_composition(node: ASTNode, scopes: ScopeHandler):
    assert_node_type(node, NodeType.PREDICATE)
    assert node.child, "No child"
    return parse_set_expression(node.child, scopes)


def parse_scaling(node: ASTNode, scopes: ScopeHandler) -> LinearScale:
    assert_node_type(node, NodeType.MUL)

    # Try both operand orders
    for scalar_idx, set_idx in [(0, 1), (1, 0)]:
        try:
            scalar = as_scalar(node.children[scalar_idx])
            base_set = parse_set_expression(node.children[set_idx], scopes)
            return LinearScale(scalar, base_set)
        except (ValueError, AssertionError):
            continue

    assert False, "Could not be parsed as scale"

def parse_shift(node: ASTNode, scopes: ScopeHandler) -> Shift:
    assert_node_type(node, NodeType.PLUS)

    # Try both operand orders
    for scalar_idx, set_idx in [(0, 1), (1, 0)]:
        try:
            vector = as_vector(node.children[scalar_idx])
            base_set = parse_set_expression(node.children[set_idx], scopes)
            return Shift(vector, base_set)
        except (ValueError, AssertionError):
            continue

    assert False, "Could not be parsed as scale"

def parse_set_expression(node: ASTNode, scopes: ScopeHandler) -> SymbolicSet:
    # Base Cases
    match node.type:
        case NodeType.IDENTIFIER: 
            value = scopes.lookup(node.value)
            if not isinstance(value, SymbolicSet):
                raise ValueError(f"Wanted symbolic set got: {value} in expression {node} at line {node.line}")
            return value
        case NodeType.INTEGER: return Scalar(node.value)
        case NodeType.PAREN: return as_vector(node)


    operands : map[SymbolicSet]= map(lambda child: parse_set_expression(child, scopes), node.children)
    assert all(isinstance(op, SymbolicSet) for op in operands), "All operands must be of type SymbolicSet"

    match node.type:
        case NodeType.UNION: return UnionSet(tuple(operands))
        case NodeType.INTERSECTION: pass
        case NodeType.DIFFERENCE: pass
        case NodeType.XOR: pass
        case NodeType.COMPLEMENT: pass
        case NodeType.CARTESIAN_PRODUCT: pass

    match node.type:
        case NodeType.PLUS: return parse_shift(node, scopes)
        case NodeType.MINUS: pass
        case NodeType.MUL: return parse_scaling(node, scopes)
        case NodeType.DIV: pass
        case NodeType.MOD: pass
        case NodeType.POWER: pass

    assert False, f"The operand {node.type} is not implemented as composition"



def process_definition(node: ASTNode, scopes: ScopeHandler):
    assert_node_type(node, NodeType.DEFINITION)
    assert node.child and node.child.next

    ident = node.child.value
    value = process_predicate(node.child.next, scopes)
    scopes.add_definition(ident, value)

def process_predicate_context(node: ASTNode, scopes: ScopeHandler):
    assert_node_type(node, NodeType.PREDICATE_CONTEXT)
    for child in node:
        assert_node_type(child, NodeType.DEFINITION)
        process_definition(child, scopes)

def process_predicate(node: ASTNode, scopes: ScopeHandler):
    assert_node_type(node, NodeType.PREDICATE)
    assert node.child

    has_context = node.child.type == NodeType.PREDICATE_CONTEXT
    if has_context:
        scopes.enter_scope()
        process_predicate_context(node.child, scopes)

    parsers = [as_vector_space, as_domain, as_composition]

    predicate = try_parsers(node, scopes, parsers)


    if has_context:
        scopes.exit_scope()

    if not predicate:
        node.print_tree()
        raise ValueError(f"Unkown predicate")

    return predicate


def convert(ast: ASTNode):
    assert ast.child

    assert_node_type(ast, NodeType.PREDICATE)
    assert_node_type(ast.child, NodeType.PREDICATE_CONTEXT)

    result = process_predicate(ast, ScopeHandler())
    return result
