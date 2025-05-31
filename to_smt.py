from __future__ import annotations
from typing import Tuple, cast
from arm_ast import ASTNode, NodeType
from scope_handler import ScopeHandler

from common import Variable, SetType
from guards import Guard, SimpleGuard, SetGuard
from sets import SetExpression, BaseSet, SetOperation, TupleDomain, LinearSet, FiniteSet, SetComprehension, Predicate, SetType

def assert_node_type(node: ASTNode, expected_type):
    if node.type != expected_type:
        raise AssertionError(
            f"Expected node of type {expected_type}, got {node.type} "
            f"at line {node.line}"
        )



def _process_guard(node: ASTNode, variables: Tuple[Variable, ...], scopes: ScopeHandler) -> Guard:
    # Set guard
    if node.type == NodeType.IN:
        arguments = node.child
        assert_node_type(arguments, NodeType.VECTOR)
        set_expr = _process_set_expression(node.child.next, scopes)
        return SetGuard(variables, set_expr)

    # Formula guard
    return SimpleGuard(node, variables)

def _process_members(node: ASTNode) -> Tuple[str, ...]:
    assert_node_type(node, NodeType.VECTOR)
    return tuple(child.value for child in node)

def _process_set_expression(node: ASTNode, scopes: ScopeHandler) -> SetExpression:
    # Base sets
    if node.type in [NodeType.NATURALS, NodeType.INTEGERS, 
                    NodeType.POSITIVES, NodeType.REALS, NodeType.EMPTY]:
        return BaseSet(node.type)
    
    # Set operations
    if node.type in [NodeType.UNION, NodeType.INTERSECTION, NodeType.DIFFERENCE, 
                    NodeType.XOR, NodeType.COMPLEMENT, NodeType.CARTESIAN_PRODUCT]:
        operands = [_process_set_expression(child, scopes) for child in node]
        return SetOperation(node.type, operands)
    
    # Identifiers
    if node.type == NodeType.IDENTIFIER:
        return scopes[node.value]
    
    # Set comprehensions
    if node.type == NodeType.SET:
        return _process_set(node, scopes)
    
    # Handle PAREN - could be tuple domain or vector
    if node.type == NodeType.PAREN:
        # Check if all children are base sets (tuple domain)
        if all(child.type in [NodeType.NATURALS, NodeType.INTEGERS, 
                            NodeType.POSITIVES, NodeType.REALS] 
               for child in node):
            types = tuple("Nat" if c.type == NodeType.NATURALS else 
                         "Int" if c.type == NodeType.INTEGERS else
                         "Real" for c in node)
            return TupleDomain(cast(Tuple[SetType, ...], types))

        # Otherwise treat as vector
        vec = tuple(int(child.value) for child in node)
        return LinearSet([vec])


    # Handle multiplication: INTEGERS * vector (linear set generator)
    if node.type == NodeType.MUL:
        left = node.child
        right = node.child.next
        
        # Case 1: INTEGERS * vector (linear set generator)
        if left.type == NodeType.INTEGERS:
            # Process vector expression
            right_expr = _process_set_expression(right, scopes)
            
            if isinstance(right_expr, FiniteSet) and len(right_expr.elements) == 1:
                return LinearSet([right_expr.elements[0]])  # Extract vector from FiniteSet
            elif isinstance(right_expr, LinearSet):
                return right_expr
            else:
                raise NotImplementedError()
                # Fallback to generic operation
                return SetOperation(node.type, [BaseSet("INTEGERS"), right_expr])
        
        # Case 2: Integer constant * vector (scalar multiplication)
        elif left.type == NodeType.INTEGER:
            coefficient = int(left.value)
            right_expr = _process_set_expression(right, scopes)
            
            if isinstance(right_expr, FiniteSet) and len(right_expr.elements) == 1:
                # Scale the vector
                scaled_vec = tuple(coefficient * x for x in right_expr.elements[0])
                return FiniteSet([scaled_vec])
            elif isinstance(right_expr, LinearSet):
                new_basis = [tuple(coefficient * x for x in vec) for vec in right_expr.basis]
                return LinearSet(new_basis)
            else:
                print(right_expr)
                print(coefficient)
                raise NotImplementedError()
                # Fallback to generic operation
                return SetOperation(node.type, [
                    _process_set_expression(left, scopes),
                    _process_set_expression(right, scopes)
                ])
            
        # Generic multiplication
        return SetOperation(node.type, [
            _process_set_expression(left, scopes),
            _process_set_expression(right, scopes)
    ])
    
    # Handle sum of linear sets: term1 + term2 + ...
    if node.type == NodeType.PLUS:
        vectors = []
        current = node.child
        success = True
        
        while current:
            term = _process_set_expression(current, scopes)
            
            if isinstance(term, FiniteSet) and len(term.elements) == 1:
                vectors.append(term.elements[0])  # Extract single vector
            elif isinstance(term, LinearSet):
                vectors.extend(term.basis)
            else:
                success = False
                break
            current = current.next
        
        if success and vectors:
            return LinearSet(vectors)
            
        # Fallback to generic set operation
        operands = []
        current = node.child
        while current:
            operands.append(_process_set_expression(current, scopes))
            current = current.next
        return SetOperation(node.type, operands)
    
    # Handle vectors directly
    if node.type == NodeType.VECTOR:
        vec = tuple(int(child.value) for child in node)
        return FiniteSet([vec])  # Wrap vector in FiniteSet
    
    node.print_tree()
    raise NotImplementedError(f"Unsupported set expression type: {node.type} in line: {node.line}")

def _process_set(node: ASTNode, scopes: ScopeHandler) -> SetComprehension:
    assert_node_type(node, NodeType.SET)
    members_in_domain = node.child
    assert_node_type(members_in_domain, NodeType.IN)
    
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
    assert_node_type(node, NodeType.DEFINITION)
    
    ident = node.child.value
    value = _process_predicate(node.child.next, scopes) # FIXME: this does not have to be a predicate


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
    scopes = ScopeHandler()

    assert_node_type(ast, NodeType.PREDICATE)
    assert_node_type(ast.child, NodeType.PREDICATE_CONTEXT)
    assert ast.child.type == NodeType.PREDICATE_CONTEXT

    res = _process_predicate(ast, scopes)
    assert isinstance(res, SetComprehension)
    return res
    
