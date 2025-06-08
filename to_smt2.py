from __future__ import annotations
from typing import Callable, Generic, List, Optional, TypeVar, Iterable, Union, Dict, Any, cast
from enum import Enum
from dataclasses import dataclass

from arm_ast import BASE_SET_TYPES, ASTNode, NodeType, ValueType
from common import assert_children_types, assert_node_type
from expressions import ComplementSet, DifferenceSet, IntersectionSet, LinearScale, ProductDomain, Scalar, Shift, SymbolicSet, UnionSet, Vector, VectorSpace, domain_from_node_type
from scope_handler import ScopeHandler

T = TypeVar("T")

class ParseErrorType(Enum):
    WRONG_NODE_TYPE = "wrong_node_type"
    WRONG_CHILD_COUNT = "wrong_child_count"
    WRONG_CHILD_TYPE = "wrong_child_type"
    INVALID_VALUE = "invalid_value"
    SCOPE_ERROR = "scope_error"
    STRUCTURAL_ERROR = "structural_error"

@dataclass
class ParseError:
    error_type: ParseErrorType
    message: str
    node: Optional[ASTNode] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None

@dataclass
class ParseResult(Generic[T]):
    success: bool
    value: Optional[T] = None
    error: Optional[ParseError] = None

    @classmethod
    def success_result(cls, value: T) -> ParseResult[T]:
        return cls(success=True, value=value)
    
    @classmethod
    def error_result(cls, error: ParseError) -> ParseResult[Any]:
        return cls(success=False, error=error)

    def get_error(self) -> ParseError:
        if not self.success:
            assert self.error
            return self.error
        raise Exception("The parser did succeed but you called get_error")

    def get_value(self) -> T:
        if self.success:
            assert self.value
            return self.value
        raise Exception("The parser did not succeed but you called get_value")

def safe_assert_node_type(node: ASTNode, expected_type: NodeType) -> Optional[ParseError]:
    """Returns ParseError if assertion would fail, None if it passes"""
    if node.type != expected_type:
        return ParseError(
            ParseErrorType.WRONG_NODE_TYPE,
            f"Expected {expected_type}, got {node.type}",
            node=node,
            expected=expected_type,
            actual=node.type
        )
    return None

def safe_assert_children_types(node: ASTNode, expected_types: tuple) -> Optional[ParseError]:
    """Returns ParseError if assertion would fail, None if it passes"""
    if not all(child.type in expected_types for child in node.children):
        actual_types = [child.type for child in node.children]
        return ParseError(
            ParseErrorType.WRONG_CHILD_TYPE,
            f"Expected children of types {expected_types}, got {actual_types}",
            node=node,
            expected=expected_types,
            actual=actual_types
        )
    return None

def try_parse_vector(node: ASTNode) -> ParseResult[Vector]:
    """Parse a vector from parenthesized integers"""
    # Check node type
    if error := safe_assert_node_type(node, NodeType.PAREN):
        return ParseResult.error_result(error)
    
    # Check all children are integers
    if not all(child.type == NodeType.INTEGER for child in node.children):
        return ParseResult.error_result(ParseError(
            ParseErrorType.WRONG_CHILD_TYPE,
            "All children must be integers for vector parsing",
            node=node
        ))
    
    try:
        ints = [child.value for child in node.children]
        return ParseResult.success_result(Vector(ints))
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to create vector: {e}",
            node=node
        ))

def try_parse_scalar(node: ASTNode) -> ParseResult[Scalar]:
    """Parse a scalar from an integer node"""
    if error := safe_assert_node_type(node, NodeType.INTEGER):
        return ParseResult.error_result(error)
    
    try:
        return ParseResult.success_result(Scalar(node.value))
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to create scalar: {e}",
            node=node
        ))

def try_parse_domain(node: ASTNode) -> ParseResult[ProductDomain]:
    """Parse a domain from a predicate node"""
    if error := safe_assert_node_type(node, NodeType.PREDICATE):
        return ParseResult.error_result(error)
    
    if not node.child:
        return ParseResult.error_result(ParseError(
            ParseErrorType.STRUCTURAL_ERROR,
            "Predicate node must have a child for domain parsing",
            node=node
        ))
    
    if error := safe_assert_node_type(node.child, NodeType.PAREN):
        return ParseResult.error_result(error)
    
    if error := safe_assert_children_types(node.child, tuple(BASE_SET_TYPES)):
        return ParseResult.error_result(error)
    
    try:
        types = map(domain_from_node_type, (child.type for child in node.child.children))
        return ParseResult.success_result(ProductDomain(tuple(types)))
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to create domain: {e}",
            node=node
        ))

def try_parse_vector_space(node: ASTNode) -> ParseResult[VectorSpace]:
    """Parse a vector space from a predicate node with specific structure"""
    if error := safe_assert_node_type(node, NodeType.PREDICATE):
        return ParseResult.error_result(error)
    
    if not node.child:
        return ParseResult.error_result(ParseError(
            ParseErrorType.STRUCTURAL_ERROR,
            "Predicate node must have a child for vector space parsing",
            node=node
        ))
    
    if error := safe_assert_node_type(node.child, NodeType.PLUS):
        return ParseResult.error_result(error)
    
    try:
        sums: List[ASTNode] = flatten(node.child)
        basis_vecs = []
        domains: List[NodeType] = []
        
        for s in sums:
            # Filter away PLUS node from children
            fil = tuple(filter(lambda child: child.type == NodeType.MUL, s.children))
            if len(fil) < 1:
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    "Expected MUL node in sum",
                    node=s
                ))
            
            mul = fil[0]
            if not mul.child:
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    "MUL node must have children",
                    node=mul
                ))
            
            basis, domain = (mul.child, mul.child.next) if mul.child.type == NodeType.PAREN else (mul.child.next, mul.child)
            if not basis or not domain:
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    "Could not identify basis and domain in MUL node",
                    node=mul
                ))
            
            vec_result = try_parse_vector(basis)
            if not vec_result.success:
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    f"Failed to parse basis vector: {vec_result.get_error().message}",
                    node=basis
                ))
            
            basis_vecs.append(vec_result.value)
            domains.append(domain.type)
        
        if not all(domain == domains[0] for domain in domains[1:]):
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                "All domains must be the same",
                node=node
            ))
        
        if not basis_vecs:
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                "No basis vectors found",
                node=node
            ))
        
        dim = basis_vecs[0].dimension
        if not all(v.dimension == dim for v in basis_vecs[1:]):
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                "All basis vectors must have the same dimension",
                node=node
            ))
        
        domain = ProductDomain(tuple(map(domain_from_node_type, [domains[0] for _ in range(dim)])))
        return ParseResult.success_result(VectorSpace(domain, basis_vecs))
        
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to create vector space: {e}",
            node=node
        ))

def try_parse_composition(node: ASTNode, scopes: ScopeHandler) -> ParseResult[SymbolicSet]:
    """Parse a composition by delegating to parse_set_expression"""
    if error := safe_assert_node_type(node, NodeType.PREDICATE):
        return ParseResult.error_result(error)
    
    if not node.child:
        return ParseResult.error_result(ParseError(
            ParseErrorType.STRUCTURAL_ERROR,
            "Predicate node must have a child for composition parsing",
            node=node
        ))
    
    try:
        result = parse_set_expression(node.child, scopes)
        return ParseResult.success_result(result)
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to parse composition: {e}",
            node=node
        ))

class PredicateParser:
    """Centralized predicate parser that tries parsers in order"""
    
    def __init__(self):
        # Define parsers in order of preference
        self.parsers = [
            ("vector_space", try_parse_vector_space),
            ("domain", try_parse_domain),
            ("composition", try_parse_composition),
        ]
    
    def parse(self, node: ASTNode, scopes: ScopeHandler) -> ParseResult:
        """Try parsers in order until one succeeds"""
        errors = []
        
        for name, parser in self.parsers:
            try:
                if parser.__code__.co_argcount == 2:  # Takes scopes parameter
                    result = parser(node, scopes)
                else:
                    result = parser(node)
                
                if result.success:
                    return result
                else:
                    errors.append((name, result.error))
            except Exception as e:
                # Handle any unexpected exceptions during parsing
                errors.append((name, ParseError(
                    ParseErrorType.INVALID_VALUE,
                    f"Unexpected error in {name}: {e}",
                    node=node
                )))
        
        # If we get here, all parsers failed
        error_details = "\n".join(
            f"- {name}: {error.message}" for name, error in errors
        )

        # Try to get the most specific line number from the errors
        most_specific_line = None
        most_specific_node = None
        
        for name, error in errors:
            if error.node and error.node.line is not None:
                # Prefer higher line numbers as they're likely more specific
                if most_specific_line is None or error.node.line > most_specific_line:
                    most_specific_line = error.node.line
                    most_specific_node = error.node
        
        # Fallback to the input node if no specific error location found
        if most_specific_line is None:
            most_specific_line = node.line if node.line is not None else "unknown"
            most_specific_node = node

        # Build location string
        location_parts = []
        if most_specific_line is not None:
            location_parts.append(f"line {most_specific_line}")
        if most_specific_node and most_specific_node.type is not None:
            location_parts.append(f"node type {most_specific_node.type}")
        location = ", ".join(location_parts) if location_parts else "unknown location"

        return ParseResult.error_result(ParseError(
            ParseErrorType.STRUCTURAL_ERROR,
            f"All predicate parsers failed ({location}):\n{error_details}",
            node=most_specific_node or node
        ))

def process_predicate(node: ASTNode, scopes: ScopeHandler):
    """Main predicate processing function"""
    if error := safe_assert_node_type(node, NodeType.PREDICATE):
        raise ValueError(f"Expected PREDICATE node: {error.message}")
    
    if not node.child:
        raise ValueError("Predicate node must have a child")
    
    has_context = node.child.type == NodeType.PREDICATE_CONTEXT
    if has_context:
        scopes.enter_scope()
        process_predicate_context(node.child, scopes)
        
        # After processing context, the actual predicate content is the next sibling
        if not node.child.next:
            raise ValueError("Predicate with context must have content after the context")
        predicate_content = node.child.next
    else:
        predicate_content = node.child
    
    try:
        # Use PredicateParser to handle different types of predicates
        parser = PredicateParser()
        
        # Create a temporary predicate node containing the content
        temp_predicate = ASTNode(NodeType.PREDICATE, ValueType.EMPTY, None, node.line, node.filename, None, None)
        temp_predicate.child = predicate_content
        
        result = parser.parse(temp_predicate, scopes)
        
        if not result.success:
            cast(ASTNode, result.get_error().node).print_tree()
            raise ValueError(f"Failed to parse predicate: {result.get_error().message}")
        
        return result.get_value()
        
    finally:
        if has_context:
            scopes.exit_scope()

# Keep existing helper functions
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

def parse_scaling(node: ASTNode, scopes: ScopeHandler) -> LinearScale:
    if error := safe_assert_node_type(node, NodeType.MUL):
        raise AssertionError(error.message)
    
    # Try both operand orders
    for scalar_idx, set_idx in [(0, 1), (1, 0)]:
        scalar_result = try_parse_scalar(node.children[scalar_idx])
        if scalar_result.success:
            assert scalar_result.value
            try:
                base_set = parse_set_expression(node.children[set_idx], scopes)
                return LinearScale(scalar_result.value, base_set)
            except (ValueError, AssertionError):
                continue
    
    raise AssertionError("Could not be parsed as scale")

def parse_shift(node: ASTNode, scopes: ScopeHandler) -> Shift:
    if error := safe_assert_node_type(node, NodeType.PLUS):
        raise ValueError(error.message)
    
    # Try both operand orders
    for vector_idx, set_idx in [(0, 1), (1, 0)]:
        vector_result = try_parse_vector(node.children[vector_idx])
        if vector_result.success:
            assert vector_result.value
            try:
                base_set = parse_set_expression(node.children[set_idx], scopes)
                return Shift(vector_result.value, base_set)
            except (ValueError, AssertionError):
                continue
    
    raise ValueError("Could not be parsed as shift")

def parse_set_expression(node: ASTNode, scopes: ScopeHandler) -> SymbolicSet:
    # Base Cases
    match node.type:
        case NodeType.IDENTIFIER: 
            value = scopes.lookup(node.value)
            if not isinstance(value, SymbolicSet):
                raise ValueError(f"Wanted symbolic set got: {value} in expression {node} at line {node.line}")
            return value
        case NodeType.INTEGER: 
            scalar_result = try_parse_scalar(node)
            if scalar_result.success:
                return scalar_result.get_value()
            raise ValueError(f"Failed to parse integer as scalar: {scalar_result.get_error().message}")
        case NodeType.PAREN: 
            vector_result = try_parse_vector(node)
            if vector_result.success:
                return vector_result.get_value()
            raise ValueError(f"Failed to parse parentheses as vector: {vector_result.get_error().message}")

    operands = [parse_set_expression(child, scopes) for child in node.children]
    
    match node.type:
        case NodeType.UNION: return UnionSet(tuple(operands))
        case NodeType.INTERSECTION: return IntersectionSet(tuple(operands))
        case NodeType.DIFFERENCE:
            assert len(operands) == 2, f"Can only build a difference of two sets got {len(operands)}"
            return DifferenceSet(operands[0], operands[1])
        case NodeType.COMPLEMENT: 
            assert len(operands) == 1, f"Can only build the complement of one set got {len(operands)}"
            return ComplementSet(operands[0])
        case NodeType.XOR:
            assert len(operands) == 2, f"Can only build a XOR of two sets got {len(operands)}"
            minuend = UnionSet(tuple(operands))
            subtrahend = IntersectionSet(tuple(operands))
            return DifferenceSet(minuend, subtrahend)
        case NodeType.CARTESIAN_PRODUCT: pass

    match node.type:
        case NodeType.PLUS: return parse_shift(node, scopes)
        case NodeType.MINUS: pass
        case NodeType.MUL: return parse_scaling(node, scopes)
        case NodeType.DIV: pass
        case NodeType.MOD: pass
        case NodeType.POWER: pass

    raise ValueError(f"The operand {node.type} is not implemented as set expression in line {node.line}")

# Keep existing functions for context processing
def process_definition(node: ASTNode, scopes: ScopeHandler):
    if error := safe_assert_node_type(node, NodeType.DEFINITION):
        raise ValueError(error.message)
    
    if not (node.child and node.child.next):
        raise ValueError("Definition node must have identifier and value children")
    
    ident = node.child.value
    value = process_predicate(node.child.next, scopes)
    scopes.add_definition(ident, value)

def process_predicate_context(node: ASTNode, scopes: ScopeHandler):
    if error := safe_assert_node_type(node, NodeType.PREDICATE_CONTEXT):
        raise ValueError(error.message)
    
    for child in node:
        if error := safe_assert_node_type(child, NodeType.DEFINITION):
            raise ValueError(error.message)
        process_definition(child, scopes)

def convert(ast: ASTNode):
    if not ast.child:
        raise ValueError("AST root must have a child")
    
    if error := safe_assert_node_type(ast, NodeType.PREDICATE):
        raise ValueError(error.message)
    
    if error := safe_assert_node_type(ast.child, NodeType.PREDICATE_CONTEXT):
        raise ValueError(error.message)
    
    result = process_predicate(ast, ScopeHandler())
    return result
