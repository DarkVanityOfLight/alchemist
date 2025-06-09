from __future__ import annotations
from typing import Generic, List, Optional, Tuple, TypeVar, Any
from enum import Enum
from dataclasses import dataclass

from arm_ast import BASE_SET_TYPES, ASTNode, NodeType
from expressions import Argument, ComplementSet, DifferenceSet, Identifier, IntersectionSet, LinearScale, ProductDomain, Scalar, SetComprehension, Shift, SymbolicSet, UnionSet, Vector, VectorSpace, domain_from_node_type
from guards import Guard, SetGuard, SimpleGuard
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
        raise Exception("The parser did not succeed but you called get_error")

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
        return ParseResult.success_result(Vector(tuple(ints)))
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to create vector: {e}",
            node=node
        ))

def parse_argument_vector(vector_node: ASTNode) -> ParseResult[Tuple[str, ...]]:
    """Extract member names from a vector node"""
    if error := safe_assert_node_type(vector_node, NodeType.VECTOR):
        return ParseResult.error_result(error)
    
    try:
        member_names = tuple(child.value for child in vector_node.children)
        return ParseResult.success_result(member_names)
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to extract vector members: {e}",
            node=vector_node
        ))

def try_parse_domain_expression(node: ASTNode, scopes: ScopeHandler) -> ParseResult[SetComprehension]:
    """Parse domain expressions with error handling"""
    try:
        # Handle different domain types
        if node.type == NodeType.IDENTIFIER:
            value = scopes.lookup(node.value)
            if not isinstance(value, SymbolicSet) and not isinstance(value, ProductDomain):
                return ParseResult.error_result(ParseError(
                    ParseErrorType.INVALID_VALUE,
                    f"Expected SymbolicSet or ProductDomain, got {type(value)}",
                    node=node
                ))
            return ParseResult.success_result(value)
        
        elif node.type == NodeType.PAREN:
            # Handle tuple domains like (INTEGERS, NATURALS)
            if all(child.type in BASE_SET_TYPES for child in node.children):
                try:
                    types = tuple(domain_from_node_type(child.type) for child in node.children)
                    domain = ProductDomain(types)
                    return ParseResult.success_result(domain)
                except Exception as e:
                    return ParseResult.error_result(ParseError(
                        ParseErrorType.INVALID_VALUE,
                        f"Failed to create product domain: {e}",
                        node=node
                    ))
        
        elif node.type == NodeType.SET:
            # Handle nested set comprehensions
            return try_parse_set_comprehension(node, scopes)
        
        # Try to parse as general set expression for all other node types
        try:
            result = parse_set_expression(node, scopes)
            return ParseResult.success_result(result)
        except Exception as e:
            return ParseResult.error_result(ParseError(
                ParseErrorType.INVALID_VALUE,
                f"Failed to parse domain expression: {e}",
                node=node
            ))
                
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Unexpected error in domain parsing: {e}",
            node=node
        ))

def try_parse_guard(node: ASTNode, arguments: Tuple[Argument, ...], scopes: ScopeHandler) -> ParseResult[Guard]:
    """Parse a guard expression, handling both simple guards and set guards (IN expressions)"""
    try:
        # Check if this is an IN guard (SetGuard)
        if node.type == NodeType.IN:
            # Structure: IN has left child (vector) and right child (set expression)
            if not (node.child and node.child.next):
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    "IN node must have vector and set expression children",
                    node=node
                ))
            
            vec_node = node.child
            if error := safe_assert_node_type(vec_node, NodeType.VECTOR):
                return ParseResult.error_result(error)
            
            # Parse the set expression on the right side of IN
            try:
                set_expr = parse_set_expression(node.child.next, scopes)
                return ParseResult.success_result(SetGuard(arguments, set_expr))
            except Exception as e:
                return ParseResult.error_result(ParseError(
                    ParseErrorType.INVALID_VALUE,
                    f"Failed to parse set expression in SetGuard: {e}",
                    node=node.child.next
                ))
        
        # For all other guard types, create a SimpleGuard
        return ParseResult.success_result(SimpleGuard(node, arguments))
        
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to parse guard: {e}",
            node=node
        ))

def try_parse_set_comprehension(node: ASTNode, scopes: ScopeHandler) -> ParseResult[SetComprehension]:
    """Parse a set comprehension from a SET node"""
    if error := safe_assert_node_type(node, NodeType.SET):
        return ParseResult.error_result(error)
    
    # Check for IN node as first child
    if not node.child or node.child.type != NodeType.IN:
        return ParseResult.error_result(ParseError(
            ParseErrorType.STRUCTURAL_ERROR,
            "Set comprehension must start with IN clause",
            node=node
        ))
    
    arguments_in_domain = node.child
    
    # Validate IN node structure
    if not (arguments_in_domain.child and arguments_in_domain.child.next):
        return ParseResult.error_result(ParseError(
            ParseErrorType.STRUCTURAL_ERROR,
            "IN clause must have member names and domain expression",
            node=arguments_in_domain
        ))
    
    # Check for guard clause
    if not arguments_in_domain.next:
        return ParseResult.error_result(ParseError(
            ParseErrorType.STRUCTURAL_ERROR,
            "Set comprehension must have a guard clause after IN",
            node=arguments_in_domain
        ))
    
    try:
        # Extract member names from the left side of IN
        member_names_result = parse_argument_vector(arguments_in_domain.child)
        if not member_names_result.success:
            return ParseResult.error_result(member_names_result.get_error())
        
        member_names = member_names_result.get_value()
        
        # Parse domain expression from the right side of IN
        domain_result = try_parse_domain_expression(arguments_in_domain.child.next, scopes)
        if not domain_result.success:
            return ParseResult.error_result(domain_result.get_error())
        
        domain_expr = domain_result.get_value()
        
        # Create member variables based on domain type
        if isinstance(domain_expr, ProductDomain):
            if len(member_names) != len(domain_expr.types):
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    f"Number of member names ({len(member_names)}) doesn't match domain dimension ({len(domain_expr.types)})",
                    node=arguments_in_domain.child
                ))
            arguments = tuple(
                Argument(name, typ) for name, typ in zip(member_names, domain_expr.types)
            )
        elif isinstance(domain_expr, SetComprehension):
            # For nested set comprehensions, create variables with appropriate types
            if len(member_names) != len(domain_expr.arguments):
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    f"Number of member names ({len(member_names)}) doesn't match nested comprehension dimension ({len(domain_expr.arguments)})",
                    node=arguments_in_domain.child
                ))
            arguments = tuple(
                Argument(name, "Inferred") for name, var in zip(member_names, domain_expr.arguments)
            )
        else:
            # Default to integers for other domain types
            arguments = tuple(Argument(name, "Inferred") for name in member_names)
        
        # Parse guard expression using the enhanced guard parser
        guard_node = arguments_in_domain.next
        guard_result = try_parse_guard(guard_node, arguments, scopes)
        if not guard_result.success:
            return ParseResult.error_result(guard_result.get_error())
        
        guard = guard_result.get_value()
        
        return ParseResult.success_result(SetComprehension(arguments, domain_expr, guard))
        
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to create set comprehension: {e}",
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
    """Parse a domain from a PAREN node containing base set types"""
    if error := safe_assert_node_type(node, NodeType.PAREN):
        return ParseResult.error_result(error)
    
    if error := safe_assert_children_types(node, tuple(BASE_SET_TYPES)):
        return ParseResult.error_result(error)
    
    try:
        types = map(domain_from_node_type, (child.type for child in node.children))
        return ParseResult.success_result(ProductDomain(tuple(types)))
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to create domain: {e}",
            node=node
        ))

def try_parse_vector_space(node: ASTNode) -> ParseResult[VectorSpace]:
    """Parse a vector space from either a PLUS node (multiple basis vectors) or MUL node (single basis vector)"""
    
    # Handle single basis vector case (MUL node)
    if node.type == NodeType.MUL:
        return _parse_single_basis_vector_space(node)
    
    # Handle multiple basis vectors case (PLUS node)
    if node.type == NodeType.PLUS:
        return _parse_multi_basis_vector_space(node)
    
    return ParseResult.error_result(ParseError(
        ParseErrorType.WRONG_NODE_TYPE,
        f"Expected PLUS or MUL node for vector space, got {node.type}",
        node=node,
        expected=[NodeType.PLUS, NodeType.MUL],
        actual=node.type
    ))

def _parse_single_basis_vector_space(node: ASTNode) -> ParseResult[VectorSpace]:
    """Parse a vector space with a single basis vector from a MUL node"""
    try:
        if not node.child or not node.child.next:
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                "MUL node must have exactly two children (domain and vector)",
                node=node
            ))
        
        # The children should be domain (INTEGERS) and vector (PAREN)
        left_child = node.child
        right_child = node.child.next
        
        # Determine which is domain and which is vector
        if left_child.type in BASE_SET_TYPES and right_child.type == NodeType.PAREN:
            domain_node, vector_node = left_child, right_child
        elif right_child.type in BASE_SET_TYPES and left_child.type == NodeType.PAREN:
            domain_node, vector_node = right_child, left_child
        else:
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                f"MUL node children must be domain type and PAREN, got {left_child.type} and {right_child.type}",
                node=node
            ))
        
        # Parse the vector
        vec_result = try_parse_vector(vector_node)
        if not vec_result.success:
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                f"Failed to parse basis vector: {vec_result.get_error().message}",
                node=vector_node
            ))
        
        basis_vec = vec_result.get_value()
        
        # Create the domain based on vector dimension
        domain = ProductDomain(tuple(domain_from_node_type(domain_node.type) for _ in range(basis_vec.dimension)))
        
        return ParseResult.success_result(VectorSpace(domain, [basis_vec]))
        
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to create single basis vector space: {e}",
            node=node
        ))

def _parse_multi_basis_vector_space(node: ASTNode) -> ParseResult[VectorSpace]:
    """Parse a vector space with multiple basis vectors from a PLUS node with MUL children"""
    try:
        # Collect all MUL nodes from the nested PLUS structure
        mul_nodes = []
        
        def collect_mul_nodes(plus_node: ASTNode):
            """Recursively collect MUL nodes from nested PLUS structure"""
            if plus_node.type != NodeType.PLUS:
                return
            
            # Check direct children for MUL nodes
            for child in plus_node.children:
                if child.type == NodeType.MUL:
                    mul_nodes.append(child)
                elif child.type == NodeType.PLUS:
                    # Recursively process nested PLUS nodes
                    collect_mul_nodes(child)
        
        collect_mul_nodes(node)
        
        if not mul_nodes:
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                "No MUL nodes found in vector space structure",
                node=node
            ))
        
        basis_vecs = []
        domains = []
        
        for mul in mul_nodes:
            if not mul.child or not mul.child.next:
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    "MUL node must have exactly two children (domain and vector)",
                    node=mul
                ))
            
            # The children should be domain (INTEGERS) and vector (PAREN)
            left_child = mul.child
            right_child = mul.child.next
            
            # Determine which is domain and which is vector
            if left_child.type in BASE_SET_TYPES and right_child.type == NodeType.PAREN:
                domain_node, vector_node = left_child, right_child
            elif right_child.type in BASE_SET_TYPES and left_child.type == NodeType.PAREN:
                domain_node, vector_node = right_child, left_child
            else:
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    f"MUL node children must be domain type and PAREN, got {left_child.type} and {right_child.type}",
                    node=mul
                ))
            
            # Parse the vector
            vec_result = try_parse_vector(vector_node)
            if not vec_result.success:
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    f"Failed to parse basis vector: {vec_result.get_error().message}",
                    node=vector_node
                ))
            
            basis_vecs.append(vec_result.get_value())
            domains.append(domain_node.type)
        
        # Validate all domains are the same
        if not all(domain == domains[0] for domain in domains[1:]):
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                f"All domains must be the same, got {domains}",
                node=node
            ))
        
        # Validate we have basis vectors
        if not basis_vecs:
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                "No basis vectors found",
                node=node
            ))
        
        # Validate all vectors have the same dimension
        dim = basis_vecs[0].dimension
        if not all(v.dimension == dim for v in basis_vecs[1:]):
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                f"All basis vectors must have the same dimension, got dimensions {[v.dimension for v in basis_vecs]}",
                node=node
            ))
        
        # Create the domain
        domain = ProductDomain(tuple(domain_from_node_type(domains[0]) for _ in range(dim)))
        return ParseResult.success_result(VectorSpace(domain, basis_vecs))
        
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to create vector space: {e}",
            node=node
        ))

def try_parse_composition(node: ASTNode, scopes: ScopeHandler) -> ParseResult[SymbolicSet]:
    """Parse a composition by delegating to parse_set_expression"""
    try:
        result = parse_set_expression(node, scopes)
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
            ("set_comprehension", try_parse_set_comprehension),
            ("composition", try_parse_composition)
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
        
        # Pass the content node directly to the parser
        result = parser.parse(predicate_content, scopes)
        
        if not result.success:
            # node.print_tree()
            raise ValueError(f"Failed to parse predicate: {result.get_error().message}")
        
        return result.get_value()
        
    finally:
        if has_context:
            scopes.exit_scope()

def flatten_plus_nodes(node: ASTNode) -> List[ASTNode]:
    """
    Alternative helper function to flatten nested PLUS structure
    Returns all non-PLUS nodes found in the structure
    """
    result = []
    
    def traverse(current: ASTNode):
        if current.type == NodeType.PLUS:
            # Traverse children of PLUS nodes
            for child in current.children:
                traverse(child)
        else:
            # Add non-PLUS nodes to result
            result.append(current)
    
    traverse(node)
    return result

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
                raise ValueError(f"Wanted symbolic set got: {value} in expression {node}")
            return Identifier(node.value, value.id)
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

    raise ValueError(f"The operand {node.type} is not implemented as set expression")

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
