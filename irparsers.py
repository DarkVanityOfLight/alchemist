from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from types import NotImplementedType
import typing

from arm_ast import BASE_SET_TYPES, NodeType
from expressions import Argument, ComplementSet, IRNode, Identifier, IntersectionSet, LinearScale, ProductDomain, Scalar, SetComprehension, Shift, SymbolicSet, UnionSet, Vector, VectorSpace, domain_from_node_type, make_difference
from guards import SetGuard, SimpleGuard
from scope_handler import ScopeHandler

from typing import List, Optional, Tuple, TypeVar, Any, Generic

# Type variable for generic ParseResult
T = TypeVar("T", bound=IRNode)

if typing.TYPE_CHECKING:
    from arm_ast import ASTNode
    from guards import Guard


class ParseErrorType(Enum):
    """Enumeration of different types of parsing errors that can occur."""
    WRONG_NODE_TYPE = "wrong_node_type"
    WRONG_CHILD_COUNT = "wrong_child_count"
    WRONG_CHILD_TYPE = "wrong_child_type"
    INVALID_VALUE = "invalid_value"
    SCOPE_ERROR = "scope_error"
    STRUCTURAL_ERROR = "structural_error"


@dataclass
class ParseError(Exception):
    """
    Represents a parsing error with detailed context information.
    
    Provides structured error information including the type of error,
    a message, the AST node where it occurred, and expected vs actual values.
    """
    error_type: ParseErrorType
    message: str
    node: Optional[ASTNode] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None

    def __str__(self):
        """Format error as a multi-line string with all available context."""
        parts = [f"[{self.error_type.value}] {self.message}"]

        if self.node is not None:
            parts.append(f"At node: {repr(self.node)}")

        if self.expected is not None or self.actual is not None:
            parts.append(f"Expected: {self.expected}, Actual: {self.actual}")

        return "\n".join(parts)

    def merge_with(self, other: ParseError) -> ParseError:
        """
        Combine two parse errors into a single error describing multiple failures.
        
        Used when multiple parsing strategies are attempted and all fail.
        
        Args:
            other: Another ParseError to merge with this one
            
        Returns:
            A new ParseError describing both failures
        """
        merged_message = (
            f"Multiple parse attempts failed:\n"
            f"  1) {self}\n"
            f"  2) {other}"
        )
        return ParseError(
            error_type=ParseErrorType.STRUCTURAL_ERROR,
            message=merged_message,
            node=self.node or other.node,
        )


@dataclass
class ParseResult(Generic[T]):
    """
    Result wrapper for parsing operations that can succeed or fail.
    
    Implements a Result/Either monad pattern to handle success and failure cases
    without using exceptions for control flow.
    """
    success: bool
    value: Optional[T] = None
    error: Optional[ParseError] = None

    @classmethod
    def success_result(cls, value: T) -> ParseResult[T]:
        """Create a successful parse result containing a value."""
        return cls(success=True, value=value)
    
    @classmethod
    def error_result(cls, error: ParseError) -> ParseResult[Any]:
        """Create a failed parse result containing an error."""
        return cls(success=False, error=error)

    def get_error(self) -> ParseError:
        """
        Extract the error from a failed result.
        
        Returns:
            The ParseError if this is a failure
            
        Raises:
            Exception: If called on a successful result
        """
        if not self.success:
            assert self.error
            return self.error
        raise Exception("The parser did not succeed but you called get_error")

    def get_value(self) -> T:
        """
        Extract the value from a successful result.
        
        Returns:
            The parsed value if this is a success
            
        Raises:
            Exception: If called on a failed result
        """
        if self.success:
            assert self.value
            return self.value
        raise Exception("The parser did not succeed but you called get_value")


def safe_assert_node_type(node: ASTNode, expected_type: NodeType) -> Optional[ParseError]:
    """
    Safely check if a node has the expected type without raising exceptions.
    
    Args:
        node: The AST node to check
        expected_type: The expected NodeType
        
    Returns:
        ParseError if types don't match, None if they do
    """
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
    """
    Safely check if all children have types from the expected set.
    
    Args:
        node: The AST node whose children to check
        expected_types: Tuple of acceptable NodeTypes for children
        
    Returns:
        ParseError if any child has wrong type, None if all match
    """
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


def parse_vector(node: ASTNode) -> ParseResult[Vector]:
    """
    Parse a vector from a parenthesized list of integers.
    
    Handles both positive integers and negated integers.
    Example: (1, -2, 3) becomes Vector((1, -2, 3))
    
    Args:
        node: PAREN node containing INTEGER or NEG children
        
    Returns:
        ParseResult containing Vector on success or error on failure
    """
    # Check node type
    if error := safe_assert_node_type(node, NodeType.PAREN):
        return ParseResult.error_result(error)
    
    # Check all children are integers or negations
    if not all(child.type == NodeType.INTEGER or child.type == NodeType.NEG for child in node.children):
        return ParseResult.error_result(ParseError(
            ParseErrorType.WRONG_CHILD_TYPE,
            "All children must be integers for vector parsing",
            node=node
        ))
    
    try:
        ints = []
        for child in node.children:
            if child.type == NodeType.INTEGER:
                ints.append(child.value)
            elif child.type == NodeType.NEG:
                # Handle negation by multiplying by -1
                ints.append(-1 * child.children[0].value)
        return ParseResult.success_result(Vector(tuple(ints)))
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to create vector: {e}",
            node=node
        ))


def parse_argument_vector(vector_node: ASTNode) -> Tuple[str, ...]:
    """
    Extract variable names from a vector node in a set comprehension.
    
    Args:
        vector_node: VECTOR node containing IDENTIFIER children
        
    Returns:
        Tuple of variable name strings
    """
    return tuple(child.value for child in vector_node.children)


def parse_domain_expression(node: ASTNode, scopes: ScopeHandler) -> ParseResult[SetComprehension]:
    """
    Parse a domain expression which can be an identifier, product domain, or set.
    
    Handles:
    - Named identifiers referencing previously defined sets
    - Product domains like (INTEGERS, NATURALS)
    - Nested set comprehensions
    - General set expressions
    
    Args:
        node: The AST node to parse as a domain
        scopes: Scope handler for identifier resolution
        
    Returns:
        ParseResult containing the parsed domain expression
    """
    try:
        # Handle identifier references to named sets
        if node.type == NodeType.IDENTIFIER:
            ident_node = scopes.lookup_by_name(node.value)
            assert isinstance(ident_node, IRNode)
            return ParseResult.success_result(Identifier(node.value, ident_node.id, ident_node.dimension))
        
        # Handle product domains (tuples of base types)
        elif node.type == NodeType.PAREN:
            # Check if all children are base set types (INTEGERS, NATURALS, etc.)
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
        
        # Handle nested set comprehensions
        elif node.type == NodeType.SET:
            return parse_set_comprehension(node, scopes)
        
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


def parse_guard(node: ASTNode, arguments: Tuple[Argument, ...], scopes: ScopeHandler) -> ParseResult[Guard]:
    """
    Parse a guard expression for set comprehensions.
    
    Handles two types of guards:
    - SetGuard: Membership tests (x in S)
    - SimpleGuard: All other predicates and constraints
    
    Args:
        node: The AST node representing the guard condition
        arguments: Variables bound in the comprehension
        scopes: Scope handler for identifier resolution
        
    Returns:
        ParseResult containing the parsed Guard
    """
    try:
        # Check if this is an IN guard (membership test)
        if node.type == NodeType.IN:
            # IN node structure: left child is vector, right child is set expression
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
            set_expr = parse_set_expression(node.child.next, scopes)
            if set_expr.success:
                return ParseResult.success_result(SetGuard(arguments, set_expr.get_value()))
            else:
                return ParseResult.error_result(ParseError(
                    ParseErrorType.INVALID_VALUE,
                    f"Failed to parse set expression in SetGuard: {set_expr.get_error()}",
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


def parse_set_comprehension(node: ASTNode, scopes: ScopeHandler) -> ParseResult[SetComprehension]:
    """
    Parse a set comprehension expression: { (vars) in domain | guard }.
    
    Validates structure and creates appropriate Argument types based on domain.
    
    Args:
        node: SET node containing the comprehension
        scopes: Scope handler for identifier resolution
        
    Returns:
        ParseResult containing the parsed SetComprehension
    """
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
    
    # Validate IN node structure (needs variables and domain)
    if not (arguments_in_domain.child and arguments_in_domain.child.next):
        return ParseResult.error_result(ParseError(
            ParseErrorType.STRUCTURAL_ERROR,
            "IN clause must have member names and domain expression",
            node=arguments_in_domain
        ))
    
    # Check for guard clause after IN
    if not arguments_in_domain.next:
        return ParseResult.error_result(ParseError(
            ParseErrorType.STRUCTURAL_ERROR,
            "Set comprehension must have a guard clause after IN",
            node=arguments_in_domain
        ))
    
    try:
        # Extract variable names from the left side of IN
        member_names = parse_argument_vector(arguments_in_domain.child)
        
        # Parse domain expression from the right side of IN
        domain_result = parse_domain_expression(arguments_in_domain.child.next, scopes)
        if not domain_result.success:
            return ParseResult.error_result(domain_result.get_error())
        
        domain_expr = domain_result.get_value()
        
        # Create argument objects with appropriate types based on domain
        if isinstance(domain_expr, ProductDomain):
            # Product domain: each variable gets a specific base type
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
            # Nested comprehension: infer types from inner comprehension
            if len(member_names) != len(domain_expr.arguments):
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    f"Number of member names ({len(member_names)}) doesn't match nested comprehension dimension ({len(domain_expr.arguments)})",
                    node=arguments_in_domain.child
                ))
            arguments = tuple(
                Argument(name, "Inferred") for name, _ in zip(member_names, domain_expr.arguments)
            )
        else:
            # Other domain types: use inferred types
            arguments = tuple(Argument(name, "Inferred") for name in member_names)
        
        # Parse the guard expression
        guard_node = arguments_in_domain.next
        guard_result = parse_guard(guard_node, arguments, scopes)
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


def parse_scalar(node: ASTNode) -> ParseResult[Scalar]:
    """
    Parse a scalar value from an integer node.
    
    Args:
        node: INTEGER node to parse
        
    Returns:
        ParseResult containing Scalar on success
    """
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


def parse_domain(node: ASTNode) -> ParseResult[ProductDomain]:
    """
    Parse a product domain from parenthesized base set types.
    
    Example: (INTEGERS, NATURALS) becomes ProductDomain((INT, NAT))
    
    Args:
        node: PAREN node containing base set type children
        
    Returns:
        ParseResult containing ProductDomain on success
    """
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


def parse_vector_space(node: ASTNode) -> ParseResult[VectorSpace]:
    """
    Parse a vector space from basis vector specifications.
    
    Handles:
    - Single basis: INTEGERS * (1, 2)
    - Multiple bases: INTEGERS * (1, 0) + INTEGERS * (0, 1)
    
    Args:
        node: MUL or PLUS node containing vector space definition
        
    Returns:
        ParseResult containing VectorSpace on success
    """
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
    """
    Parse a vector space with a single basis vector from MUL node.
    
    Example: INTEGERS * (1, 2) creates a 1D vector space
    
    Args:
        node: MUL node with domain and vector children
        
    Returns:
        ParseResult containing VectorSpace with one basis vector
    """
    try:
        if not node.child or not node.child.next:
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                "MUL node must have exactly two children (domain and vector)",
                node=node
            ))
        
        # Children should be domain type and vector (PAREN)
        left_child = node.child
        right_child = node.child.next
        
        # Determine which child is domain and which is vector (order may vary)
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
        
        # Parse the basis vector
        vec_result = parse_vector(vector_node)
        if not vec_result.success:
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                f"Failed to parse basis vector: {vec_result.get_error().message}",
                node=vector_node
            ))
        
        basis_vec = vec_result.get_value()
        
        # Create product domain matching vector dimension
        domain = ProductDomain(tuple(domain_from_node_type(domain_node.type) for _ in range(basis_vec.dimension)))
        
        return ParseResult.success_result(VectorSpace(domain, (basis_vec,)))
        
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to create single basis vector space: {e}",
            node=node
        ))


def _parse_multi_basis_vector_space(node: ASTNode) -> ParseResult[VectorSpace]:
    """
    Parse a vector space with multiple basis vectors from PLUS node.
    
    Example: INT * (1, 0) + INT * (0, 1) creates a 2D vector space
    
    Args:
        node: PLUS node with nested MUL children
        
    Returns:
        ParseResult containing VectorSpace with multiple basis vectors
    """
    try:
        # Collect all MUL nodes from the nested PLUS structure
        mul_nodes = []
        
        def collect_mul_nodes(plus_node: ASTNode):
            """Recursively collect MUL nodes from nested PLUS structure."""
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
        
        # Parse each MUL node to extract basis vectors
        for mul in mul_nodes:
            if not mul.child or not mul.child.next:
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    "MUL node must have exactly two children (domain and vector)",
                    node=mul
                ))
            
            # Children should be domain type and vector (PAREN)
            left_child = mul.child
            right_child = mul.child.next
            
            # Determine which child is domain and which is vector
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
            
            # Parse the basis vector
            vec_result = parse_vector(vector_node)
            if not vec_result.success:
                return ParseResult.error_result(ParseError(
                    ParseErrorType.STRUCTURAL_ERROR,
                    f"Failed to parse basis vector: {vec_result.get_error().message}",
                    node=vector_node
                ))
            
            basis_vecs.append(vec_result.get_value())
            domains.append(domain_node.type)
        
        # Validate all domains are the same type
        if not all(domain == domains[0] for domain in domains[1:]):
            return ParseResult.error_result(ParseError(
                ParseErrorType.STRUCTURAL_ERROR,
                f"All domains must be the same, got {domains}",
                node=node
            ))
        
        # Validate we have at least one basis vector
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
        
        # Create the product domain
        domain = ProductDomain(tuple(domain_from_node_type(domains[0]) for _ in range(dim)))
        return ParseResult.success_result(VectorSpace(domain, tuple(basis_vecs)))
        
    except Exception as e:
        return ParseResult.error_result(ParseError(
            ParseErrorType.INVALID_VALUE,
            f"Failed to create vector space: {e}",
            node=node
        ))


class PredicateParser:
    """
    Centralized parser that tries multiple parsing strategies in order.
    
    Attempts to parse predicates using different interpreters until one succeeds.
    Collects errors from all attempts to provide detailed failure information.
    """
    
    def __init__(self):
        """Define parsing strategies in order of preference."""
        self.parsers = [
            ("vector_space", parse_vector_space),
            ("domain", parse_domain),
            ("set_comprehension", parse_set_comprehension),
            ("composition", parse_set_expression)
        ]
    
    def parse(self, node: ASTNode, scopes: ScopeHandler) -> ParseResult[SymbolicSet]:
        """
        Try parsers in order until one succeeds.
        
        Args:
            node: AST node to parse
            scopes: Scope handler for identifier resolution
            
        Returns:
            ParseResult from first successful parser, or combined error if all fail
        """
        errors = []
        
        for name, parser in self.parsers:
            try:
                # Check if parser needs scopes parameter
                if parser.__code__.co_argcount == 2:
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
        
        # All parsers failed - create comprehensive error message
        error_details = "\n".join(
            f"- {name}: {error.message}" for name, error in errors
        )

        # Find the most specific error location from all attempts
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

        # Build location description
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


def process_predicate(node: ASTNode, scopes: ScopeHandler) -> ParseResult[SymbolicSet]:
    """
    Main entry point for processing predicate nodes.
    
    Handles predicate context (local definitions) if present, then delegates
    to PredicateParser for the actual predicate content.
    
    Args:
        node: PREDICATE node to process
        scopes: Scope handler for identifier resolution
        
    Returns:
        ParseResult containing the parsed predicate
    """
    if error := safe_assert_node_type(node, NodeType.PREDICATE):
        return ParseResult.error_result(ParseError(ParseErrorType.WRONG_CHILD_TYPE, error.message, node))
    
    if not node.child:
        return ParseResult.error_result(ParseError(ParseErrorType.STRUCTURAL_ERROR, "Predicate node must have a child", node))
    
    # Check if predicate has a context (local definitions)
    has_context = node.child.type == NodeType.PREDICATE_CONTEXT
    if has_context:
        scopes.enter_scope()
        process_predicate_context(node.child, scopes)
        
        # After processing context, the actual predicate content is the next sibling
        if not node.child.next:
            return ParseResult.error_result(ParseError(ParseErrorType.STRUCTURAL_ERROR, 
                                                       "Predicate node must have content after the context", node))
        predicate_content = node.child.next
    else:
        predicate_content = node.child
    
    try:
        # Use PredicateParser to handle different types of predicates
        parser = PredicateParser()
        
        # Pass the content node directly to the parser
        result = parser.parse(predicate_content, scopes)
        return result
        
    finally:
        if has_context:
            scopes.exit_scope()


def flatten_plus_nodes(node: ASTNode) -> List[ASTNode]:
    """
    Flatten nested PLUS structure into a list of leaf nodes.
    
    Alternative helper function that recursively traverses PLUS nodes
    and returns all non-PLUS nodes found in the structure.
    
    Args:
        node: Root node to flatten
        
    Returns:
        List of all non-PLUS nodes in the tree
    """
    result = []
    
    def traverse(current: ASTNode):
        """Recursively collect non-PLUS nodes."""
        if current.type == NodeType.PLUS:
            # Traverse children of PLUS nodes
            for child in current.children:
                traverse(child)
        else:
            # Add non-PLUS nodes to result
            result.append(current)
    
    traverse(node)
    return result


def parse_scaling(node: ASTNode, scopes: ScopeHandler) -> ParseResult[LinearScale]:
    """
    Parse a scalar multiplication of a set.
    
    Handles expressions like: 2 * S or S * 2 (both orders)
    Creates a LinearScale that multiplies each element by the scalar.
    
    Args:
        node: MUL node with scalar and set children
        scopes: Scope handler for identifier resolution
        
    Returns:
        ParseResult containing LinearScale on success
    """
    if error := safe_assert_node_type(node, NodeType.MUL):
        raise AssertionError(error.message)
    
    # Try both operand orders (scalar * set and set * scalar)
    for scalar_idx, set_idx in [(0, 1), (1, 0)]:
        scalar_result = parse_scalar(node.children[scalar_idx])
        if scalar_result.success:
            base_set = parse_set_expression(node.children[set_idx], scopes)
            if base_set.success:
                return ParseResult.success_result(
                    LinearScale.from_scalar(scalar_result.get_value(), base_set.get_value()))
    
    return ParseResult.error_result(ParseError(ParseErrorType.STRUCTURAL_ERROR, "Could not be parsed as scale"))


def parse_shift(node: ASTNode, scopes: ScopeHandler) -> ParseResult[Shift]:
    """
    Parse a vector shift (translation) of a set.
    
    Handles expressions like: (1, 2) + S or S + (1, 2) (both orders)
    Creates a Shift that adds the vector to each element.
    
    Args:
        node: PLUS node with vector and set children
        scopes: Scope handler for identifier resolution
        
    Returns:
        ParseResult containing Shift on success
    """
    if error := safe_assert_node_type(node, NodeType.PLUS):
        raise ValueError(error.message)
    
    errors = []
    
    # Try both operand orders (vector + set and set + vector)
    for vector_idx, set_idx in [(0, 1), (1, 0)]:
        vector_result = parse_vector(node.children[vector_idx])
        if not vector_result.success:
            errors.append(f"Attempt with child {vector_idx} as vector failed: {vector_result.get_error().message}")
            continue

        base_set = parse_set_expression(node.children[set_idx], scopes)
        if not base_set.success:
            errors.append(f"Attempt with child {set_idx} as set failed: {base_set.get_error().message}")
            continue
            
        return ParseResult.success_result(Shift(vector_result.get_value(), base_set.get_value()))
    
    # If both orders failed, return a more informative error
    detailed_error_message = "Could not be parsed as shift. Reasons: " + "; ".join(errors)
    return ParseResult.error_result(ParseError(ParseErrorType.STRUCTURAL_ERROR, detailed_error_message))


def parse_parenthesized(node: ASTNode, scopes: ScopeHandler) -> ParseResult[SymbolicSet]:
    """
    Parse a parenthesized set expression.
    
    Unwraps parentheses and recursively parses the inner expression.
    
    Args:
        node: PAREN node containing a set expression
        scopes: Scope handler for identifier resolution
        
    Returns:
        ParseResult containing the parsed inner expression
    """
    if error := safe_assert_node_type(node, NodeType.PAREN):
        return ParseResult.error_result(ParseError(ParseErrorType.WRONG_CHILD_TYPE, error.message, node))

    if node.child is None:
        return ParseResult.error_result(ParseError(ParseErrorType.STRUCTURAL_ERROR, "Parenthesized expression should have children", node))
    
    return parse_set_expression(node.child, scopes)


def parse_set_expression(node: ASTNode, scopes: ScopeHandler) -> ParseResult[SymbolicSet]:
    """
    Main recursive parser for set expressions.
    
    Handles:
    - Base cases: identifiers, integers, vectors, parentheses
    - Set operations: union, intersection, difference, complement, XOR
    - Arithmetic operations: shift (plus), scaling (multiply)
    
    Args:
        node: AST node to parse as a set expression
        scopes: Scope handler for identifier resolution
        
    Returns:
        ParseResult containing the parsed SymbolicSet
        
    Raises:
        ValueError: If an unsupported operation is encountered
    """
    # Base Cases
    match node.type:
        case NodeType.IDENTIFIER:
            # Look up named set in scope
            ident_node = scopes.lookup_by_name(node.value)
            assert isinstance(ident_node, IRNode)
            return ParseResult.success_result(Identifier(node.value, ident_node.id, ident_node.dimension))

        case NodeType.INTEGER:
            # Parse as scalar value
            return parse_scalar(node)  # type: ignore
        
        case NodeType.PAREN:
            # Try parsing as vector first, then as parenthesized expression
            vector_result = parse_vector(node)
            if vector_result.success:
                return vector_result  # type: ignore
            
            paren_result = parse_parenthesized(node, scopes)
            if paren_result.success:
                return paren_result
            
            # Both failed - merge errors
            return ParseResult.error_result(vector_result.get_error().merge_with(paren_result.get_error()))

    # Recursive cases - parse all operands first
    operands = [parse_set_expression(child, scopes) for child in node.children]

    # Check if any operand parsing failed
    for operand in operands:
        if operand.error:
            return operand
    
    # Extract successful values
    operands = [it.get_value() for it in operands]
    
    # Set operations
    match node.type:
        case NodeType.UNION:
            return ParseResult.success_result(UnionSet(tuple(operands)))
        
        case NodeType.INTERSECTION:
            return ParseResult.success_result(IntersectionSet(tuple(operands)))
        
        case NodeType.DIFFERENCE:
            # Set difference A - B
            assert len(operands) == 2, f"Can only build a difference of two sets got {len(operands)}"
            return ParseResult.success_result(make_difference(operands[0], operands[1]))
        
        case NodeType.COMPLEMENT:
            # Set complement (NOT A)
            assert len(operands) == 1, f"Can only build the complement of one set got {len(operands)}"
            return ParseResult.success_result(ComplementSet(operands[0]))
        
        case NodeType.XOR:
            # Symmetric difference: (A union B) - (A intersect B)
            assert len(operands) == 2, f"Can only build a XOR of two sets got {len(operands)}"
            minuend = UnionSet(tuple(operands))
            subtrahend = IntersectionSet(tuple(operands))
            return ParseResult.success_result(make_difference(minuend, subtrahend))
        
        case NodeType.CARTESIAN_PRODUCT:
            # Not yet implemented
            pass

    # Arithmetic operations (shift and scale)
    match node.type:
        case NodeType.PLUS:
            # Try parsing as shift (vector translation)
            return parse_shift(node, scopes)  # type: ignore
        
        case NodeType.MINUS:
            # Not yet implemented
            pass
        
        case NodeType.MUL:
            # Try parsing as scaling (scalar multiplication)
            return parse_scaling(node, scopes)  # type: ignore
        
        case NodeType.DIV:
            # Not yet implemented
            pass
        
        case NodeType.MOD:
            # Not yet implemented
            pass
        
        case NodeType.POWER:
            # Not yet implemented
            pass

    # If we reach here, the operation is not supported
    raise ValueError(f"The operand {node.type} is not implemented as set expression")


def process_definition(node: ASTNode, scopes: ScopeHandler) -> None:
    """
    Process a definition statement that binds a name to a set expression.
    
    Example: S := { x in INTEGERS | x > 0 }
    
    Args:
        node: DEFINITION node with identifier and value children
        scopes: Scope handler to register the definition
        
    Raises:
        ValueError: If the definition structure is invalid
        ParseError: If the value expression fails to parse
    """
    if error := safe_assert_node_type(node, NodeType.DEFINITION):
        raise ValueError(error.message)
    
    if not (node.child and node.child.next):
        raise ValueError("Definition node must have identifier and value children")
    
    # Extract identifier name and parse value expression
    ident = node.child.value
    value = process_predicate(node.child.next, scopes)
    
    if value.success:
        # Register the definition in scope
        scopes.add_definition(ident, value.get_value())
    else:
        raise value.get_error()


def process_predicate_context(node: ASTNode, scopes: ScopeHandler) -> None:
    """
    Process a predicate context containing local definitions.
    
    A context is a list of definitions that are available within a predicate.
    Example:
      where
        S := INTEGERS
        T := NATURALS
      end
    
    Args:
        node: PREDICATE_CONTEXT node containing DEFINITION children
        scopes: Scope handler to register the definitions
        
    Raises:
        ValueError: If the structure is invalid or children are not definitions
    """
    if error := safe_assert_node_type(node, NodeType.PREDICATE_CONTEXT):
        raise ValueError(error.message)
    
    # Process each definition in the context
    for child in node:
        if error := safe_assert_node_type(child, NodeType.DEFINITION):
            raise ValueError(error.message)
        process_definition(child, scopes)


def convert(ast: ASTNode):
    """
    Convert an AST to intermediate representation (IR).
    
    Main entry point for the conversion process. Validates the AST structure
    and initiates parsing with a fresh scope handler.
    
    Args:
        ast: Root PREDICATE node with PREDICATE_CONTEXT child
        
    Returns:
        Tuple of (parsed IR node, scope handler)
        
    Raises:
        ValueError: If the AST structure is invalid
        ParseError: If parsing fails
    """
    if not ast.child:
        raise ValueError("AST root must have a child")
    
    if error := safe_assert_node_type(ast, NodeType.PREDICATE):
        raise ValueError(error.message)
    
    if error := safe_assert_node_type(ast.child, NodeType.PREDICATE_CONTEXT):
        raise ValueError(error.message)
    
    # Create scope handler and process the predicate
    scope_handler = ScopeHandler()
    result = process_predicate(ast, scope_handler)
    
    if result.success:
        return result.get_value(), scope_handler
    else:
        raise result.get_error()
