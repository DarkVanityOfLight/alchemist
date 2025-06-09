from typing import Any, Dict, Union, Sequence, Iterable, Optional
from dataclasses import dataclass
from enum import Enum
from arm_ast import ASTNode, NodeType

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

def assert_node_type(node: ASTNode, expected: Union[NodeType, Sequence[NodeType]]) -> None:
    """
    Ensure a node is of the expected type(s). Raises AssertionError otherwise.
    """
    types = (expected,) if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)) else expected
    if node.type not in types:
        raise AssertionError(
            f"Expected node of type {types}, got {node.type} at line {getattr(node, 'line', '?')}"
        )


def assert_all_node_type(
    nodes: Iterable[ASTNode], expected: Union[NodeType, Sequence[NodeType]]
) -> None:
    """
    Ensure every node in an iterable is of the expected type(s).
    """
    for idx, node in enumerate(nodes):
        try:
            assert_node_type(node, expected)
        except AssertionError as e:
            raise AssertionError(f"At index {idx}: {e}")


def assert_optional_node_type(
    node: Optional[ASTNode], expected: Union[NodeType, Sequence[NodeType]]
) -> None:
    """
    If node is not None, ensure it's of the expected type(s).
    """
    if node is not None:
        assert_node_type(node, expected)


def assert_node_union_type(
    node: ASTNode, expected: Sequence[NodeType]
) -> None:
    """
    Ensure node type matches one of a union of allowed types.
    """
    assert_node_type(node, expected)


def assert_children_types(
    parent: ASTNode, expected: Union[NodeType, Sequence[NodeType]]
) -> None:
    """
    Ensure all child nodes under parent.attr are of expected type(s).
    """
    children =parent.children
    assert_all_node_type(children, expected)


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

class Namespace(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    VARIABLE = "variable"
    BOUND_ARGUMENT = "bound_argument"
    ARGUMENT = "argument"


class FreshSymbolOracle:
    """
    Manages counters for different namespaces and produces fresh names
    of the form "<namespace>_<counter>".
    """

    def __init__(self) -> None:
        # Initialize a counter for every Namespace member
        self._counter: Dict[Namespace, int] = {ns: 0 for ns in Namespace}

    def fresh(self, namespace: Namespace) -> str:
        """
        Return a fresh variable name in the given namespace.
        Example: if namespace == Namespace.SCALAR, first call returns "scalar_0",
        next returns "scalar_1", and so on.
        """
        count = self._counter[namespace]
        self._counter[namespace] = count + 1
        return f"{namespace.value}_{count}"


# Single, module‐level oracle instance
_fresh_oracle = FreshSymbolOracle()


def fresh_variable(namespace: Namespace) -> str:
    """
    Fetch a fresh variable name for the specified namespace.
    
    Usage:
        from your_module import Namespace, fresh_variable

        name1 = fresh_variable(Namespace.SCALAR)   # "scalar_0"
        name2 = fresh_variable(Namespace.SCALAR)   # "scalar_1"
        name3 = fresh_variable(Namespace.VECTOR)   # "vector_0"
    """
    return _fresh_oracle.fresh(namespace)

class UnsupportedOperationError(Exception):
    pass

class UnhandledASTNodeError(Exception):
    def __init__(self, message: str, ast_node: ASTNode) -> None:
        ast_node.print_tree()
        super().__init__(message)

class OutOfScopeError(Exception):
    pass

