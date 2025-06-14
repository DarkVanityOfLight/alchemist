from enum import Enum
from typing import Iterator, Optional, Self, List

class NodeType(Enum):
    DEFINITION = "TREE_DEFINITION"
    IDENTIFIER = "TREE_IDENTIFIER"
    INTEGER = "TREE_INTEGER"
    UNION = "TREE_UNION"
    ID_TUPLE = "ID_TUPLE" 
    PREDICATE = "PREDICATE"
    PREDICATE_CONTEXT = "PREDICATE_CONTEXT"
    PREDICATE_TUPLE = "PREDICATE_TUPLE"
    DIFFERENCE = "DIFFERENCE"
    XOR = "XOR"
    INTERSECTION = "INTERSECTION"
    PLUS = "PLUS"
    MINUS = "MINUS"
    MUL = "MUL"
    DIV = "DIV"
    MOD = "MOD"
    POWER = "POWER"
    CARTESIAN_PRODUCT = "CARTESIAN_PRODUCT"
    PAREN = "PAREN"
    COMPLEMENT = "COMPLEMENT"
    NATURALS = "NATURALS"
    INTEGERS = "INTEGERS"
    POSITIVES = "POSITIVES"
    REALS = "REALS"
    EMPTY = "EMPTY"
    LIST_ELEMENT = "LIST_ELEMENT"
    VECTOR = "VECTOR"
    OR = "OR"
    AND = "AND"
    EQUIV = "EQUIV"
    IMPLY = "IMPLY"
    IN = "IN"
    NOT = "NOT"
    CALL = "CALL"
    TRUE = "TRUE"
    FALSE = "FALSE"
    NEG = "NEG"
    ENUMERATED_SET = "ENUMERATED_SET"
    SET = "SET"
    EQ = "EQ"
    NEQ = "NEQ"
    LT = "LT"
    GT = "GT"
    GEQ = "GEQ"
    LEQ = "LEQ"
    FORALL = "FORALL"
    EXISTS = "EXISTS"

class BaseSetType(Enum):
    INTEGERS = NodeType.INTEGERS
    NATURALS = NodeType.NATURALS
    POSITIVES = NodeType.POSITIVES
    REALS = NodeType.REALS
    EMPTY = NodeType.EMPTY

class SetOperationType(Enum):
    UNION = NodeType.UNION
    INTERSECTION = NodeType.INTERSECTION
    DIFFERENCE = NodeType.DIFFERENCE
    XOR = NodeType.XOR
    COMPLEMENT = NodeType.COMPLEMENT
    CARTESIAN_PRODUCT = NodeType.CARTESIAN_PRODUCT

    
BASE_SET_TYPES = {t.value for t in BaseSetType}
SET_OPERATION_TYPES = {t.value for t in SetOperationType}

ARITHMETIC_OPERATIONS = {
    NodeType.PLUS,
    NodeType.MINUS, 
    NodeType.MUL,
    NodeType.DIV,
    NodeType.MOD,
    NodeType.POWER 
}

class ValueType(Enum):
    EMPTY = "CCL_PARSE_TREE_EMPTY"
    INT = "CCL_PARSE_TREE_INT"
    ID = "CCL_PARSE_TREE_ID"

class ASTNode:
    def __init__(self, node_type: NodeType, value_type: ValueType, value, line, filename, child: Optional[Self], next_node: Optional[Self]):
        self.type = node_type
        self.value_type = value_type
        self.value = value
        self.line = line
        self.filename = filename
        self.child : Optional[Self] = child
        self.next : Optional[Self] = next_node
    
    def print_tree(self, level:int=0):
        indent = "  " * level
        print(f"{indent}{self.type} {self.value if self.value_type != ValueType.EMPTY else ""}")
        if self.child:
            self.child.print_tree(level + 1)
        if self.next:
            self.next.print_tree(level)


    def find_child(self, node_type: NodeType) -> Optional[Self]:
        """
        Find the first direct child of this node matching the given node_type.
        """
        current = self.child
        while current:
            if current.type == node_type:
                return current
            current = current.next
        return None

    def __iter__(self) -> Iterator[Self]:
        """
        Iterate over all direct children of this node (linked by .next).
        """
        current = self.child
        while current:
            yield current
            current = current.next

    @property
    def children(self) -> List[Self]:
        return list(self)

    def __repr__(self) -> str:
        return f"ASTNode({self.type}, {self.value})"


def is_base_set(node: ASTNode) -> bool:
    return node.type in BASE_SET_TYPES

# Helper functions to create ASTNodes
def make_int_node(value: int) -> ASTNode:
    return ASTNode(NodeType.INTEGER, ValueType.INT, value, None, None, None, None)

def make_identifier_node(name: str, sibling: ASTNode | None) -> ASTNode:
    return ASTNode(NodeType.IDENTIFIER, ValueType.ID, name, None, None, None, sibling)

def make_operation_node(node_type: NodeType, operands: List[ASTNode]) -> ASTNode:
    if not operands:
        raise ValueError(f"Operation {node_type} requires at least one operand.")

    root_op_node = ASTNode(node_type, ValueType.EMPTY, None, None, None, None, None)
    
    # Link the operands as children
    root_op_node.child = operands[0]
    current_child = root_op_node.child
    for i in range(1, len(operands)):
        current_child.next = operands[i]
        current_child = current_child.next
    return root_op_node
