from enum import Enum
from typing import Optional, Self


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

class ValueType:
    EMPTY = "CCL_PARSE_TREE_EMPTY"
    INT = "CCL_PARSE_TREE_INT"
    ID = "CCL_PARSE_TREE_ID"

class ASTNode:
    def __init__(self, node_type, value_type, value, line, filename, child, next_node):
        self.type = node_type
        self.value_type = value_type
        self.value = value
        self.line = line
        self.filename = filename
        self.child = child
        self.next = next_node
    
    def print_tree(self, level=0):
        indent = "  " * level
        print(f"{indent}{self.type} {self.value if self.value_type != ValueType.EMPTY else ""}")
        if self.child:
            self.child.print_tree(level + 1)
        if self.next:
            self.next.print_tree(level)


    def find_child(self, node_type) -> Optional[Self]:
        """
        Find the first direct child of this node matching the given node_type.
        """
        current = self.child
        while current:
            if current.type == node_type:
                return current
            current = current.next
        return None

    def __iter__(self):
        """
        Iterate over all direct children of this node (linked by .next).
        """
        current = self.child
        while current:
            yield current
            current = current.next
