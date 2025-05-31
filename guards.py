from abc import abstractmethod, ABC
from typing import Tuple


from common import Variable
from arm_ast import ASTNode, NodeType

class Guard(ABC):
    """
    Represent a guard.
    Either we have a "function" application to a lower set
    Or we have a presburger guard
    """
    @abstractmethod
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        pass

from sets import SetComprehension, SetExpression

class SimpleGuard(Guard):
    def __init__(self, node: ASTNode, variables: Tuple[Variable, ...]) -> None:
        self.node = node
        self.variables = variables
        self.var_positions = {var.name: i for i, var in enumerate(variables)}

    def _lookup_var(self, variable: str) -> int:
        return self.var_positions[variable]
    
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        if any(var.domain == "Real" for var in self.variables):
            raise NotImplementedError("Real variables not supported yet")
        
        var_map = {var.name: args[i] for i, var in enumerate(self.variables)}
        return self._convert_node(self.node, var_map)

    def _get_op_str(self, node_typ: NodeType) -> str:
        op_map = {
            NodeType.AND: "and",
            NodeType.OR: "or",
            NodeType.IMPLY: "=>",
            NodeType.EQUIV: "=",
            NodeType.PLUS: "+",
            NodeType.MINUS: "-",
            NodeType.MUL: "*",
            NodeType.DIV: "div",
            NodeType.MOD: "mod",
            NodeType.POWER: "pow",
            NodeType.EQ: "=",
            NodeType.NEQ: "distinct",
            NodeType.LT: "<",
            NodeType.GT: ">",
            NodeType.LEQ: "<=",
            NodeType.GEQ: ">=",
            NodeType.NOT: "not",
            NodeType.NEG: "-",
            NodeType.UNION: "or",
            NodeType.INTERSECTION: "and",
            NodeType.DIFFERENCE: "and",  # A \ B = A and not B
            NodeType.XOR: "xor",
            NodeType.COMPLEMENT: "not",
        }
        return op_map.get(node_typ, f"<UNKNOWN_OP:{node_typ}>")

    def _convert_node(self, node: ASTNode, var_map: dict[str, str]) -> str:
        # Handle identifiers
        if node.type == NodeType.IDENTIFIER:
            return var_map[node.value]
        
        # Handle constants
        if node.type == NodeType.INTEGER:
            return str(node.value)
        if node.type == NodeType.TRUE:
            return "true"
        if node.type == NodeType.FALSE:
            return "false"
        
        # Unary operators
        if node.type in (NodeType.NOT, NodeType.NEG, NodeType.COMPLEMENT):
            child_str = self._convert_node(node.child, var_map)
            op_str = self._get_op_str(node.type)
            return f"({op_str} {child_str})"
        
        # Binary operators
        binary_ops = [
            NodeType.AND, NodeType.OR, NodeType.IMPLY, NodeType.EQUIV,
            NodeType.PLUS, NodeType.MINUS, NodeType.MUL, NodeType.DIV, NodeType.MOD,
            NodeType.EQ, NodeType.NEQ, NodeType.LT, NodeType.GT, NodeType.LEQ, NodeType.GEQ,
            NodeType.POWER, NodeType.UNION, NodeType.INTERSECTION, NodeType.DIFFERENCE, NodeType.XOR
        ]
        if node.type in binary_ops:
            left = node.child
            right = node.child.next
            if left is None or right is None:
                raise Exception("Binary operator missing children")
            left_str = self._convert_node(left, var_map)
            right_str = self._convert_node(right, var_map)
            op_str = self._get_op_str(node.type)
            
            # Special handling for set operations
            if node.type == NodeType.DIFFERENCE:
                return f"(and {left_str} (not {right_str}))"
            return f"({op_str} {left_str} {right_str})"
        
        # Quantifiers
        if node.type in (NodeType.FORALL, NodeType.EXISTS):
            quantifier = "forall" if node.type == NodeType.FORALL else "exists"
            bindings_node = node.child
            body_node = bindings_node.next
            
            binding_strs = []
            new_var_map = var_map.copy()
            for i, binding in enumerate(bindings_node.children):
                var_name = binding.value
                smt_var = f"q_{i}"
                binding_strs.append(f"({smt_var} Int)")
                new_var_map[var_name] = smt_var
                
            body_str = self._convert_node(body_node, new_var_map)
            return f"({quantifier} ({' '.join(binding_strs)}) {body_str})"
        
        # Cartesian product (special handling)
        if node.type == NodeType.CARTESIAN_PRODUCT:
            left = self._convert_node(node.child, var_map)
            right = self._convert_node(node.child.next, var_map)
            return f"(tuple-concat {left} {right})"
        
        # Enumerated sets
        if node.type == NodeType.ENUMERATED_SET:
            elements = []
            current = node.child
            while current:
                elements.append(self._convert_node(current, var_map))
                current = current.next
            return f"(set {' '.join(elements)})"
        
        # Function calls
        if node.type == NodeType.CALL:
            func = self._convert_node(node.child, var_map)
            args = [self._convert_node(arg, var_map) for arg in node.child.next.children]
            return f"({func} {' '.join(args)})"
        
        # Vectors
        if node.type == NodeType.VECTOR:
            elements = [self._convert_node(child, var_map) for child in node]
            return f"(vector {' '.join(elements)})"
        
        raise NotImplementedError(f"Unsupported node type: {node.type}")

    def __repr__(self):
        return f"SimpleGuard(variables={[v.name for v in self.variables]})"


class SetGuard(Guard):
    def __init__(self, arguments: Tuple[Variable, ...], set_expr: SetExpression) -> None:
        self.arguments = arguments
        self.set_expr = set_expr
        
        # For SetComprehension: map positions to match member order
        if isinstance(set_expr, SetComprehension):
            self.arg_to_member_pos = list(range(len(arguments)))
        else:
            self.arg_to_member_pos = None

    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        if self.arg_to_member_pos:
            # Reorder args to match set predicate's member order
            reordered_args = tuple(args[i] for i in self.arg_to_member_pos)
            return self.set_expr.realize_constraints(reordered_args)
        return self.set_expr.realize_constraints(args)

    def __repr__(self):
        return f"SetGuard(args={[v.name for v in self.arguments]}, expr={repr(self.set_expr)})"
