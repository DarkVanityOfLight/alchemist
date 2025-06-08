from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Tuple, Optional
from string import Template

from common import Variable
from arm_ast import ASTNode, NodeType
from sets import SetComprehension

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sets import SetExpression
    from expressions import Argument

class Guard(ABC):
    """
    Represent a guard.
    Either we have a "function" application to a lower set
    Or we have a Presburger guard
    """
    @abstractmethod
    def realize_constraints(self, args: Tuple[str, ...]) -> Optional[str]:
        pass


class SimpleGuard(Guard):
    def __init__(self, node: ASTNode, variables: Tuple[Argument, ...]) -> None:
        self.node = node
        self.variables = variables
        # Map each variable-name to its index so that IDENTIFIER → args[index]
        self.var_positions = {var.name: i for i, var in enumerate(variables)}

    def _lookup_var_index(self, variable_name: str) -> int:
        return self.var_positions[variable_name]
    
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        # (we still reject Real‐typed domains for now)
        if any(var.type == "Real" for var in self.variables):
            raise NotImplementedError("Real variables not supported yet")
        # Pass the raw args tuple; we’ll index into it by position
        return self._convert_node(self.node, args)

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

    def _convert_node(self, node: ASTNode, args: Tuple[str, ...]) -> str:

        # IDENTIFIER → look up its index, then pick that arg
        if node.type == NodeType.IDENTIFIER:
            idx = self._lookup_var_index(node.value)
            return args[idx]

        # INTEGER, TRUE, FALSE
        if node.type == NodeType.INTEGER:
            return str(node.value)
        if node.type == NodeType.TRUE:
            return "true"
        if node.type == NodeType.FALSE:
            return "false"

        # Unary operators
        if node.type in (NodeType.NOT, NodeType.NEG, NodeType.COMPLEMENT):
            assert node.child is not None
            child_str = self._convert_node(node.child, args)
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
            assert node.child is not None
            left = node.child
            right = node.child.next
            if left is None or right is None:
                raise Exception("Binary operator missing children")
            left_str = self._convert_node(left, args)
            right_str = self._convert_node(right, args)
            op_str = self._get_op_str(node.type)
            if node.type == NodeType.DIFFERENCE:
                # A \ B → (and A (not B))
                return f"(and {left_str} (not {right_str}))"
            return f"({op_str} {left_str} {right_str})"

        # Quantifiers
        if node.type in (NodeType.FORALL, NodeType.EXISTS):
            assert node.child is not None

            quantifier = "forall" if node.type == NodeType.FORALL else "exists"
            bindings_node = node.child
            body_node = bindings_node.next
            assert body_node is not None

            # Each binding gets a fresh SMT name, but we ignore its position in `args`,
            # since quantified variables are entirely new here.
            binding_strs = []
            new_var_map = {}  # only for the body of this quantifier
            for i, binding in enumerate(bindings_node.children):
                smt_var = f"q_{i}"
                binding_strs.append(f"({smt_var} Int)")
                new_var_map[binding.value] = smt_var

            # Inner conversion now uses mixed var‐map (outer args + new quantifier vars)
            body_str = self._convert_node_with_map(body_node, args, new_var_map)
            return f"({quantifier} ({' '.join(binding_strs)}) {body_str})"

        # Cartesian product
        if node.type == NodeType.CARTESIAN_PRODUCT:
            assert node.child is not None
            assert node.child.next is not None

            left = self._convert_node(node.child, args)
            right = self._convert_node(node.child.next, args)
            return f"(tuple-concat {left} {right})"

        # Enumerated sets
        if node.type == NodeType.ENUMERATED_SET:
            elements = []
            current = node.child
            while current:
                elements.append(self._convert_node(current, args))
                current = current.next
            return f"(set {' '.join(elements)})"

        # Function calls
        if node.type == NodeType.CALL:
            assert node.child is not None
            assert node.child.next is not None

            func = self._convert_node(node.child, args)
            arg_nodes = node.child.next.children
            arg_strs = [self._convert_node(arg, args) for arg in arg_nodes]
            return f"({func} {' '.join(arg_strs)})"

        # Vectors
        if node.type == NodeType.VECTOR:
            elems = [self._convert_node(child, args) for child in node]
            return f"(vector {' '.join(elems)})"

        raise NotImplementedError(f"Unsupported node type: {node.type}")

    def _convert_node_with_map(
        self,
        node: ASTNode,
        args: Tuple[str, ...],
        extra_map: dict[str, str]
    ) -> str:
        """
        Helper for quantifiers: we have `extra_map` for new bound vars.
        If an identifier is in `extra_map`, use that; otherwise, fall back
        to positional `args`.
        """
        if node.type == NodeType.IDENTIFIER:
            if node.value in extra_map:
                return extra_map[node.value]
            idx = self._lookup_var_index(node.value)
            return args[idx]

        if node.type == NodeType.INTEGER:
            return str(node.value)
        if node.type == NodeType.TRUE:
            return "true"
        if node.type == NodeType.FALSE:
            return "false"

        # Unary
        if node.type in (NodeType.NOT, NodeType.NEG, NodeType.COMPLEMENT):
            assert node.child is not None
            child_str = self._convert_node_with_map(node.child, args, extra_map)
            op_str = self._get_op_str(node.type)
            return f"({op_str} {child_str})"

        # Binary
        binary_ops = [
            NodeType.AND, NodeType.OR, NodeType.IMPLY, NodeType.EQUIV,
            NodeType.PLUS, NodeType.MINUS, NodeType.MUL, NodeType.DIV, NodeType.MOD,
            NodeType.EQ, NodeType.NEQ, NodeType.LT, NodeType.GT, NodeType.LEQ, NodeType.GEQ,
            NodeType.POWER, NodeType.UNION, NodeType.INTERSECTION, NodeType.DIFFERENCE, NodeType.XOR
        ]
        if node.type in binary_ops:
            assert node.child is not None
            assert node.child.next is not None
            left = node.child
            right = node.child.next
            left_str = self._convert_node_with_map(left, args, extra_map)
            right_str = self._convert_node_with_map(right, args, extra_map)
            op_str = self._get_op_str(node.type)
            if node.type == NodeType.DIFFERENCE:
                return f"(and {left_str} (not {right_str}))"
            return f"({op_str} {left_str} {right_str})"

        # Nested quantifiers (rare, but handle similarly)
        if node.type in (NodeType.FORALL, NodeType.EXISTS):
            assert node.child is not None
            assert node.child.next is not None

            quantifier = "forall" if node.type == NodeType.FORALL else "exists"
            bindings_node = node.child
            body_node = bindings_node.next
            assert body_node is not None

            binding_strs = []
            inner_map = extra_map.copy()
            for i, binding in enumerate(bindings_node.children):
                smt_var = f"q_{i}"
                binding_strs.append(f"({smt_var} Int)")
                inner_map[binding.value] = smt_var

            body_str = self._convert_node_with_map(body_node, args, inner_map)
            return f"({quantifier} ({' '.join(binding_strs)}) {body_str})"

        # Cartesian product
        if node.type == NodeType.CARTESIAN_PRODUCT:
            pass
            # assert node.child is not None
            # assert node.child.next is not None
            # left = self._convert_node_with_map(node.child, args, extra_map)
            # right = self._convert_node_with_map(node.child.next, args, extra_map)
            # return f"(tuple-concat {left} {right})"

        # Enumerated sets
        if node.type == NodeType.ENUMERATED_SET:
            pass
            # elements = []
            # current = node.child
            # while current:
            #     elements.append(self._convert_node_with_map(current, args, extra_map))
            #     current = current.next
            # return f"(set {' '.join(elements)})"

        # Function calls
        if node.type == NodeType.CALL:
            assert node.child is not None
            assert node.child.next is not None
            
            # func = self._convert_node_with_map(node.child, args, extra_map)
            # arg_nodes = node.child.next.children
            # arg_strs = [self._convert_node_with_map(arg, args, extra_map) for arg in arg_nodes]
            # return f"({func} {' '.join(arg_strs)})"

        # Vectors
        if node.type == NodeType.VECTOR:
            elems = [self._convert_node_with_map(child, args, extra_map) for child in node]
            return f"(vector {' '.join(elems)})"

        raise NotImplementedError(f"Unsupported node type: {node.type}")

    def __repr__(self):
        names = [v.name for v in self.variables]
        return f"SimpleGuard(variables={names})"


class SetGuard(Guard):
    def __init__(self, arguments: Tuple[Variable, ...], set_expr: SetExpression) -> None:
        self.arguments = arguments
        self.set_expr = set_expr
        
        # For SetComprehension: map positions to match member order
        if isinstance(set_expr, SetComprehension):
            self.arg_to_member_pos = list(range(len(arguments)))
        else:
            self.arg_to_member_pos = None

    def realize_constraints(self, args: Tuple[str, ...]) -> Optional[str]:
        if self.arg_to_member_pos:
            reordered_args = tuple(args[i] for i in self.arg_to_member_pos)
            return self.set_expr.realize_constraints(reordered_args)
        return self.set_expr.realize_constraints(args)

    def __repr__(self):
        return f"SetGuard(args={[v.name for v in self.arguments]}, expr={repr(self.set_expr)})"

class SMTGuard(Guard):
    """Internal guard class to directly generate smt guards"""


    def __init__(self, variables: Tuple[Variable, ...], template: Template) -> None:
        self.variables = variables
        # Map each variable-name to its index so that IDENTIFIER → args[index]
        self.var_positions = {var.name: i for i, var in enumerate(variables)}
        self.template = template


    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        # Build a dict { variable_name: args[index] }
        subs = {var.name: args[i] for i, var in enumerate(self.variables)}
        return self.template.substitute(subs)
    

    def __repr__(self) -> str:
        vars_str = ", ".join(var.name for var in self.variables)
        return f"SMTGuard(variables=({vars_str}), template={self.template.template})"
