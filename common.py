from typing import Dict
from dataclasses import dataclass
from enum import Enum
from arm_ast import ASTNode, BaseSetType

@dataclass(frozen=True)
class Variable:
    name: str
    domain: BaseSetType

    def __repr__(self):
        return f"{self.name}: {self.domain}"

def _smt_sort(base_type: BaseSetType) -> str:
    match base_type:
        case BaseSetType.INTEGERS: return "Int"
        case BaseSetType.NATURALS:  return "Int"
        case BaseSetType.POSITIVES:return "Int"
        case BaseSetType.REALS:     return "Real"
        case _: raise ValueError(f"No SMT sort for {base_type}")

class Namespace(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    VARIABLE = "variable"
    BOUND_ARGUMENT = "bound_argument"
    ARGUMENT = "argument"


class FreshVariableOracle:
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
_fresh_oracle = FreshVariableOracle()


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

