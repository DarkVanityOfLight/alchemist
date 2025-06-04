from typing import Dict, List, Literal, Optional
from dataclasses import dataclass

from arm_ast import BASE_SET_TYPES, NodeType

type SetType = Literal["NATURALS", "INTEGERS", "POSITIVES", "REALS", "EMPTY"]

def node_type_to_set_type(node_type: NodeType) -> SetType:
    if node_type.value in Literal["NATURALS", "INTEGERS", "POSITIVES", "REALS", "EMPTY"].__args__:
        return node_type.value  # type: ignore
    raise ValueError(f"Unsupported NodeType: {node_type}")

@dataclass(frozen=True)
class Variable:
    name: str
    domain: Optional[SetType]

    def __repr__(self):
        return f"{self.name}: {self.domain}"

class FreshVariableOracle:

    def __init__(self, namespaces: List[str]) -> None:
        self.namespace_counter : Dict[str, int] = {namespace: 0 for namespace in namespaces}

    def fresh_variable(self, namespace: str) -> str:
        if namespace not in self.namespace_counter:
            self.namespace_counter[namespace] = -1 # Init to 0

        self.namespace_counter[namespace] += 1
        return f"{namespace}_{self.namespace_counter[namespace]}"


fresh_variable_oracle = FreshVariableOracle(
    ["scalar", "vector", "variable", "argument"]
)

def get_fresh_variable(namespace: str) -> str:
    return fresh_variable_oracle.fresh_variable(namespace)

