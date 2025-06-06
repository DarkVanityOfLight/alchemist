from typing import Dict, List, Optional
from dataclasses import dataclass

from arm_ast import BaseSetType

@dataclass(frozen=True)
class Variable:
    name: str
    domain: BaseSetType

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
    ["scalar", "vector", "variable", "bound_argument"]
)

def get_fresh_variable(namespace: str) -> str:
    return fresh_variable_oracle.fresh_variable(namespace)

class UnsupportedOperationError(Exception):
    pass
