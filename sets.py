from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Tuple, List

from common import Variable, SetType
from arm_ast import NodeType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from guards import Guard

type SymbolicalSet = TupleDomain | FiniteSet | SetOperation | ArithmeticExpression
type Predicate = SetExpression

class SetExpression(ABC):
    @abstractmethod
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

class ArithmeticExpression(SetExpression):
    """Represents scalar multiplication of a set (e.g., 2 * G0)"""
    def __init__(self,set_expr: SetExpression):
        self.set_expr = set_expr
    
    @property
    def dim(self) -> int:
        return self.set_expr.dim
    
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        return NotImplemented
    
    def __repr__(self):
        return NotImplemented

class TupleDomain(SetExpression):
    """Represents a product type domain like (int, int, nat)"""
    def __init__(self, types: Tuple[SetType, ...]) -> None:
        self.types = types

    @property
    def dim(self) -> int:
        return len(self.types)

    def __len__(self):
        return len(self.types)

    def __getitem__(self, index: int) -> SetType:
        return self.types[index]

    def __iter__(self):
        return iter(self.types)

    def __repr__(self):
        return f"TupleDomain({', '.join(self.types)})"
    
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        if len(self.types) != len(args):
            raise ValueError("Tuple dimension mismatch")
        
        conditions = []
        for i, typ in enumerate(self.types):
            if typ == "Nat":
                conditions.append(f"(>= {args[i]} 0)")
            elif typ == "Real":
                raise NotImplementedError("Real numbers not supported")
        if conditions:
            return "(and " + " ".join(conditions) + ")"
        return "true"


class BaseSet(SetExpression):
    def __init__(self, set_type: SetType):
        self.set_type = set_type

    @property
    def dim(self) -> int:
        """Asking a base set for a dimension is not intended, since it can take any dimension"""
        return -1
    
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        """Impose the base set constraint on all args"""
        if not args:
            raise ValueError("No arguments provided for base set constraint")

        if self.set_type == "NATURALS":
            return "(and " + " ".join(f"(>= {arg} 0)" for arg in args) + ")"
        elif self.set_type == "POSITIVES":
            return "(and " + " ".join(f"(> {arg} 0)" for arg in args) + ")"
        elif self.set_type == "INTEGERS":
            return "true"
        elif self.set_type == "REALS":
            return "true"  # Placeholder
        elif self.set_type == "EMPTY":
            return "false"
        
        return "true"

    def __repr__(self):
        return f"BaseSet({self.set_type})"

class FiniteSet(SetExpression):
    def __init__(self, elements: List[Tuple[int, ...]]):
        self.elements = elements
        self._dim = len(elements[0]) if elements else 0

    @property
    def dim(self) -> int:
        return self._dim

    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        return NotImplemented
    
class SetOperation(SetExpression):
    """
    A set operation is not a real set,
    but rather represents the set we get by applying the operation
    """
    def __init__(self, op: NodeType, sets: List[SetExpression]):
        self.op = op
        self.sets = sets

    @property
    def dim(self) -> int:
        return self.sets[0].dim
        
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        set_strs = [s.realize_constraints(args) for s in self.sets]
        
        if self.op == NodeType.UNION:
            return "(or " + " ".join(set_strs) + ")"
        elif self.op == NodeType.INTERSECTION:
            return "(and " + " ".join(set_strs) + ")"
        elif self.op == NodeType.DIFFERENCE:
            if len(self.sets) != 2:
                raise ValueError("Difference requires exactly two sets")
            return f"(and {set_strs[0]} (not {set_strs[1]}))"
        elif self.op == NodeType.XOR:
            if len(self.sets) != 2:
                raise ValueError("XOR requires exactly two sets")
            return f"(xor {set_strs[0]} {set_strs[1]})"
        elif self.op == NodeType.COMPLEMENT:
            if len(self.sets) != 1:
                raise ValueError("Complement requires exactly one set")
            return f"(not {set_strs[0]})"
        elif self.op == NodeType.CARTESIAN_PRODUCT:
            # Requires specialized handling
            raise NotImplementedError("Cartesian product not implemented")
        else:
            raise NotImplementedError(f"Unsupported set operation: {self.op}")

    def __repr__(self):
        return f"SetOperation({self.op}, {len(self.sets)} sets)"

class SetComprehension(SetExpression):
    def __init__(self, members: Tuple[Variable, ...], domain: SetExpression, guard: Guard):
        self.members = members
        self.domain = domain
        self.guard = guard

    @property
    def dim(self) -> int:
        return len(self.members)

    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        domain_cond = self.domain.realize_constraints(args)
        guard_cond = self.guard.realize_constraints(args)
        return f"(and {domain_cond} {guard_cond})"

    def __repr__(self):
        return f"SetPredicate({self.members} IN {self.domain} WHERE {self.guard})"
