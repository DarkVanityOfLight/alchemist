from __future__ import annotations
from enum import Enum, auto
from abc import ABC
from dataclasses import dataclass, field
import itertools

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Tuple, List
    from guards import Guard

from arm_ast import NodeType
from common import OutOfScopeError
from scope_handler import ScopeHandler


class OpType(Enum):
    CONST = auto()
    VAR = auto()
    ADD = auto()
    MUL = auto()
    VECTOR = auto()
    MUL_SCALAR_VECTOR = auto()

class BaseDomain(Enum):
    INT  = auto()
    NAT  = auto()
    REAL = auto()
    POS  = auto()

    def __str__(self):
        return {
            BaseDomain.INT:  "ℤ",   # Integers
            BaseDomain.NAT:  "ℕ",   # Natural numbers
            BaseDomain.REAL: "ℝ",   # Reals
            BaseDomain.POS:  "ℕ⁺",  # Positive naturals
        }[self]

def domain_from_node_type(node_type: NodeType) -> BaseDomain:
    match node_type:
        case NodeType.INTEGERS: return BaseDomain.INT
        case NodeType.REALS: return BaseDomain.REAL
        case NodeType.POSITIVES: return BaseDomain.POS
        case NodeType.NATURALS: return BaseDomain.NAT
        case t: raise ValueError(f"Unknown domain {t}")

@dataclass(frozen=True)
class ProductDomain():
    types: Tuple[BaseDomain, ...]

    def __len__(self):
        return len(self.types)

    def __getitem__(self, i: int) -> BaseDomain:
        return self.types[i]

    def __repr__(self):
        if len(self.types) == 1:
            return str(self.types[0])
        else:
            inner = " × ".join(str(t) for t in self.types)
            return f"({inner})"

    @property
    def dimension(self) -> int:
        return len(self.types)

type Domain = BaseDomain | ProductDomain

_global_id_counter = itertools.count(1)

@dataclass(frozen=True)
class SymbolicSet(ABC):
    id: int = field(default_factory=lambda: next(_global_id_counter), init=False)

# These value classes can be interpreted as their singelton set
@dataclass(frozen=True)
class Vector(SymbolicSet):
    comps: Tuple[int, ...]

    def __len__(self):
        return len(self.comps)

    def __getitem__(self, i: int) -> int:
        return self.comps[i]

    def __repr__(self):
        return f"Vector({self.comps})"

    @property
    def dimension(self) -> int:
        return len(self.comps)

 
@dataclass(frozen=True)
class Scalar(SymbolicSet):
    value: int

    def __repr__(self):
        return f"Scalar({self.value})"

    @property
    def dimension(self) -> int:
        return 1

type Value = Vector | Scalar

@dataclass(frozen=True)
class VectorSpace(SymbolicSet):
    domain: ProductDomain
    basis: List[Vector]


@dataclass(frozen=True)
class UnionSpace:
    parts: List[VectorSpace]
    

@dataclass(frozen=True)
class FiniteSet(SymbolicSet):
    members: frozenset


@dataclass(frozen=True)
class UnionSet(SymbolicSet):
    parts: Tuple[SymbolicSet, ...]

@dataclass(frozen=True)
class IntersectionSet(SymbolicSet):
    parts: Tuple[SymbolicSet, ...]

@dataclass(frozen=True)
class DifferenceSet(SymbolicSet):
    minuend: SymbolicSet
    subtrahend: SymbolicSet

@dataclass(frozen=True)
class ComplementSet(SymbolicSet):
    complemented_set: SymbolicSet

@dataclass(frozen=True)
class LinearScale(SymbolicSet):
    factor: Scalar
    scaled_set: SymbolicSet

@dataclass(frozen=True)
class Shift(SymbolicSet):
    shift: Vector
    shifted_set: SymbolicSet

@dataclass(frozen=True)
class Argument():
    name: str
    type: BaseDomain | Literal["Inferred"]

@dataclass(frozen=True)
class SetComprehension(SymbolicSet):
    arguments: Tuple[Argument, ...]
    domain: SymbolicSet | ProductDomain
    guard: Guard
    
    def __repr__(self):
        args_repr = ", ".join(
            f"{arg.name}: {arg.type}" if arg.type != "Inferred" else arg.name
            for arg in self.arguments
        )
        return f"{{ ({args_repr}) ∈ {self.domain} | {self.guard} }}"

# This one is a bit special
# We trust that its expression is in scope
# When being called
@dataclass(frozen=True)
class Identifier(SymbolicSet):
    name: str
    target_id: int

    def lookup(self, scope: ScopeHandler):
        v = scope.lookup(self.name)
        if not v:
            raise OutOfScopeError(f"The value: {self.name} is not in scope")
        return scope.lookup(self.name)

