from __future__ import annotations
from enum import Enum, auto
from abc import ABC
from typing import Iterable, Literal, Tuple, List
from dataclasses import dataclass

from arm_ast import ASTNode, NodeType, ValueType
from common import OutOfScopeError, assert_children_types, assert_node_type
from guards import Guard
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
        return f"BaseDomain({self.types})"

    @property
    def dimension(self) -> int:
        return len(self.types)

type Domain = BaseDomain | ProductDomain

@dataclass(frozen=True)
class SymbolicSet(ABC):
    pass

# These value classes can be interpreted as their singelton set
@dataclass(frozen=True)
class Vector(SymbolicSet):
    comps: Tuple[int, ...]

    def __init__(self, comps: Iterable[int]) -> None:
        object.__setattr__(self, "comps", tuple(comps))

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
    """Adapted SetComprehension class matching the old interface"""
    arguments: Tuple[Argument, ...]
    domain: SymbolicSet | ProductDomain
    guard: Guard
    
    def __repr__(self):
        return f"SetComprehension({self.arguments} IN {self.domain} WHERE {self.guard})"

# This one is a bit special
# We trust that its expression is in scope
# When being called
@dataclass(frozen=True)
class Identifier(SymbolicSet):
    name: str

    def lookup(self, scope: ScopeHandler):
        v = scope.lookup(self.name)
        if not v:
            raise OutOfScopeError(f"The value: {self.name} is not in scope")
        return scope.lookup(self.name)

