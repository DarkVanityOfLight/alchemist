from __future__ import annotations
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import itertools

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal, Tuple, Tuple, Any
    from guards import Guard

from arm_ast import NodeType


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

_global_id_counter = itertools.count(1)

@dataclass(frozen=True)
class IRNode():
    id: int = field(default_factory=lambda: next(_global_id_counter), init=False)

    @property
    @abstractmethod
    def children(self) -> Tuple[IRNode, ...]: raise NotImplementedError(f"Abstract method children was not implemented for {self}")

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass
    

@dataclass(frozen=True)
class ProductDomain(IRNode):
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

    @property
    def children(self) -> Tuple[IRNode, ...]:
        return ()

type Domain = BaseDomain | ProductDomain

@dataclass(frozen=True)
class SymbolicSet(IRNode):
    pass

# These value classes can be interpreted as their singelton set
@dataclass(frozen=True)
class Vector(SymbolicSet):
    comps: Tuple[int, ...]

    @property
    def children(self):
        return ()

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

    @property
    def children(self):
        return ()

type Value = Vector | Scalar

# FIXME: Assert that all the dimensions agree

@dataclass(frozen=True)
class VectorSpace(SymbolicSet):
    domain: ProductDomain
    basis: Tuple[Vector, ...]

    @property
    def children(self) -> Tuple[Vector, ...]:
        return self.basis

    @property
    def dimension(self) -> int:
        return self.domain.dimension


@dataclass(frozen=True)
class UnionSpace(SymbolicSet):
    parts: Tuple[VectorSpace]

    @property
    def children(self) -> Tuple[VectorSpace, ...]:
        return self.parts

    @property
    def dimension(self) -> int:
        return self.parts[0].dimension
    

@dataclass(frozen=True)
class FiniteSet(SymbolicSet):
    members: frozenset[SymbolicSet]

    @property
    def children(self) -> Tuple[SymbolicSet, ...]:
        return tuple(self.members)

    @property
    def dimension(self) -> int:
        return tuple(self.members)[0].dimension


@dataclass(frozen=True)
class UnionSet(SymbolicSet):
    parts: Tuple[SymbolicSet, ...]

    @property
    def children(self) -> Tuple[SymbolicSet, ...]:
        return self.parts

    @property
    def dimension(self) -> int:
        return self.parts[0].dimension

@dataclass(frozen=True)
class IntersectionSet(SymbolicSet):
    parts: Tuple[SymbolicSet, ...]

    @property
    def children(self) -> Tuple[SymbolicSet, ...]:
        return self.parts

    @property
    def dimension(self) -> int:
        return self.parts[0].dimension

@dataclass(frozen=True)
class DifferenceSet(SymbolicSet):
    minuend: SymbolicSet
    subtrahend: SymbolicSet

    @property
    def children(self) -> Tuple[SymbolicSet, ...]:
        return (self.minuend, self.subtrahend)

    @property
    def dimension(self) -> int:
        return self.minuend.dimension

@dataclass(frozen=True)
class ComplementSet(SymbolicSet):
    complemented_set: SymbolicSet

    @property
    def children(self) -> Tuple[SymbolicSet, ...]:
        return (self.complemented_set,)

    @property
    def dimension(self) -> int:
        return self.complemented_set.dimension

@dataclass(frozen=True)
class LinearScale(SymbolicSet):
    factor: Vector
    scaled_set: SymbolicSet

    @property
    def children(self) -> tuple[SymbolicSet, ...]:
        return (self.scaled_set,)

    @classmethod
    def from_scalar(cls, scalar: Scalar, scaled_set: SymbolicSet):
        return LinearScale(Vector(tuple([scalar.value for _ in range(scaled_set.dimension)])), scaled_set)

    @property
    def dimension(self) -> int:
        return self.scaled_set.dimension

@dataclass(frozen=True)
class Shift(SymbolicSet):
    shift: Vector
    shifted_set: SymbolicSet

    @property
    def children(self) -> tuple[SymbolicSet, ...]:
        return (self.shifted_set,)

    @property
    def dimension(self) -> int:
        return self.shifted_set.dimension

@dataclass(frozen=True)
class Argument():
    name: str
    type: BaseDomain | Literal["Inferred"]

@dataclass(frozen=True)
class SetComprehension(SymbolicSet):
    arguments: Tuple[Argument, ...]
    domain: SymbolicSet | ProductDomain
    guard: Guard
    
    @property
    def children(self) -> Tuple[IRNode, ...]:
        from guards import SetGuard
        if isinstance(self.guard, SetGuard):
            return (self.domain, self.guard.set_expr)
        else:
            return (self.domain,)

    def __repr__(self):
        args_repr = ", ".join(
            f"{arg.name}: {arg.type}" if arg.type != "Inferred" else arg.name
            for arg in self.arguments
        )
        return f"{{ ({args_repr}) ∈ {self.domain} | {self.guard} }}"

    @property
    def dimension(self) -> int:
        return len(self.arguments)

# This one is a bit special
# We trust that its expression is in scope
# When being called
@dataclass(frozen=True)
class Identifier(SymbolicSet):
    name: str
    ptr: int # The id this identifier should point to
    dim: int

    @property
    def children(self) -> Tuple[IRNode, ...]:
        return ()

    @property
    def dimension(self) -> int:
        return self.dim
