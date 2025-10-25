from __future__ import annotations
from enum import Enum, auto
from abc import abstractmethod
from dataclasses import dataclass, field
import itertools

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal, Tuple, Tuple
    from guards import Guard

from arm_ast import NodeType


class OpType(Enum):
    """Enumeration of operation types for symbolic expressions."""
    CONST = auto()
    VAR = auto()
    ADD = auto()
    MUL = auto()
    VECTOR = auto()
    MUL_SCALAR_VECTOR = auto()


class BaseDomain(Enum):
    """
    Fundamental mathematical domains that define the type of values.
    
    INT: All integers (positive, negative, and zero)
    NAT: Natural numbers (non-negative integers)
    REAL: Real numbers
    POS: Positive natural numbers (excluding zero)
    """
    INT  = auto()
    NAT  = auto()
    REAL = auto()
    POS  = auto()

    def __str__(self):
        """Returns the mathematical notation for each domain."""
        return {
            BaseDomain.INT:  "ℤ",   # Integers
            BaseDomain.NAT:  "ℕ",   # Natural numbers
            BaseDomain.REAL: "ℝ",   # Reals
            BaseDomain.POS:  "ℕ⁺",  # Positive naturals
        }[self]


def domain_from_node_type(node_type: NodeType) -> BaseDomain:
    """
    Converts an AST node type to its corresponding base domain.
    
    Args:
        node_type: The AST node type to convert
        
    Returns:
        The corresponding BaseDomain
        
    Raises:
        ValueError: If the node type doesn't map to a known domain
    """
    match node_type:
        case NodeType.INTEGERS: return BaseDomain.INT
        case NodeType.REALS: return BaseDomain.REAL
        case NodeType.POSITIVES: return BaseDomain.POS
        case NodeType.NATURALS: return BaseDomain.NAT
        case t: raise ValueError(f"Unknown domain {t}")


def smt2_domain_from_base_domain(base: BaseDomain) -> str:
    """
    Converts a base domain to its SMT-LIB 2 type representation.
    
    SMT-LIB 2 is a standard format for SMT solvers. Natural and positive
    domains are represented as Int with additional constraints.
    
    Args:
        base: The base domain to convert
        
    Returns:
        The SMT-LIB 2 type string ("Int" or "Real")
    """
    match base:
        case BaseDomain.INT: return "Int"
        case BaseDomain.NAT: return "Int"
        case BaseDomain.REAL: return "Real"
        case BaseDomain.POS: return "Int"


# Global counter for generating unique IDs across all IR nodes
_global_id_counter = itertools.count(1)


@dataclass(frozen=True)
class IRNode():
    """
    Base class for all Intermediate Representation nodes.
    
    Each node gets a unique ID for identification and tracking.
    Frozen dataclass ensures immutability for safe use in sets/dicts.
    """
    id: int = field(default_factory=lambda: next(_global_id_counter), init=False)

    @property
    @abstractmethod
    def children(self) -> Tuple[IRNode, ...]:
        """Returns tuple of child nodes for tree traversal."""
        raise NotImplementedError(f"Abstract method children was not implemented for {self}")

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of this node (e.g., 1 for scalar, n for n-tuple)."""
        pass
    

@dataclass(frozen=True)
class ProductDomain(IRNode):
    """
    Represents a Cartesian product of base domains.
    
    For example: Z x R represents pairs (integer, real number).
    Can be indexed to get individual domain components.
    """
    types: Tuple[BaseDomain, ...]

    def __len__(self):
        """Returns the number of domains in the product."""
        return len(self.types)

    def __getitem__(self, i: int) -> BaseDomain:
        """Gets the i-th domain in the product."""
        return self.types[i]

    def __repr__(self):
        """
        String representation using mathematical product notation.
        Single domains shown without parentheses, products use x symbol.
        """
        if len(self.types) == 1:
            return str(self.types[0])
        else:
            inner = " × ".join(str(t) for t in self.types)
            return f"({inner})"

    @property
    def dimension(self) -> int:
        """Dimension equals the number of component domains."""
        return len(self.types)

    @property
    def children(self) -> Tuple[IRNode, ...]:
        """Product domains have no child nodes."""
        return ()


# Type alias: a domain can be either a base domain or a product of domains
type Domain = BaseDomain | ProductDomain


@dataclass(frozen=True)
class SymbolicSet(IRNode):
    """Base class for all symbolic set representations."""
    pass


@dataclass(frozen=True)
class Vector(SymbolicSet):
    """
    Represents a concrete vector value with integer components.
    
    Can be interpreted as a singleton set containing just this vector.
    """
    comps: Tuple[int, ...]

    @property
    def children(self):
        """Vectors are leaf nodes with no children."""
        return ()

    def __len__(self):
        """Returns the number of components in the vector."""
        return len(self.comps)

    def __getitem__(self, i: int) -> int:
        """Gets the i-th component of the vector."""
        return self.comps[i]

    def __repr__(self):
        return f"Vector({self.comps})"

    @property
    def dimension(self) -> int:
        """Dimension equals the number of vector components."""
        return len(self.comps)

 
@dataclass(frozen=True)
class Scalar(SymbolicSet):
    """
    Represents a concrete scalar (single) integer value.
    
    Can be interpreted as a singleton set containing just this value.
    """
    value: int

    def __repr__(self):
        return f"Scalar({self.value})"

    @property
    def dimension(self) -> int:
        """Scalars are always 1-dimensional."""
        return 1

    @property
    def children(self):
        """Scalars are leaf nodes with no children."""
        return ()


# Type alias for concrete values
type Value = Vector | Scalar


@dataclass(frozen=True)
class VectorSpace(SymbolicSet):
    """
    Represents a vector space defined by a domain and basis vectors.
    
    The space consists of all linear combinations of the basis vectors
    with coefficients from the specified domain.
    """
    domain: ProductDomain
    basis: Tuple[Vector, ...]

    @property
    def children(self) -> Tuple[Vector, ...]:
        """Child nodes are the basis vectors."""
        return self.basis

    @property
    def dimension(self) -> int:
        """Dimension is determined by the domain."""
        return self.domain.dimension


@dataclass(frozen=True)
class UnionSpace(SymbolicSet):
    """
    Represents the union of multiple vector spaces.
    
    Contains all points that belong to at least one of the constituent spaces.
    """
    parts: Tuple[VectorSpace]

    @property
    def children(self) -> Tuple[VectorSpace, ...]:
        """Child nodes are the component vector spaces."""
        return self.parts

    @property
    def dimension(self) -> int:
        """All parts must have same dimension; returns that dimension."""
        return self.parts[0].dimension
    

@dataclass(frozen=True)
class FiniteSet(SymbolicSet):
    """
    Represents a finite set with explicitly enumerated members.
    
    Uses frozenset to ensure uniqueness and immutability of members.
    """
    members: frozenset[SymbolicSet]

    @property
    def children(self) -> Tuple[SymbolicSet, ...]:
        """Child nodes are the set members."""
        return tuple(self.members)

    @property
    def dimension(self) -> int:
        """All members must have same dimension; returns that dimension."""
        return tuple(self.members)[0].dimension


@dataclass(frozen=True)
class UnionSet(SymbolicSet):
    """
    Represents the union of multiple sets.
    
    Contains all elements that belong to at least one of the constituent sets.
    """
    parts: Tuple[SymbolicSet, ...]

    @property
    def children(self) -> Tuple[SymbolicSet, ...]:
        """Child nodes are the component sets."""
        return self.parts

    @property
    def dimension(self) -> int:
        """All parts must have same dimension; returns that dimension."""
        return self.parts[0].dimension


@dataclass(frozen=True)
class IntersectionSet(SymbolicSet):
    """
    Represents the intersection of multiple sets.
    
    Contains only elements that belong to all constituent sets.
    """
    parts: Tuple[SymbolicSet, ...]

    @property
    def children(self) -> Tuple[SymbolicSet, ...]:
        """Child nodes are the component sets."""
        return self.parts

    @property
    def dimension(self) -> int:
        """All parts must have same dimension; returns that dimension."""
        return self.parts[0].dimension


@dataclass(frozen=True)
class ComplementSet(SymbolicSet):
    """
    Represents the complement of a set.
    
    Contains all elements (in some universe) that are NOT in the complemented set.
    """
    complemented_set: SymbolicSet

    @property
    def children(self) -> Tuple[SymbolicSet, ...]:
        """Child node is the set being complemented."""
        return (self.complemented_set,)

    @property
    def dimension(self) -> int:
        """Dimension matches the complemented set."""
        return self.complemented_set.dimension


@dataclass(frozen=True)
class LinearScale(SymbolicSet):
    """
    Represents a set scaled by a vector of factors.
    
    Each element in the original set is multiplied component-wise by the factor vector.
    For example, scaling {(1,2), (3,4)} by (2,3) gives {(2,6), (6,12)}.
    """
    factor: Vector
    scaled_set: SymbolicSet

    @property
    def children(self) -> tuple[SymbolicSet, ...]:
        """Child node is the set being scaled."""
        return (self.scaled_set,)

    @classmethod
    def from_scalar(cls, scalar: Scalar, scaled_set: SymbolicSet):
        """
        Creates a LinearScale from a scalar by replicating it to match set dimension.
        
        Converts scalar s and n-dimensional set to scaling vector (s, s, ..., s).
        """
        return LinearScale(Vector(tuple([scalar.value for _ in range(scaled_set.dimension)])), scaled_set)

    @property
    def dimension(self) -> int:
        """Dimension matches the scaled set."""
        return self.scaled_set.dimension


@dataclass(frozen=True)
class Shift(SymbolicSet):
    """
    Represents a set shifted (translated) by a vector.
    
    Each element in the original set has the shift vector added to it.
    For example, shifting {(1,2), (3,4)} by (5,6) gives {(6,8), (8,10)}.
    """
    shift: Vector
    shifted_set: SymbolicSet

    @property
    def children(self) -> tuple[SymbolicSet, ...]:
        """Child node is the set being shifted."""
        return (self.shifted_set,)

    @property
    def dimension(self) -> int:
        """Dimension matches the shifted set."""
        return self.shifted_set.dimension


@dataclass(frozen=True)
class Argument():
    """
    Represents an argument in a set comprehension.
    
    Can have either an explicit type (BaseDomain) or be inferred from context.
    """
    name: str
    type: BaseDomain | Literal["Inferred"]


@dataclass(frozen=True)
class SetComprehension(SymbolicSet):
    """
    Represents a set defined by comprehension notation: { args in domain | guard }.
    
    For example: { (x, y) in Z x Z | x > 0 and y < 10 }
    The guard is a predicate that filters which elements are included.
    """
    arguments: Tuple[Argument, ...]
    domain: SymbolicSet | ProductDomain
    guard: Guard
    
    @property
    def children(self) -> Tuple[IRNode, ...]:
        """
        Child nodes include the domain and potentially the guard's set expression.
        
        If guard is a SetGuard (membership test), includes the referenced set.
        """
        from guards import SetGuard
        if isinstance(self.guard, SetGuard):
            return (self.domain, self.guard.set_expr)
        else:
            return (self.domain,)

    def __repr__(self):
        """String representation in mathematical set comprehension notation."""
        args_repr = ", ".join(
            f"{arg.name}: {arg.type}" if arg.type != "Inferred" else arg.name
            for arg in self.arguments
        )
        return f"{{ ({args_repr}) ∈ {self.domain} | {self.guard} }}"

    @property
    def dimension(self) -> int:
        """Dimension equals the number of arguments."""
        return len(self.arguments)


def make_difference(minuend: SymbolicSet, subtrahend: SymbolicSet):
    """
    Creates a set difference (minuend - subtrahend).
    
    Implemented as intersection of minuend with complement of subtrahend.
    This is mathematically equivalent: A - B = A intersect (complement of B).
    
    Args:
        minuend: The set to subtract from
        subtrahend: The set to subtract
        
    Returns:
        IntersectionSet representing the difference
    """
    return IntersectionSet((minuend, ComplementSet(subtrahend)))


@dataclass(frozen=True)
class Identifier(SymbolicSet):
    """
    Represents a reference to a named set defined elsewhere.
    
    Acts as a pointer to another IR node by ID. The referenced set must be
    in scope when this identifier is evaluated. Used for variable references
    and named set definitions.
    """
    name: str
    ptr: int  # The ID of the IR node this identifier refers to
    dim: int  # Cached dimension to avoid lookup

    @property
    def children(self) -> Tuple[IRNode, ...]:
        """Identifiers are leaf nodes (references don't create tree edges)."""
        return ()

    @property
    def dimension(self) -> int:
        """Returns the cached dimension of the referenced set."""
        return self.dim
