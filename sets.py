from abc import abstractmethod, ABC
from typing import Tuple, Literal, List

from common import Variable, SetType
from arm_ast import NodeType


type SymbolicalSet = BaseSet | TupleDomain | FiniteSet | LinearSet | SetOperation

class SetExpression(ABC):
    @abstractmethod
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        pass

class TupleDomain(SetExpression):
    """Represents a product type domain like (int, int, nat)"""
    def __init__(self, types: Tuple[SetType, ...]) -> None:
        self.types = types

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
    
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        if len(args) != 1:
            raise ValueError(f"Base set {self.set_type} requires exactly 1 argument")
        
        if self.set_type == "NATURALS":
            return f"(>= {args[0]} 0)"
        elif self.set_type == "POSITIVES":
            return f"(> {args[0]} 0)"
        elif self.set_type == "INTEGERS":
            return "true"
        elif self.set_type == "REALS":
            return "true"  # Placeholder for future real support
        elif self.set_type == "EMPTY":
            return "false"
        return "true"

    def __repr__(self):
        return f"BaseSet({self.set_type})"

class FiniteSet(SetExpression):
    def __init__(self, elements: List[Tuple[int, ...]]):
        self.elements = elements
        self.dim = len(elements[0]) if elements else 0

    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        if len(args) != self.dim:
            raise ValueError("Dimension mismatch in FiniteSet")
        or_conditions = []
        for element in self.elements:
            eq_conditions = [f"(= {args[i]} {val})" for i, val in enumerate(element)]
            and_cond = "(and " + " ".join(eq_conditions) + ")" if eq_conditions else "true"
            or_conditions.append(and_cond)
        return "(or " + " ".join(or_conditions) + ")" if or_conditions else "false"

class LinearSet(SetExpression):
    def __init__(self, basis: List[Tuple[int, ...]]):
        self.basis = basis
        self.dim = len(basis[0]) if basis else 0
        
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        if self.dim != len(args):
            raise ValueError("Dimension mismatch in LinearSet")
        
        eqns = []
        for i in range(self.dim):
            terms = []
            for j, vec in enumerate(self.basis):
                terms.append(f"(* a_{j} {vec[i]})")
            expr = "(+ " + " ".join(terms) + ")" if terms else "0"
            eqns.append(f"(= {args[i]} {expr})")
        
        coeff_vars = [f"a_{i}" for i in range(len(self.basis))]
        quantifier = f"(exists ({' '.join([f'({a} Int)' for a in coeff_vars])})"
        return f"{quantifier} (and {' '.join(eqns)}))"

    def __repr__(self):
        return f"LinearSet(dim={self.dim}, basis_size={len(self.basis)})"
    
class SetOperation(SetExpression):
    def __init__(self, op: NodeType, sets: List[SetExpression]):
        self.op = op
        self.sets = sets
        
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


from guards import Guard

class SetComprehension(SetExpression):
    def __init__(self, members: Tuple[Variable, ...], domain: SetExpression, guard: Guard):
        self.members = members
        self.domain = domain
        self.guard = guard

    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        domain_cond = self.domain.realize_constraints(args)
        guard_cond = self.guard.realize_constraints(args)
        return f"(and {domain_cond} {guard_cond})"

    def __repr__(self):
        return f"SetPredicate({self.members} IN {self.domain} WHERE {self.guard})"

type Predicate = SetExpression
