from __future__ import annotations
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from string import Template

from arm_ast import ASTNode, NodeType

from typing import TYPE_CHECKING

from expressions import IRNode, SetComprehension
if TYPE_CHECKING:
    from expressions import SymbolicSet
    from expressions import Argument


@dataclass(frozen=True)
class Guard(IRNode):
    """
    Represent a guard.
    Either we have a "function" application to a lower set
    Or we have a Presburger guard
    """
    @abstractmethod
    def realize_constraints(self, args: Tuple[str, ...]) -> Optional[str]:
        pass


@dataclass(frozen=True)
class SimpleGuard(Guard):
    node: ASTNode
    variables: Tuple[Argument, ...]
    var_names_by_pos: Dict[int, str] = None #type: ignore

    def __post_init__(self):
        if self.var_names_by_pos is None:
            object.__setattr__(self, 'var_names_by_pos',
                               {i: var.name for i, var in enumerate(self.variables)})

    @property
    def children(self) -> Tuple[IRNode, ...]:
        return ()


    def __repr__(self):
        names = [v.name for v in self.variables]
        return f"SimpleGuard(variables={names}, guardNode={self.node})"


@dataclass(frozen=True)
class SetGuard(Guard):
    arguments: Tuple[Argument, ...]
    set_expr: SymbolicSet
    arg_to_member_pos: Optional[List[int]] = None
    
    def __post_init__(self):
        # For SetComprehension: map positions to match member order
        if self.arg_to_member_pos is None:
            if isinstance(self.set_expr, SetComprehension):
                object.__setattr__(self, 'arg_to_member_pos', list(range(len(self.arguments))))
            else:
                object.__setattr__(self, 'arg_to_member_pos', None)

    def realize_constraints(self, args: Tuple[str, ...]) -> Optional[str]:
        raise NotImplementedError()

    @property
    def children(self) -> Tuple[IRNode, ...]:
        return (self.set_expr,)
    
    def __repr__(self):
        return f"SetGuard(args={[v.name for v in self.arguments]}, expr={repr(self.set_expr)})"


@dataclass(frozen=True)
class SMTGuard(Guard):
    """Internal guard class to directly generate smt guards"""
    variables: Tuple[Argument, ...]
    template: Template
    var_positions: Dict[str, int] = None #type: ignore
    
    def __post_init__(self):
        # Map each variable-name to its index so that IDENTIFIER -> args[index]
        if self.var_positions is None:
            object.__setattr__(self, 'var_positions', 
                             {var.name: i for i, var in enumerate(self.variables)})

    @property
    def children(self) -> Tuple[IRNode, ...]:
        return ()
    
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        # Build a dict { variable_name: args[index] }
        subs = {var.name: args[i] for i, var in enumerate(self.variables)}
        return self.template.substitute(subs)
    
    def __repr__(self) -> str:
        vars_str = ", ".join(var.name for var in self.variables)
        return f"SMTGuard(variables=({vars_str}), template={self.template.template})"
