from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from string import Template
from arm_ast import ASTNode
from typing import TYPE_CHECKING
from expressions import IRNode, SetComprehension

if TYPE_CHECKING:
    from expressions import SymbolicSet
    from expressions import Argument


@dataclass(frozen=True)
class Guard(IRNode):
    """
    Base class for guard expressions that filter elements in set comprehensions.
    
    Guards act as predicates or constraints that determine which elements
    from a domain are included in a set. Can represent either:
    - Function applications to lower sets (membership tests)
    - Presburger arithmetic constraints (linear integer arithmetic)
    """
    
    @abstractmethod
    def realize_constraints(self, args: Tuple[str, ...]) -> Optional[str]:
        """
        Converts the guard into concrete constraints using provided argument names.
        
        Args:
            args: Tuple of variable names to use in the constraint expression
            
        Returns:
            String representation of the constraint, or None if not applicable
        """
        pass


@dataclass(frozen=True)
class SimpleGuard(Guard):
    """
    Guard based on an AST node expression.
    
    Represents guards parsed directly from the source language AST.
    Stores variable bindings to map between variable positions and names.
    """
    node: ASTNode  # The AST node representing the guard condition
    variables: Tuple[Argument, ...]  # Variables used in this guard
    var_names_by_pos: Dict[int, str] = None  # type: ignore  # Position to name mapping
    
    def __post_init__(self):
        """Initialize the position-to-name mapping if not provided."""
        if self.var_names_by_pos is None:
            object.__setattr__(self, 'var_names_by_pos',
                               {i: var.name for i, var in enumerate(self.variables)})
    
    @property
    def children(self) -> Tuple[IRNode, ...]:
        """SimpleGuards have no IR children (AST node is not part of IR tree)."""
        return ()
    
    def __repr__(self):
        """String representation showing variables and guard AST node."""
        names = [v.name for v in self.variables]
        return f"SimpleGuard(variables={names}, guardNode={self.node})"


@dataclass(frozen=True)
class SetGuard(Guard):
    """
    Guard that tests membership in another set.
    
    Represents constraints of the form "x in S" where S is a symbolic set.
    Maps argument positions to member positions for complex set expressions.
    """
    arguments: Tuple[Argument, ...]  # Variables being tested for membership
    set_expr: SymbolicSet  # The set to test membership against
    arg_to_member_pos: Optional[List[int]] = None  # Mapping from args to set member positions
    
    def __post_init__(self):
        """
        Initialize position mapping for set comprehensions.
        
        For SetComprehension expressions, creates identity mapping (each arg maps to
        corresponding member position). For other set types, no mapping is needed.
        """
        if self.arg_to_member_pos is None:
            if isinstance(self.set_expr, SetComprehension):
                # Identity mapping: arg i corresponds to member position i
                object.__setattr__(self, 'arg_to_member_pos', list(range(len(self.arguments))))
            else:
                object.__setattr__(self, 'arg_to_member_pos', None)
    
    def realize_constraints(self, args: Tuple[str, ...]) -> Optional[str]:
        """
        Constraint realization not yet implemented for set guards.
        
        Raises:
            NotImplementedError: This method needs to be implemented
        """
        raise NotImplementedError()
    
    @property
    def children(self) -> Tuple[IRNode, ...]:
        """Child node is the set expression being tested against."""
        return (self.set_expr,)
    
    def __repr__(self):
        """String representation showing arguments and set expression."""
        return f"SetGuard(args={[v.name for v in self.arguments]}, expr={repr(self.set_expr)})"


@dataclass(frozen=True)
class SMTGuard(Guard):
    """
    Guard that directly generates SMT-LIB constraints.
    
    Used as an internal representation for guards that can be directly
    translated to SMT solver format. Uses string templates with variable
    substitution to generate the final constraint expressions.
    """
    variables: Tuple[Argument, ...]  # Variables used in the SMT expression
    template: Template  # String template for SMT constraint with variable placeholders
    var_positions: Dict[str, int] = None  # type: ignore  # Variable name to position mapping
    
    def __post_init__(self):
        """
        Initialize variable position mapping for template substitution.
        
        Creates a dictionary mapping each variable name to its index position,
        enabling efficient lookup during constraint realization.
        """
        if self.var_positions is None:
            object.__setattr__(self, 'var_positions', 
                             {var.name: i for i, var in enumerate(self.variables)})
    
    @property
    def children(self) -> Tuple[IRNode, ...]:
        """SMT guards have no IR children (template is a string, not IR)."""
        return ()
    
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        """
        Generate concrete SMT constraint by substituting argument names into template.
        
        Args:
            args: Tuple of concrete variable names to substitute
            
        Returns:
            SMT-LIB formatted constraint string with variables substituted
        """
        # Build substitution dictionary mapping variable names to concrete argument names
        subs = {var.name: args[i] for i, var in enumerate(self.variables)}
        return self.template.substitute(subs)
    
    def __repr__(self) -> str:
        """String representation showing variables and template."""
        vars_str = ", ".join(var.name for var in self.variables)
        return f"SMTGuard(variables=({vars_str}), template={self.template.template})"
