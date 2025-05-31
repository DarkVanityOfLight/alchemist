from __future__ import annotations
from abc import abstractmethod
from typing import Dict, Literal, Self, Tuple, List, Optional, cast
from arm_ast import ASTNode, NodeType
from dataclasses import dataclass


def assert_node_type(node: ASTNode, expected_type):
    if node.type != expected_type:
        raise AssertionError(
            f"Expected node of type {expected_type}, got {node.type} "
            f"at line {node.line}"
        )

class ScopeHandler:
    """Handles nested scopes and symbol lookup"""

    def __init__(self) -> None:
        self.scopes = [{}]

    def enter_scope(self):
        """Push a new scope when entering let-block"""
        self.scopes.append({})
    
    def exit_scope(self):
        """Pop scope when exiting in-block"""
        if len(self.scopes) > 1:
            self.scopes.pop()
    
    def add_definition(self, name: str, value):
        """Add definition to current scope"""
        self.scopes[-1][name] = value
        
    def lookup(self, name: str):
        """Search scopes from innermost to outermost"""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None  # Not found


    def __getitem__(self, name: str):
        value = self.lookup(name)
        if value is None:
            raise KeyError(name)
        return value

    def __contains__(self, name: str) -> bool:
        return self.lookup(name) is not None

type TypeRestriction = Literal["Int", "Nat", "Real"]

@dataclass(frozen=True)
class Variable:
    name: str
    domain: TypeRestriction

    def __repr__(self):
        return f"{self.name}: {self.domain}"



class TupleDomain:
    """Represents a product type domain like (int, int, nat)"""

    def __init__(self, types: Tuple[TypeRestriction, ...]) -> None:
        self.types = types

    def __len__(self):
        return len(self.types)

    def __getitem__(self, index: int) -> TypeRestriction:
        return self.types[index]

    def __iter__(self):
        return iter(self.types)

    def __repr__(self):
        return f"TupleDomain({', '.join(self.types)})"


class Guard:
    """
    Represent a guard.
    Either we have a "function" application to a lower set
    Or we have a presburger guard
    """
    @abstractmethod
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        pass

class SimpleGuard(Guard):
    def __init__(self, node: ASTNode, variables: Tuple[Variable, ...]) -> None:
        self.node = node
        self.variables = variables
        # Create mapping from variable name to position
        self.var_positions = {var.name: i for i, var in enumerate(variables)}

    def _lookup_var(self, variable: str) -> int:
        return self.var_positions[variable]
    
    # Add to SimpleGuard class
    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        # Check for real variables (not supported)
        if any(var.domain == "Real" for var in self.variables):
            raise NotImplementedError("Real variables not supported yet")
        
        # Create mapping: variable name -> SMT variable string
        var_map = {var.name: args[i] for i, var in enumerate(self.variables)}
        return self._convert_node(self.node, var_map)

    def _get_op_str(self, node_typ: NodeType) -> str:
        op_map = {
            NodeType.AND: "and",
            NodeType.OR: "or",
            NodeType.IMPLY: "=>",
            NodeType.EQUIV: "=",
            NodeType.PLUS: "+",
            NodeType.MINUS: "-",
            NodeType.MUL: "*",
            NodeType.DIV: "div",
            NodeType.MOD: "mod",
            NodeType.EQ: "=",
            NodeType.NEQ: "distinct",
            NodeType.LT: "<",
            NodeType.GT: ">",
            NodeType.LEQ: "<=",
            NodeType.GEQ: ">=",
            NodeType.NOT: "not",
            NodeType.NEG: "-",
        }
        return op_map[node_typ]

    def _convert_node(self, node: ASTNode, var_map: dict[str, str]) -> str:
        # Handle identifiers
        if node.type == NodeType.IDENTIFIER:
            return var_map[node.value]
        
        # Handle constants
        if node.type == NodeType.INTEGER:
            return str(node.value)
        if node.type == NodeType.TRUE:
            return "true"
        if node.type == NodeType.FALSE:
            return "false"
        
        # Unary operators
        if node.type in (NodeType.NOT, NodeType.NEG):
            child_str = self._convert_node(node.child, var_map)
            op_str = self._get_op_str(node.type)
            return f"({op_str} {child_str})"
        
        # Binary operators
        binary_ops = [
            NodeType.AND, NodeType.OR, NodeType.IMPLY, NodeType.EQUIV,
            NodeType.PLUS, NodeType.MINUS, NodeType.MUL, NodeType.DIV, NodeType.MOD,
            NodeType.EQ, NodeType.NEQ, NodeType.LT, NodeType.GT, NodeType.LEQ, NodeType.GEQ
        ]
        if node.type in binary_ops:
            left = node.child
            right = node.child.next
            if left is None or right is None:
                raise Exception("Binary operator missing children")
            left_str = self._convert_node(left, var_map)
            right_str = self._convert_node(right, var_map)
            op_str = self._get_op_str(node.type)
            return f"({op_str} {left_str} {right_str})"
        
        raise NotImplementedError(f"Unsupported node type: {node.type}")


    def __repr__(self):
        return f"SimpleGuard(variables={[v.name for v in self.variables]})"
    
class SetGuard(Guard):
    def __init__(self, arguments: Tuple[Variable, ...], set_predicate: SetPredicate) -> None:
        self.arguments = arguments
        self.set_predicate = set_predicate
        
        # Map positions: [index in arguments -> index in set_predicate.members]
        self.arg_to_member_pos = [
            i for i, _ in enumerate(set_predicate.members)
        ]  # Default to same order

    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        # Reorder args to match set predicate's member order
        reordered_args = tuple(
            args[i] for i in self.arg_to_member_pos
        )
        # Emit the full SetPredicate constraints (domain + guard)
        return self.set_predicate.realize_constraints(reordered_args)

    def __repr__(self):
            return f"SetGuard(args={[v.name for v in self.arguments]}, pred={repr(self.set_predicate)})"

class SetPredicate:
    """Represent a set comprehension"""
    def __init__(self, members: Tuple[Variable, ...], domain : SetPredicate | TupleDomain, guard: Guard) -> None:
        self.members = members
        self.domain = domain
        self.guard = guard

        if isinstance(domain, SetPredicate):
            # Make sure members match up
            assert len(self.members) == len(cast(Self, domain).members)
        elif isinstance(domain, TupleDomain):
            assert len(self.members) == len(cast(TupleDomain, domain))

    def realize_constraints(self, args: Tuple[str, ...]) -> str:
        """
        Convert set predicate to SMT-LIB constraints
        :param args: SMT variables corresponding to self.members
        :return: SMT-LIB formula representing the set membership condition
        """
        # Domain constraints based on domain type
        domain_cond = ""
        if isinstance(self.domain, TupleDomain):
            # Handle base domain (Int, Nat, Real)
            conditions = []
            for i, typ in enumerate(self.domain.types):
                smt_var = args[i]
                if typ == "Nat":
                    conditions.append(f"(>= {smt_var} 0)")
                elif typ == "Real":
                    raise NotImplementedError("Real numbers not supported")
            if conditions:
                domain_cond = "(and " + " ".join(conditions) + ")"
        elif isinstance(self.domain, SetPredicate):
            # Recursively handle nested set domain
            domain_cond = self.domain.realize_constraints(args)
        
        # Convert the guard condition
        guard_cond = self.guard.realize_constraints(args)
        
        # Combine domain and guard conditions
        if domain_cond:
            return f"(and {domain_cond} {guard_cond})"
        return guard_cond

        

    def __repr__(self):
        return f"SetPredicate({self.members} IN {self.domain} WHERE {self.guard})"

type Predicate = SetPredicate | TupleDomain

def _process_guard(node: ASTNode, variables: Tuple[Variable, ...], scopes: ScopeHandler) -> Guard:

    # Set guard
    if node.type == NodeType.IN:

        arguments = node.child
        assert_node_type(arguments, NodeType.VECTOR)
        set_predicate = scopes.lookup(node.child.next.value)
        assert set_predicate != None
        assert isinstance(set_predicate, SetPredicate)

        return SetGuard(variables, set_predicate)

    # Formula guard
    return SimpleGuard(node, variables)



def _process_members(node: ASTNode) -> Tuple[str, ...]:
    assert_node_type(node, NodeType.VECTOR)
    return tuple(child.value for child in node)


def _process_set(node: ASTNode, scopes: ScopeHandler) -> SetPredicate:
    assert_node_type(node, NodeType.SET)

    # SET 
    #   IN
    #       VECTOR
    #       DOMAIN 
    #   GUARD

    members_in_domain = node.child
    assert_node_type(members_in_domain, NodeType.IN)

    member_names = _process_members(members_in_domain.child)
    domain_name = members_in_domain.child.next.value
    domain = scopes[domain_name]
    assert isinstance(domain, (SetPredicate, TupleDomain))

    # Lookup members
    members : Tuple[Variable, ...]
    if isinstance(domain, TupleDomain):
        assert len(cast(TupleDomain, domain)) == len(member_names)

        domain_types = domain
        members = tuple(Variable(name, typ) for name, typ in zip(member_names, domain_types))

    elif isinstance(domain, SetPredicate):
        members = domain.members
    else:
        raise Exception("Unknown domain type")

    # Make the guard
    guard = _process_guard(members_in_domain.next, members, scopes)

    return SetPredicate(members, domain, guard)


def _process_definition(node: ASTNode, scopes: ScopeHandler):
    """Process the definition into scope"""
    assert_node_type(node, NodeType.DEFINITION)
    
    ident = node.child.value
    value = _process_predicate(node.child.next, scopes) # FIXME: this does not have to be a predicate


    scopes.add_definition(ident, value)

def _process_predicate_context(node: ASTNode, scopes: ScopeHandler):
    """Process the predicate context into the current scope"""
    assert_node_type(node, NodeType.PREDICATE_CONTEXT)

    for child in node:
        assert_node_type(child, NodeType.DEFINITION)
        _process_definition(child, scopes)



def process_typed_tuple(node: ASTNode) -> TupleDomain:
    assert_node_type(node, NodeType.PAREN)
    children = [child for child in node]

    def _map(node: ASTNode) -> TypeRestriction:
        match node.type:
            case NodeType.INTEGERS: return "Int"
            case NodeType.NATURALS: return "Nat"
            case NodeType.REALS: return "Real"
        raise Exception("Unknown type")

    mapped_children = cast(Tuple[TypeRestriction, ...], tuple(map(_map, children)))
    return TupleDomain(mapped_children)
            

def _process_predicate(node: ASTNode, scopes: ScopeHandler) -> Predicate:
    assert_node_type(node, NodeType.PREDICATE)

    has_context = node.child.type == NodeType.PREDICATE_CONTEXT

    # Check if we need to process context first
    if has_context:
        scopes.enter_scope()
        _process_predicate_context(node.child, scopes)


    content: ASTNode  = node.child.next if has_context else node.child

    # Process the predicate, either a set or a tuple
    predicate : Predicate
    match content.type:
        case NodeType.PAREN:
            predicate = process_typed_tuple(content)
        case NodeType.SET:
            predicate = _process_set(content, scopes)
        case typ:
            raise Exception(f"Unknown predicate {typ} in line: {content.line}")

    if has_context:
        scopes.exit_scope() # Leave the scope before returning the set

    return predicate


def convert(ast: ASTNode) -> SetPredicate:
    scopes = ScopeHandler()

    assert_node_type(ast, NodeType.PREDICATE)
    assert_node_type(ast.child, NodeType.PREDICATE_CONTEXT)
    assert ast.child.type == NodeType.PREDICATE_CONTEXT

    res = _process_predicate(ast, scopes)
    assert isinstance(res, SetPredicate)
    return res
    
