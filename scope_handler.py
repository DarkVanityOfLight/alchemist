from __future__ import annotations
from typing import Dict
import typing

if typing.TYPE_CHECKING:
    from expressions import IRNode

class ScopeHandler:
    """
    Handles nested scopes while parsing, globalizes the node identifiers and 
    keeps track of the globalized id -> node mapping
    """

    def __init__(self) -> None:
        self.scopes = [{}] # Local definitions
        self.parsed_ids : Dict[int, IRNode]= {} # Globalized identifiers

    def enter_scope(self):
        """Push a new scope when entering let-block"""
        self.scopes.append({})
    
    def exit_scope(self):
        """Pop scope when exiting in-block"""
        if len(self.scopes) > 1:
            self.scopes.pop()
    
    def add_definition(self, name: str, value: IRNode):
        """Add definition to current scope"""
        self.scopes[-1][name] = value
        self.parsed_ids[value.id] = value
        
    def lookup_by_name(self, name: str):
        """Search scopes from innermost to outermost"""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None  # Not found

    def lookup_by_id(self, id: int):
        """Lookup a global parsed subtree identifier by it's id"""
        return self.parsed_ids.get(id, None)

