
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
