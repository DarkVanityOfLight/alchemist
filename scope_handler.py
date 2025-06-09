from typing import Any, Dict, List

from arm_ast import ASTNode

class InProgressMarker:
    """Sentinel to detect recursive resolution."""
    def __init__(self, ast: ASTNode):
        self.ast = ast
    def __repr__(self):
        return f"<InProgress {self.ast}>"

class ScopeHandler:
    """Handles nested scopes, raw AST defs, and on-demand parsed cache."""

    def __init__(self) -> None:
        # Each frame maps name → raw ASTNode
        self._raw_scopes: List[Dict[str, ASTNode]] = [{}]
        # Cache maps raw ASTNode → parsed result (SymbolicSet, VectorSpace, etc.)
        self._parsed_cache: Dict[ASTNode, Any] = {}

    def enter_scope(self) -> None:
        """Push a fresh inner scope frame."""
        self._raw_scopes.append({})

    def exit_scope(self) -> None:
        """Pop the innermost scope frame (but never the global)."""
        if len(self._raw_scopes) > 1:
            self._raw_scopes.pop()

    def add_definition(self, name: str, raw_ast: ASTNode) -> None:
        """
        Record the un-parsed AST for `name` in the current scope.
        This invalidates any previously cached parse for the same AST.
        """
        self._raw_scopes[-1][name] = raw_ast
        # If this AST was previously parsed (e.g. via shadowing), drop it:
        self._parsed_cache.pop(raw_ast, None)

    def _lookup_raw(self, name: str) -> ASTNode:
        """
        Find the raw ASTNode for `name` by searching
        from innermost to outermost scope. Raises KeyError if undefined.
        """
        for frame in reversed(self._raw_scopes):
            if name in frame:
                return frame[name]
        raise KeyError(f"Undefined identifier: {name}")

    def resolve(self, name: str) -> Any:
        """
        On-demand parse & cache of the definition named `name`.
        Uses the ASTNode itself as the cache key (so shadowing works).
        Detects simple recursive cycles via InProgressMarker.
        """
        raw_ast = self._lookup_raw(name)

        # If we've already parsed this exact AST node, return it:
        if raw_ast in self._parsed_cache:
            val = self._parsed_cache[raw_ast]
            if isinstance(val, InProgressMarker):
                raise ValueError(f"Recursive definition detected for '{name}'")
            return val

        # Mark as in-progress to catch self-recursion
        self._parsed_cache[raw_ast] = InProgressMarker(raw_ast)

        # Actually parse the AST into your IR (e.g., SetComprehension, VectorSpace, etc.)
        from irparsers import process_predicate
        parsed = process_predicate(raw_ast, self)

        # Cache and return the real result
        self._parsed_cache[raw_ast] = parsed
        return parsed

    def __contains__(self, name: str) -> bool:
        """True if `name` is defined in any scope frame."""
        try:
            self._lookup_raw(name)
            return True
        except KeyError:
            return False

    def __getitem__(self, name: str) -> Any:
        """
        Convenient lookup that returns the parsed definition.
        If you want the raw AST, use _lookup_raw instead.
        """
        return self.resolve(name)
