from ply.lex import LexError
from arm_ast import ASTNode, ValueType, NodeType
from ply import yacc
from lexer import tokens

armoise_syntax_tree = None
armoise_current_line = 1
armoise_current_filename = ""
armoise_number_of_errors = 0
MAX_NB_SYNTAX_ERROR = 1

def U_NE(node_type, child, line):
    return ASTNode(node_type, ValueType.EMPTY, None, line, armoise_current_filename, child, None)

def B_NE(node_type, child1, child2, line):
    aux = U_NE(node_type, child1, line)
    child1.next = child2
    return aux

def NID(value, line):
    return ASTNode(NodeType.IDENTIFIER, ValueType.ID, value, line, armoise_current_filename, None, None)

def NINT(value, line):
    return ASTNode(NodeType.INTEGER, ValueType.INT, value, line, armoise_current_filename, None, None)


precedence = (
    ('left', '+', '-'),
    ('left', '*', '/', '%'),
    ('right', 'UMINUS', 'UPLUS', 'NOT', '!'),
)


def p_grammar_start_rule(p):
    '''grammar_start_rule : armoise_description
                          | empty'''
    global armoise_syntax_tree
    armoise_syntax_tree = p[1]
    p[0] = p[1]

def p_empty(p):
    'empty :'
    p[0] = None

def p_armoise_description(p):
    'armoise_description : definition_or_predicate_list'
    p[0] = p[1]

def p_definition_or_predicate_list(p):
    '''definition_or_predicate_list : definition_or_predicate ';' definition_or_predicate_list
    | definition_or_predicate ';' '''
    if len(p) == 4:
        p[1].next = p[3]
        p[0] = p[1]
    else:
        p[0] = p[1]

def p_definition_or_predicate(p):
    '''definition_or_predicate : predicate
                              | predicate_definition'''
    p[0] = p[1]

def p_predicate_definition(p):
    'predicate_definition : identifier_or_tuple ASSIGN predicate'
    p[0] = B_NE(NodeType.DEFINITION, p[1], p[3], p.lineno(1))

def p_identifier_or_tuple(p):
    '''identifier_or_tuple : tuple_of_identifiers
                           | identifier'''
    p[0] = p[1]

def p_tuple_of_identifiers(p):
    '''tuple_of_identifiers : '<' list_of_tuples '>' '''
    p[0] = U_NE(NodeType.ID_TUPLE, p[2], p.lineno(1))

def p_list_of_tuples(p):
    '''list_of_tuples : identifier_or_tuple ',' list_of_tuples
                      | identifier_or_tuple'''
    p[0] = p[1]
    if len(p) > 2:
        p[1].next = p[3]

def p_identifier(p):
    'identifier : IDENTIFIER'
    p[0] = NID(p[1], p.lineno(1))

def p_predicate(p):
    '''predicate : predicate_without_local_context
                 | local_context tuple_of_predicates_without_local_context
                 | local_context predicate_without_local_context'''
    if len(p) == 2:
        p[0] = U_NE(NodeType.PREDICATE, p[1], p.lineno(1))
    else:
        p[0] = B_NE(NodeType.PREDICATE, p[1], p[2], p.lineno(1))

def p_local_context(p):
    'local_context : LET list_of_predicate_definitions IN'
    p[0] = U_NE(NodeType.PREDICATE_CONTEXT, p[2], p.lineno(1))

def p_list_of_predicate_definitions(p):
    '''list_of_predicate_definitions : predicate_definition ';' list_of_predicate_definitions
                                     | predicate_definition ';' '''
    p[0] = p[1]
    if len(p) > 3:
        p[1].next = p[3]

def p_tuple_of_predicates_without_local_context(p):
    '''tuple_of_predicates_without_local_context : '<' list_of_tuple_of_predicates '>' '''
    p[0] = U_NE(NodeType.PREDICATE_TUPLE, p[2], p.lineno(1))

def p_list_of_tuple_of_predicates(p):
    '''list_of_tuple_of_predicates : predicate_without_local_context ',' list_of_tuple_of_predicates
                                   | predicate_without_local_context'''
    p[0] = p[1]
    if len(p) > 2:
        p[1].next = p[3]

def p_predicate_without_local_context(p):
    'predicate_without_local_context : set'
    p[0] = p[1]

def p_set(p):
    '''set : set UNION set
           | set BACKSLASH set
           | set '^' set
           | set INTERSECTION set
           | set '+' set
           | set '-' set
           | set '*' set
           | set '/' set
           | set '%' set
           | '!' set %prec '!'
           | '-' set %prec UMINUS
           | NOT set
           | '(' list_of_boolean_combinations ')'
           | '(' set ')'
           | simple_set
           | list_element
           | range
           | set '$' integer
           | integer
           | '+' integer %prec UPLUS
           | NATURALS
           | INTEGERS
           | POSITIVES
           | REALS
           | EMPTY'''
    # Binary‐operator cases (length == 4)
    if len(p) == 4 and isinstance(p[1], ASTNode) and isinstance(p[3], ASTNode):
        # Determine which operator token was used
        tok_type = p.slice[2].type
        if tok_type == 'UNION':
            op = NodeType.UNION
        elif tok_type == 'BACKSLASH':
            op = NodeType.DIFFERENCE
        elif tok_type == '^':
            op = NodeType.XOR
        elif tok_type == 'INTERSECTION':
            op = NodeType.INTERSECTION
        elif tok_type == '+':
            op = NodeType.PLUS
        elif tok_type == '-':
            op = NodeType.MINUS
        elif tok_type == '*':
            op = NodeType.MUL
        elif tok_type == '/':
            op = NodeType.DIV
        elif tok_type == '%':
            op = NodeType.MOD
        elif tok_type == '$':
            op = NodeType.POWER
        else:
            # Parenthesized‐tuple case:  '( list_of_boolean_combinations )'
            if p[1] == '(' and isinstance(p[2], list):
                op = NodeType.CARTESIAN_PRODUCT
                p[0] = U_NE(op, p[2], p.lineno(1))
                return
            # Parenthesized‐set case: '( set )'
            else:
                op = NodeType.PAREN
                p[0] = U_NE(op, p[2], p.lineno(1))
                return

        p[0] = B_NE(op, p[1], p[3], p.lineno(2))

    # Unary‐operator cases (length == 3)
    elif len(p) == 3:
        if p.slice[1].type in ('!', 'NOT'):
            op = NodeType.COMPLEMENT
            p[0] = U_NE(op, p[2], p.lineno(1))
        elif p.slice[1].type == '-':  # UMINUS
            op = NodeType.NEG
            p[0] = U_NE(op, p[2], p.lineno(1))
        else:  # '+' integer with UPLUS precedence
            p[0] = p[2]

    # All other cases (length == 2 or more specific patterns)
    else:
        # NATURALS, INTEGERS, POSITIVES, REALS, EMPTY
        if p[1] == 'nat':
            p[0] = U_NE(NodeType.NATURALS, None, p.lineno(1))
        elif p[1] == 'int':
            p[0] = U_NE(NodeType.INTEGERS, None, p.lineno(1))
        elif p[1] == 'posi':
            p[0] = U_NE(NodeType.POSITIVES, None, p.lineno(1))
        elif p[1] == 'real':
            p[0] = U_NE(NodeType.REALS, None, p.lineno(1))
        elif p[1] == 'empty':
            p[0] = U_NE(NodeType.EMPTY, None, p.lineno(1))
        # Parenthesized boolean combinations: '( list_of_boolean_combinations )'
        elif p[1] == '(' and isinstance(p[2], list) and p[3] == ')':
            p[0] = U_NE(NodeType.CARTESIAN_PRODUCT, p[2], p.lineno(1))
        # Parenthesized single set: '( set )'
        elif p[1] == '(' and isinstance(p[2], ASTNode) and p[3] == ')':
            p[0] = U_NE(NodeType.PAREN, p[2], p.lineno(1))
        # Simple‐set, list‐element, range, or integer literal
        elif isinstance(p[1], ASTNode):
            p[0] = p[1]
        # Binary power: 'set $ integer'
        elif len(p) == 4 and p.slice[2].type == '$':
            p[0] = B_NE(NodeType.POWER, p[1], p[3], p.lineno(2))
        # '+' integer (unary plus)
        elif len(p) == 3 and p.slice[1].type == '+':
            p[0] = p[2]
        else:
            p[0] = p[1]

def p_integer(p):
    'integer : UINT'
    p[0] = NINT(p[1], p.lineno(1))

def p_list_element(p):
    '''list_element : identifier
                    | list_element '[' integer ']' '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = B_NE(NodeType.LIST_ELEMENT, p[1], p[3], p.lineno(2))

def p_list_of_boolean_combinations(p):
    '''list_of_boolean_combinations : set ',' list_of_boolean_combinations
                                    | set ',' set'''
    p[0] = p[1]
    if len(p) > 3:
        p[1].next = p[3]
    else:
        p[1].next = p[3]

def p_list_of_terms(p):
    '''list_of_terms : term ',' list_of_terms
                     | term'''
    p[0] = p[1]
    if len(p) > 2:
        p[1].next = p[3]

def p_vector_of_terms(p):
    '''vector_of_terms : '(' list_of_terms ')' '''
    p[0] = U_NE(NodeType.VECTOR, p[2], p.lineno(1))

def p_formula(p):
    '''formula : formula '|' formula
               | formula OR formula
               | formula '&' formula
               | formula AND formula
               | formula EQUIV formula
               | formula IMPLY formula
               | term_comparison
               | quantified_formula
               | '!' formula %prec NOT
               | NOT formula
               | term IN set
               | '(' formula ')' 
               | identifier vector_of_terms
               | TRUE
               | FALSE'''
    if len(p) == 4:
        if p[2] == '|' or p[2] == 'or': op = NodeType.OR
        elif p[2] == '&' or p[2] == 'and': op = NodeType.AND
        elif p[2] == '<=>': op = NodeType.EQUIV
        elif p[2] == '=>': op = NodeType.IMPLY
        elif p[2] == 'in': op = NodeType.IN
        else:  # Parentheses
            p[0] = U_NE(NodeType.PAREN, p[2], p.lineno(1))
            return
        p[0] = B_NE(op, p[1], p[3], p.lineno(2))
    elif len(p) == 3:
        if p[1] == '!' or p[1] == 'not':
            p[0] = U_NE(NodeType.NOT, p[2], p.lineno(1))
        else:  # identifier vector_of_terms
            p[0] = B_NE(NodeType.CALL, p[1], p[2], p.lineno(1))
    elif len(p) == 2:
        if p[1] == 'true':
            p[0] = U_NE(NodeType.TRUE, None, p.lineno(1))
        elif p[1] == 'false':
            p[0] = U_NE(NodeType.FALSE, None, p.lineno(1))
        else:
            p[0] = p[1]  # term_comparison or quantified_formula

def p_term_comparison(p):
    '''term_comparison : term '=' term
                       | term NEQ term
                       | term '<' term
                       | term '>' term
                       | term LEQ term
                       | term GEQ term'''
    if p[2] == '=': op = NodeType.EQ
    elif p[2] == '!=': op = NodeType.NEQ
    elif p[2] == '<': op = NodeType.LT
    elif p[2] == '>': op = NodeType.GT
    elif p[2] == '<=': op = NodeType.LEQ
    elif p[2] == '>=': op = NodeType.GEQ
    else: raise LexError("", "")
    p[0] = B_NE(op, p[1], p[3], p.lineno(2))

def p_quantified_formula(p):
    '''quantified_formula : EXISTS typed_identifier formula
                          | FORALL typed_identifier formula'''
    op = NodeType.EXISTS if p[1] == 'exists' else NodeType.FORALL
    p[0] = B_NE(op, p[3], p[2], p.lineno(1))

def p_typed_identifier(p):
    '''typed_identifier : identifier IN predicate_without_local_context
                        | vector_of_terms IN predicate_without_local_context'''
    p[0] = B_NE(NodeType.IN, p[1], p[3], p.lineno(2))

def p_term(p):
    '''term : term '+' term
            | term '-' term
            | term '*' term
            | term '/' term
            | term '%' term
            | '-' term %prec UMINUS
            | vector_of_terms
            | list_element
            | integer
            | '+' integer %prec UPLUS'''
    if len(p) == 4:
        if p[2] == '+': op = NodeType.PLUS
        elif p[2] == '-': op = NodeType.MINUS
        elif p[2] == '*': op = NodeType.MUL
        elif p[2] == '/': op = NodeType.DIV
        elif p[2] == '%': op = NodeType.MOD
        else: raise LexError("", "")
        p[0] = B_NE(op, p[1], p[3], p.lineno(2))
    elif len(p) == 3:
        if p[1] == '-':
            p[0] = U_NE(NodeType.NEG, p[2], p.lineno(1))
        else:  # UPLUS_ARITH
            p[0] = p[2]
    else:
        p[0] = p[1]

def p_error(p):
    global armoise_number_of_errors
    if p:
        print(f"Syntax error at line {p.lineno}, token '{p.value}'")
    else:
        print("Syntax error at EOF")
    armoise_number_of_errors += 1
    if armoise_number_of_errors >= MAX_NB_SYNTAX_ERROR:
        raise SyntaxError("Too many syntax errors")

def p_simple_set(p):
    '''simple_set : '{' list_of_terms '}'
                  | '{' typed_identifier '|' formula '}'
                  | '{' typed_identifier '}' '''
    if len(p) == 4:
        p[0] = U_NE(NodeType.ENUMERATED_SET, p[2], p.lineno(1))
    elif len(p) == 6:
        p[0] = B_NE(NodeType.SET, p[2], p[4], p.lineno(1))
    else:
        true_node = U_NE(NodeType.TRUE, None, p.lineno(1))
        p[0] = B_NE(NodeType.SET, p[2], true_node, p.lineno(1))

def p_range(p):
    '''range : '[' integer DOTS integer ']'
             | '[' integer ',' integer ']' '''
    p[0] = s_make_range(p[2].value, p[4].value, 1 if p[3] == '...' else 0)

def s_make_range(minv, maxv, in_nat):
    # AST construction logic
    proto = NID("_x", 0)
    min_node = NINT(minv, 0)
    max_node = NINT(maxv, 0)

    # Build comparison nodes
    min_leq_x = B_NE(NodeType.LEQ, min_node, proto, 0)
    x_leq_max = B_NE(NodeType.LEQ, proto, max_node, 0)
    and_node = B_NE(NodeType.AND, min_leq_x, x_leq_max, 0)

    # Determine domain
    if in_nat:
        domain = U_NE(NodeType.NATURALS, None, 0)
    else:
        domain = U_NE(NodeType.REALS, None, 0)

    # Build final set
    in_node = B_NE(NodeType.IN, proto, domain, 0)
    return B_NE(NodeType.SET, in_node, and_node, 0)

parser = yacc.yacc(
    debug=False,          # Disable debug output for production
    optimize=True,        # Enable optimization features
    tabmodule='armoise_parsetab',  # Specifies cache file name
    outputdir='',         # Save cache in current directory
    write_tables=True     # Force table generation/caching
)

