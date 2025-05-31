import ply.lex as lex
import re

# Token list
tokens = (
    'ASSIGN', 'LET', 'IN', 'NEQ', 'LEQ', 'GEQ', 'IMPLY', 'EQUIV', 'UNION', 
    'INTERSECTION', 'OR', 'AND', 'NOT', 'EXISTS', 'FORALL', 'POSITIVES', 
    'INTEGERS', 'NATURALS', 'EMPTY', 'REALS', 'TRUE', 'FALSE', 'DOTS', 
    'UINT', 'IDENTIFIER', 'BACKSLASH'
)

# Literal characters (single-character tokens)
literals = ['+', '-', '*', '/', '=', '<', '>', '(', ')', '{', '}', '[', ']', ',', ';', ':', '.', '|', "^"]

# Reserved keywords
reserved = {
    'let': 'LET',
    'in': 'IN',
    'or': 'OR',
    'and': 'AND',
    'not': 'NOT',
    'exists': 'EXISTS',
    'forall': 'FORALL',
    'posi': 'POSITIVES',
    'int': 'INTEGERS',
    'nat': 'NATURALS',
    'empty': 'EMPTY',
    'real': 'REALS',
    'true': 'TRUE',
    'false': 'FALSE'
}

# Global state
armoise_current_line = 1
armoise_current_filename = ""
c_comment_start_line = 0

# ------ TOKEN RULES ------

# Simple tokens
t_ASSIGN = r':='
t_NEQ = r'!='
t_LEQ = r'<='
t_GEQ = r'>='
t_IMPLY = r'=>'
t_EQUIV = r'<=>'
t_UNION = r'\|\|'
t_INTERSECTION = r'&&'
t_DOTS = r'\.\.\.'
t_BACKSLASH = r'\\'

# Complex tokens and ignored patterns
def t_UINT(t):
    r'([1-9][0-9]*)|0'
    t.value = int(t.value)
    return t

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'IDENTIFIER')
    return t

t_ignore_CPP_COMMENT = r'//.*'
t_ignore_BLANK = r'[ \t\r]'

def t_NEWLINE(t):
    r'\n'
    global armoise_current_line
    armoise_current_line += 1
    t.lexer.lineno += len(t.value)

def t_COMMENT(t):
    r'/\*(.|\n)*?\*/'
    t.lexer.lineno += t.value.count('\n')

# Error handling
def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {armoise_current_line}")
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex(reflags=re.UNICODE)

# ------ INITIALIZATION FUNCTIONS ------
def armoise_init_lexer(stream, input_name):
    global armoise_current_line, armoise_current_filename
    armoise_current_line = 1
    armoise_current_filename = input_name
    lexer.input(stream.read() if hasattr(stream, 'read') else stream)

def armoise_terminate_lexer():
    global armoise_current_line, armoise_current_filename
    armoise_current_line = -1
    armoise_current_filename = None
    lexer.lexdata = None


if __name__ == "__main__":
    import sys

    # Initialize lexer
    with open(sys.argv[1], 'r') as f:
        armoise_init_lexer(f, 'source.arm')

    # Tokenize
    while True:
        tok = lexer.token()
        if not tok:
            break
        print(tok)

    # Clean up
    armoise_terminate_lexer()
