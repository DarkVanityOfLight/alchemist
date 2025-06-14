import sys
from typing import cast
from arm_ast import ASTNode
from irparsers import convert
from lexer import lexer
from optimizer import optimize
from parser import parser, armoise_syntax_tree, armoise_current_filename
from smt2_emitter import emit

def parse_file(filename):
    global armoise_current_filename
    
    try:
        with open(filename, 'r') as file:
            data = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    
    # Set current filename for error reporting
    armoise_current_filename = filename
    
    # Parse the input
    result = parser.parse(data, lexer=lexer, tracking=True)
    
    if armoise_syntax_tree:
        return armoise_syntax_tree
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <source_file>")
        sys.exit(1)
    
    source_file = sys.argv[1]
    ast : ASTNode = cast(ASTNode, parse_file(source_file))
    ir, scopes = convert(ast)
    optimized_ir = optimize(ir, scopes)
    print(emit(optimized_ir))



    # ir = convert(ast)
    #
    # args = [f"arg{i}" for i in range(ir.dim)]
    # smt_str = ir.realize_constraints(tuple(args))
    #
    # print(args)
    # print(smt_str)

