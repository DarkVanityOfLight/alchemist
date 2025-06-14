import sys
import argparse
import logging
from typing import cast, Optional

from arm_ast import ASTNode
from irparsers import convert
from lexer import lexer
from optimizer import optimize
from parser import parser, armoise_syntax_tree, armoise_current_filename
from smt2_emitter import emit


def parse_file(filename: str) -> Optional[ASTNode]:
    """
    Parse the given source file into an ASTNode. Returns None on failure.
    """
    try:
        with open(filename, 'r') as file:
            data = file.read()
    except FileNotFoundError:
        logging.error("File not found: %s", filename)
        return None
    except Exception as e:
        logging.error("Error reading file %s: %s", filename, e)
        return None

    # Set current filename for error reporting
    global armoise_current_filename
    armoise_current_filename = filename

    # Parse the input
    result = parser.parse(data, lexer=lexer, tracking=True)
    if armoise_syntax_tree:
        return armoise_syntax_tree
    return result


def main():
    parser_cli = argparse.ArgumentParser(
        description="Alchemist: Parse, optimize, and emit SMT-LIB2 from Armoise source files."
    )
    parser_cli.add_argument(
        "files", nargs='+', help="One or more source files to compile"
    )
    parser_cli.add_argument(
        "-r", "--relation-name", default="R", help="Set the output relation name default is R"
    )
    parser_cli.add_argument(
        "-o", "--output", help="Write output to file (defaults to STDOUT)"
    )
    parser_cli.add_argument(
        "-v", "--verbose", action='store_true', help="Enable verbose logging"
    )

    args = parser_cli.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    combined_buffer = []
    for fname in args.files:
        # logging.info("Processing file: %s", fname)
        ast: Optional[ASTNode] = cast(Optional[ASTNode], parse_file(fname))
        if ast is None:
            logging.error("Skipping due to parse errors: %s", fname)
            continue

        ir, scopes = convert(ast)
        logging.debug("Initial IR: %s", ir)

        optimized_ir = optimize(ir, scopes)
        logging.debug("Optimized IR: %s", optimized_ir)

        smt_output = emit(optimized_ir, args.relation_name)
        combined_buffer.append(smt_output)

    final_output = "\n".join(combined_buffer)

    if args.output:
        try:
            with open(args.output, 'w') as out_file:
                out_file.write(final_output)
            logging.info("Output written to %s", args.output)
        except Exception as e:
            logging.error("Failed to write output to %s: %s", args.output, e)
            sys.exit(2)
    else:
        print(final_output)


if __name__ == "__main__":
    main()
