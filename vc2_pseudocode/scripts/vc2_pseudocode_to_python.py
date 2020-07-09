"""
Command-line utility for translating VC-2 pseudocode listings into Python.
"""

import sys

from argparse import ArgumentParser, FileType

from vc2_pseudocode.parser import PseudocodeParseError
from vc2_pseudocode.ast import ASTConstructionError

from vc2_pseudocode.python_transformer import pseudocode_to_python


def main(*args):
    parser = ArgumentParser(
        description="""
            Convert a VC-2 pseudocode listing into equivalent Python code.
        """
    )
    parser.add_argument(
        "pseudocode_file", type=FileType("r"), default=sys.stdin, nargs="?",
    )
    parser.add_argument(
        "python_file", type=FileType("w"), default=sys.stdout, nargs="?",
    )

    args = parser.parse_args(*args)

    try:
        python = pseudocode_to_python(args.pseudocode_file.read())
    except (PseudocodeParseError, ASTConstructionError) as e:
        sys.stderr.write(f"Syntax error: {str(e)}\n")
        return 1

    args.python_file.write(f"{python}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
