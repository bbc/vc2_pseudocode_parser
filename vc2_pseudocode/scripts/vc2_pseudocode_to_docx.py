"""
Command-line utility for translating VC-2 pseudocode listings into SMPTE-style
source listings in Word (docx) format.
"""

import sys

from typing import Any

from argparse import ArgumentParser, FileType

from vc2_pseudocode.pseudocode_parser import ParseError, ASTConstructionError

from vc2_pseudocode.docx_transformer import pseudocode_to_docx


def main(*args: Any) -> int:
    parser = ArgumentParser(
        description="""
            Convert a VC-2 pseudocode listing into a SMPTE-style code listing
            Word (docx) document.
        """
    )
    parser.add_argument("pseudocode_file", type=FileType("r"))
    parser.add_argument("docx_file", type=str)

    args = parser.parse_args(*args)

    try:
        pseudocode_to_docx(args.pseudocode_file.read(), args.docx_file)  # type: ignore
    except (ParseError, ASTConstructionError) as e:
        sys.stderr.write(f"Syntax error: {str(e)}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
