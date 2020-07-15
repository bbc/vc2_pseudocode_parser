from vc2_pseudocode_parser.parser import __all__ as pseudocode_parser_all

from vc2_pseudocode_parser.parser.grammar import __all__ as grammar_all
from vc2_pseudocode_parser.parser.operators import __all__ as operators_all
from vc2_pseudocode_parser.parser.ast import __all__ as ast_all
from vc2_pseudocode_parser.parser.parser import __all__ as parser_all


def test_all_is_complete() -> None:
    assert sorted(pseudocode_parser_all) == sorted(
        grammar_all + operators_all + ast_all + parser_all
    )
