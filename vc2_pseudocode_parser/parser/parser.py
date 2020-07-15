"""
A parser for the pseudocode language which produces an Abstract Syntax Tree
(AST) representation of the underlying code.
"""

from typing import Optional, Mapping, Set, Union, cast

from peggie import Parser, ParseError, RuleExpr, RegexExpr

from vc2_pseudocode_parser.parser.grammar import grammar
from vc2_pseudocode_parser.parser.ast import ToAST, infer_labels, Listing

__all__ = [
    "parse",
    "ParseError",  # NB: Re-exported
]


parse_error_default_expr_explanations: Mapping[
    Union[RuleExpr, RegexExpr], Optional[str]
] = {
    # Basic units
    RuleExpr("start"): "<function-definition>",
    RuleExpr("function"): "<function-definition>",
    RuleExpr("expr"): "<expression>",
    RuleExpr("stmt"): "<statement>",
    RuleExpr("single_line_stmt"): "<single-line-statement>",
    RuleExpr("identifier"): "<identifier>",
    # Expression sub-rules
    RuleExpr("maybe_log_or_expr"): "<expression>",
    RuleExpr("maybe_log_and_expr"): "<expression>",
    RuleExpr("maybe_log_not_expr"): "<expression>",
    RuleExpr("maybe_cmp_expr"): "<expression>",
    RuleExpr("maybe_or_expr"): "<expression>",
    RuleExpr("maybe_xor_expr"): "<expression>",
    RuleExpr("maybe_and_expr"): "<expression>",
    RuleExpr("maybe_shift_expr"): "<expression>",
    RuleExpr("maybe_arith_expr"): "<expression>",
    RuleExpr("maybe_prod_expr"): "<expression>",
    RuleExpr("maybe_unary_expr"): "<expression>",
    RuleExpr("maybe_pow_expr"): "<expression>",
    RuleExpr("maybe_peren_expr"): "<expression>",
    # Optional whitespace
    RuleExpr("ws"): None,
    # Mandatory vertical whitespace
    RuleExpr("comment"): "<newline>",
    RuleExpr("v_space"): "<newline>",
    RuleExpr("eol"): "<newline>",
    RuleExpr("eof"): "<newline>",
    # Mandatory horizontal whitespace
    RuleExpr("ws_"): "<space>",
    RuleExpr("h_space"): "<space>",
    # Operators
    RegexExpr(r"<<|>>"): "<operator>",
    RegexExpr(r"==|!=|<=|>=|<|>"): "<operator>",
    RegexExpr(r"not"): "<operator>",
    RegexExpr(r"and"): "<operator>",
    RegexExpr(r"or"): "<operator>",
    RegexExpr(r"\&"): "<operator>",
    RegexExpr(r"\*|//|%"): "<operator>",
    RegexExpr(r"\*\*"): "<operator>",
    RegexExpr(r"\+|-"): "<operator>",
    RegexExpr(r"\^"): "<operator>",
    RegexExpr(r"\|"): "<operator>",
    RegexExpr(r"\+|-|~"): "<operator>",
    RegexExpr(r"\("): "'('",
    RegexExpr(r"\)"): "')'",
    # Other
    RuleExpr("stmt_block"): "':'",
    RuleExpr("condition"): "'('",
    RuleExpr("for_each_list"): "<expression>",
    RuleExpr("assignment_op"): "'='",
    RuleExpr("subscript"): "'['",
    RuleExpr("function_call_arguments"): "'('",
    RuleExpr("function_arguments"): "'('",
    # Misc symbols
    RegexExpr(r"\,"): "','",
    RegexExpr(r"\="): "'='",
    RegexExpr(r"\}"): "'}'",
    RegexExpr(r"\{"): "'{'",
}

parse_error_default_last_resort_exprs: Set[Union[RuleExpr, RegexExpr]] = {
    RuleExpr("single_line_stmt"),
    RuleExpr("comment"),
    RuleExpr("ws_"),
    RuleExpr("h_space"),
    RuleExpr("v_space"),
    RuleExpr("eof"),
    RuleExpr("eol"),
}

parse_error_default_just_indentation = False


def parse(string: str) -> Listing:
    """
    Parse a pseudocode listing into an abstract syntax tree
    (:py:class:`.Listing`).

    May raise a :py:exc:`.ParseError` or
    :py:exc:`~.ASTConstructionError` exception on failure.
    """
    parser = Parser(grammar)
    try:
        parse_tree = parser.parse(string)
    except ParseError as e:
        e.expr_explanations = parse_error_default_expr_explanations
        e.last_resort_exprs = parse_error_default_last_resort_exprs
        e.just_indentation = parse_error_default_just_indentation
        raise e

    ast = cast(Listing, ToAST().transform(parse_tree))

    infer_labels(string, ast)

    return ast
