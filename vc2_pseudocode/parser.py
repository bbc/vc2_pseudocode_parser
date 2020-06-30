"""
A parser for the pseudocode language which produces an Abstract Syntax Tree
(AST) representation of the underlying code.
"""

from typing import Optional, Mapping, Set, Union, cast

from peggie.parser import Parser, ParseError, RuleExpr, RegexExpr

from vc2_pseudocode.grammar import grammar
from vc2_pseudocode.ast import ToAST, Listing

import re


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
    RuleExpr("maybe_peren_expr"): "<expression>",
    # Optional whitespace
    RuleExpr("WS"): None,
    # Mandatory vertical whitespace
    RuleExpr("comment"): "<newline>",
    RuleExpr("V_SPACE"): "<newline>",
    RuleExpr("EOL"): "<newline>",
    RuleExpr("EOF"): "<newline>",
    # Mandatory horizontal whitespace
    RuleExpr("WS_"): "<space>",
    RuleExpr("H_SPACE"): "<space>",
    # Operators
    RegexExpr(re.compile(r"<<|>>", re.DOTALL)): "<operator>",
    RegexExpr(re.compile(r"==|!=|<=|>=|<|>", re.DOTALL)): "<operator>",
    RegexExpr(re.compile(r"not", re.DOTALL)): "<operator>",
    RegexExpr(re.compile(r"and", re.DOTALL)): "<operator>",
    RegexExpr(re.compile(r"or", re.DOTALL)): "<operator>",
    RegexExpr(re.compile(r"\&", re.DOTALL)): "<operator>",
    RegexExpr(re.compile(r"\*|//|%", re.DOTALL)): "<operator>",
    RegexExpr(re.compile(r"\+|-", re.DOTALL)): "<operator>",
    RegexExpr(re.compile(r"\^", re.DOTALL)): "<operator>",
    RegexExpr(re.compile(r"\|", re.DOTALL)): "<operator>",
    RegexExpr(re.compile(r"\+|-|!", re.DOTALL)): "<operator>",
    RegexExpr(re.compile(r"\(", re.DOTALL)): "'('",
    RegexExpr(re.compile(r"\)", re.DOTALL)): "')'",
    # Other
    RuleExpr("stmt_block"): "':'",
    RuleExpr("condition"): "'('",
    RuleExpr("for_each_list"): "<expression>",
    RuleExpr("assignment_op"): "'='",
    RuleExpr("subscript"): "'['",
    RuleExpr("function_call_arguments"): "'('",
    RuleExpr("function_arguments"): "'('",
    # Misc symbols
    RegexExpr(re.compile(r"\,", re.DOTALL)): "','",
    RegexExpr(re.compile(r"\=", re.DOTALL)): "'='",
    RegexExpr(re.compile(r"\}", re.DOTALL)): "'}'",
    RegexExpr(re.compile(r"\{", re.DOTALL)): "'{'",
}
parse_error_default_last_resort_exprs: Set[Union[RuleExpr, RegexExpr]] = {
    RuleExpr("single_line_stmt"),
    RuleExpr("comment"),
    RuleExpr("WS_"),
    RuleExpr("H_SPACE"),
    RuleExpr("V_SPACE"),
    RuleExpr("EOF"),
    RuleExpr("EOL"),
}
parse_error_default_just_indentation = False


class PseudocodeParseError(ParseError):
    def explain(
        self,
        expr_explanations: Mapping[
            Union[RuleExpr, RegexExpr], Optional[str]
        ] = parse_error_default_expr_explanations,
        last_resort_exprs: Set[
            Union[RuleExpr, RegexExpr]
        ] = parse_error_default_last_resort_exprs,
        just_indentation: bool = parse_error_default_just_indentation,
    ) -> str:
        return super().explain(expr_explanations, last_resort_exprs, just_indentation)


def parse(string: str) -> Listing:
    parser = Parser(grammar)
    try:
        parse_tree = parser.parse(string)
    except ParseError as e:
        raise PseudocodeParseError(e.line, e.column, e.snippet, e.expectations)

    transformer = ToAST()
    ast = cast(Listing, transformer.transform(parse_tree))

    return ast
