"""
These test simultaneously test the pseudocode grammar, parser and AST
transfomer.
"""

import pytest  # type: ignore

from typing import Any, List, Optional, Union, cast

from itertools import dropwhile

from textwrap import indent, dedent

from dataclasses import asdict

from vc2_pseudocode.parser import parse, PseudocodeParseError

from vc2_pseudocode.ast import (
    ASTNode,
    Expr,
    VariableExpr,
    Variable,
    Subscript,
    IfElseStmt,
    IfBranch,
    ElseBranch,
    BooleanExpr,
    FunctionCallStmt,
    FunctionCallExpr,
    ForEachStmt,
    ForStmt,
    WhileStmt,
    NumberExpr,
    AssignmentStmt,
    AssignmentOp,
    ReturnStmt,
    UnaryOp,
    UnaryExpr,
    BinaryOp,
    BinaryExpr,
    EmptyMapExpr,
    Comment,
)


@pytest.mark.parametrize(
    "string, exp_function_offsets",
    [
        # Empty string should fail
        ("", []),
        # Whitespace only should fail
        (" ", []),
        ("  ", []),
        (" \t \n \r", []),
        # Non-function nonsense should fail
        ("foobar", []),
        # One-liner function (no whitespace) should be OK
        ("foobar(): a()", [0]),
        # Whitespace around function should be OK
        (" foobar(): a() ", [1]),
        ("\n  \nfoobar(): a() \n \n ", [4]),
        # Multiple functions should be OK
        ("foobar(): a()\nfoobar(): a()", [0, 14]),
        # Empty lines' indentations should be ignored
        ("    \nfoobar(): a()\nfoobar(): a()", [5, 19]),
        ("foobar(): a()\n    \nfoobar(): a()", [0, 19]),
        ("foobar(): a()\nfoobar(): a()\n    ", [0, 14]),
        # Indent must be same for functions
        ("  foobar(): a()\n  foobar(): a()", [2, 18]),
        ("  foobar(): a()\n   foobar(): a()", []),
        ("   foobar(): a()\n  foobar(): a()", []),
    ],
)
def test_listing(string: str, exp_function_offsets: List[int]) -> None:
    if exp_function_offsets:
        ast = parse(string)
        assert ast.offset == 0
        assert [f.offset for f in ast.functions] == exp_function_offsets
    else:
        with pytest.raises(PseudocodeParseError):
            parse(string)


@pytest.mark.parametrize(
    "string, exp_name, exp_vars, exp_stmts",
    [
        # Empty string should fail
        ("", None, None, None),
        # Varying number of arguments
        ("foo(): a()", "foo", [], 1),
        ("foo(x): a()", "foo", [Variable(4, "x")], 1),
        ("foo(x, y): a()", "foo", [Variable(4, "x"), Variable(7, "y")], 1),
        # With trailing commas
        ("foo(x,): a()", "foo", [Variable(4, "x")], 1),
        ("foo(x, y,): a()", "foo", [Variable(4, "x"), Variable(7, "y")], 1),
        # With extra spacing
        ("foo ( x , ) : a()", "foo", [Variable(6, "x")], 1),
        ("foo ( x , y ) : a()", "foo", [Variable(6, "x"), Variable(10, "y")], 1),
        ("foo ( x , y , ) : a()", "foo", [Variable(6, "x"), Variable(10, "y")], 1),
        # Without brackets
        ("foo: a()", None, None, None),
        ("foo x: a()", None, None, None),
        # Indented statement block
        ("foo():\n  bar()", "foo", [], 1),
        ("foo():\n  bar()\n  baz()", "foo", [], 2),
        # Badly indented block
        ("foo():\nbar()", None, None, None),
        ("foo():\n  bar()\n   baz()", None, None, None),
    ],
)
def test_function(
    string: str,
    exp_name: Optional[str],
    exp_vars: Optional[List[Variable]],
    exp_stmts: Optional[int],
) -> None:
    if exp_name is not None and exp_vars is not None and exp_stmts is not None:
        ast = parse(string)
        assert len(ast.functions) == 1

        function = ast.functions[0]
        assert function.name == exp_name
        assert function.arguments == exp_vars
        assert len(function.body) == exp_stmts
    else:
        with pytest.raises(PseudocodeParseError):
            parse(string)


def equal_ignoring_offsets(a: Any, b: Any) -> bool:
    def remove_offset(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: remove_offset(v) for k, v in x.items() if k != "offset"}
        elif isinstance(x, list):
            return [remove_offset(v) for v in x]
        elif isinstance(x, tuple):
            return tuple(remove_offset(v) for v in x)
        else:
            return x

    return bool(remove_offset(asdict(a)) == remove_offset(asdict(b)))


@pytest.mark.parametrize(
    "a, b, exp_equal",
    [
        # Actually equal
        (Variable(0, "foo"), Variable(0, "foo"), True),
        # Equal appart from offset
        (Variable(0, "foo"), Variable(1, "foo"), True),
        # Different values
        (Variable(0, "foo"), Variable(0, "bar"), False),
        (Variable(0, "foo"), Variable(1, "bar"), False),
        # Different types
        (Variable(0, "foo"), VariableExpr(Variable(0, "foo")), False),
        # Nesting
        (VariableExpr(Variable(0, "foo")), VariableExpr(Variable(0, "foo")), True),
        (VariableExpr(Variable(0, "foo")), VariableExpr(Variable(1, "foo")), True),
        (VariableExpr(Variable(0, "foo")), VariableExpr(Variable(0, "bar")), False),
        (VariableExpr(Variable(0, "foo")), VariableExpr(Variable(1, "bar")), False),
    ],
)
def test_equal_ignoring_offsets(a: ASTNode, b: ASTNode, exp_equal: bool) -> None:
    assert equal_ignoring_offsets(a, b) is exp_equal


def fcs(name: str) -> FunctionCallStmt:
    return FunctionCallStmt(FunctionCallExpr(0, name, []))


@pytest.mark.parametrize(
    "string, exp_if_else_stmt",
    [
        # Simple if
        ("if(True): a()", IfElseStmt([IfBranch(0, BooleanExpr(0, True), [fcs("a")])]),),
        # If-else
        (
            "if(True): a()\nelse: b()",
            IfElseStmt(
                [IfBranch(0, BooleanExpr(0, True), [fcs("a")])],
                ElseBranch(0, [fcs("b")]),
            ),
        ),
        # Else-if
        (
            "if(True): a()\nelse if(False): b()",
            IfElseStmt(
                [
                    IfBranch(0, BooleanExpr(0, True), [fcs("a")]),
                    IfBranch(0, BooleanExpr(0, False), [fcs("b")]),
                ],
            ),
        ),
        # Many else-ifs
        (
            "if(True): a()\nelse if(False): b()\nelse if(True): c()",
            IfElseStmt(
                [
                    IfBranch(0, BooleanExpr(0, True), [fcs("a")]),
                    IfBranch(0, BooleanExpr(0, False), [fcs("b")]),
                    IfBranch(0, BooleanExpr(0, True), [fcs("c")]),
                ],
            ),
        ),
        # If-else-if-else
        (
            "if(True): a()\nelse if(False): b()\nelse: c()",
            IfElseStmt(
                [
                    IfBranch(0, BooleanExpr(0, True), [fcs("a")]),
                    IfBranch(0, BooleanExpr(0, False), [fcs("b")]),
                ],
                ElseBranch(0, [fcs("c")]),
            ),
        ),
        # Extra whitespace
        (
            "if ( True ) : a()\nelse  if ( False ): b()\nelse : c()",
            IfElseStmt(
                [
                    IfBranch(0, BooleanExpr(0, True), [fcs("a")]),
                    IfBranch(0, BooleanExpr(0, False), [fcs("b")]),
                ],
                ElseBranch(0, [fcs("c")]),
            ),
        ),
        # Missing brackets
        ("if False: a()", None),
        ("if (False: a()", None),
        ("if False): a()", None),
        ("if (True): a()\nelse if False: b()", None),
        ("if (True): a()\nelse if (False: b()", None),
        ("if (True): a()\nelse if False): b()", None),
        # Missing space between else and if
        ("if (True): a()\nelseif (False): b()", None),
        # Else before else if
        ("if (True): a()\nelse: b()\nelse if (False): c()", None),
        # Else indented differently
        ("if (True): a()\n  else: b()\n", None),
        # Else-if indented differently
        ("if (True): a()\n  else if (True): b()\n", None),
        ("if (True): a()\nelse if (True): b()\n  else if (True): c()\n", None),
        # Else in if-else-if indented differently
        ("if (True): a()\nelse if (True): b()\n  else: c()\n", None),
    ],
)
def test_if_else_block(string: str, exp_if_else_stmt: Optional[IfElseStmt]) -> None:
    string = "foo():\n{}".format(indent(dedent(string), "    "))

    if exp_if_else_stmt is not None:
        ast = parse(string)
        (function,) = ast.functions
        (if_else_stmt,) = function.body
        assert isinstance(if_else_stmt, IfElseStmt)

        assert if_else_stmt.offset == 11
        assert equal_ignoring_offsets(if_else_stmt, exp_if_else_stmt)

        i = if_else_stmt.if_branches[0].offset
        assert string[i : i + 2] == "if"

        for if_branch in if_else_stmt.if_branches[1:]:
            i = if_branch.offset
            assert string[i : i + 4] == "else"
            assert string[i : i + 8].rstrip("( ").split() == "else if".split()

        if if_else_stmt.else_branch is not None:
            i = if_else_stmt.else_branch.offset
            assert string[i : i + 4] == "else"
    else:
        with pytest.raises(PseudocodeParseError):
            parse(string)


@pytest.mark.parametrize(
    "string, exp_variable, exp_values",
    [
        # Single value
        ("for each foo in a: a()", "foo", ["a"]),
        # Multiple values
        ("for each foo in a, b, c: a()", "foo", ["a", "b", "c"]),
        # Spacing
        ("for   each   foo   in a , b , c : a()", "foo", ["a", "b", "c"]),
        # No space between for and each
        ("foreach foo in a, b, c: a()", None, None),
        # No space between each and variable
        ("for eachfoo in a, b, c: a()", None, None),
        # No space between variable and in
        ("for each fooin a, b, c: a()", None, None),
        # No space between in and value
        ("for each foo ina, b, c: a()", None, None),
        # Trailing comma
        ("for each foo in a, b, c, : a()", None, None),
        # No values
        ("for each foo in : a()", None, None),
    ],
)
def test_for_each_stmt(
    string: str, exp_variable: str, exp_values: Optional[List[str]]
) -> None:
    string = "foo():\n{}".format(indent(dedent(string), "    "))

    if exp_variable is not None and exp_variable is not None:
        ast = parse(string)
        (function,) = ast.functions
        (for_each_stmt,) = function.body
        assert isinstance(for_each_stmt, ForEachStmt)
        assert for_each_stmt.offset == 11

        assert for_each_stmt.variable.name == exp_variable

        assert [
            cast(Variable, cast(VariableExpr, v).variable).name
            for v in for_each_stmt.values
        ] == exp_values

        assert len(for_each_stmt.body) == 1
        assert equal_ignoring_offsets(for_each_stmt.body[0], fcs("a"))
    else:
        with pytest.raises(PseudocodeParseError):
            parse(string)


@pytest.mark.parametrize(
    "string, exp_variable, exp_start, exp_end",
    [
        # Simple case
        ("for x = 1 to 3: a()", "x", 1, 3),
        # Minimal spacing
        ("for x=1 to 3: a()", "x", 1, 3),
        # Extra spacing
        ("for   x = 1  to  3 : a()", "x", 1, 3),
        # Missing space between for and variable
        ("forx = 1 to 3: a()", None, None, None),
        # Missing space between start and to
        ("for x = 1to 3: a()", None, None, None),
        # Missing space between to and end
        ("for x = 1 to3: a()", None, None, None),
    ],
)
def test_for_stmt(
    string: str, exp_variable: str, exp_start: Optional[int], exp_end: Optional[int]
) -> None:
    string = "foo():\n{}".format(indent(dedent(string), "    "))

    if exp_variable is not None and exp_start is not None and exp_end is not None:
        ast = parse(string)
        (function,) = ast.functions
        (for_stmt,) = function.body
        assert isinstance(for_stmt, ForStmt)
        assert for_stmt.offset == 11

        assert for_stmt.variable.name == exp_variable

        assert cast(NumberExpr, for_stmt.start).value == exp_start
        assert cast(NumberExpr, for_stmt.end).value == exp_end

        assert len(for_stmt.body) == 1
        assert equal_ignoring_offsets(for_stmt.body[0], fcs("a"))
    else:
        with pytest.raises(PseudocodeParseError):
            parse(string)


@pytest.mark.parametrize(
    "string, exp_condition",
    [
        # Simple case
        ("while (True): a()", BooleanExpr(0, True)),
        # Minimal space
        ("while(True): a()", BooleanExpr(0, True)),
        # Extra space
        ("while  ( True ) : a()", BooleanExpr(0, True)),
        # Missing expression
        ("while (): a()", None),
        # Missing brackets
        ("while True: a()", None),
        ("while (True: a()", None),
        ("while True): a()", None),
    ],
)
def test_while_stmt(string: str, exp_condition: Optional[Expr]) -> None:
    string = "foo():\n{}".format(indent(dedent(string), "    "))

    if exp_condition is not None:
        ast = parse(string)
        (function,) = ast.functions
        (for_stmt,) = function.body
        assert isinstance(for_stmt, WhileStmt)
        assert for_stmt.offset == 11

        assert equal_ignoring_offsets(for_stmt.condition, exp_condition)

        assert len(for_stmt.body) == 1
        assert equal_ignoring_offsets(for_stmt.body[0], fcs("a"))
    else:
        with pytest.raises(PseudocodeParseError):
            parse(string)


def test_function_call_stmt() -> None:
    # NB function call expressions tested separately
    string = "foo(n): foo(123)"

    ast = parse(string)
    (function,) = ast.functions
    (call_stmt,) = function.body
    assert isinstance(call_stmt, FunctionCallStmt)
    assert call_stmt.offset == 8


@pytest.mark.parametrize(
    "string, exp_stmt",
    [
        # Normal usage
        ("return True", ReturnStmt(0, BooleanExpr(0, True))),
        # Extra space
        ("return   True", ReturnStmt(0, BooleanExpr(0, True))),
        # Too little space
        ("returnTrue", None),
        # No value
        ("return", None),
    ],
)
def test_return_stmt(string: str, exp_stmt: Optional[ReturnStmt]) -> None:
    # NB function call expressions tested separately
    string = "foo(): {}".format(string)

    if exp_stmt is not None:
        ast = parse(string)
        (function,) = ast.functions
        (return_stmt,) = function.body
        assert return_stmt.offset == 7
        assert equal_ignoring_offsets(return_stmt, exp_stmt)
    else:
        with pytest.raises(PseudocodeParseError):
            parse(string)


@pytest.mark.parametrize(
    "lhs_string, exp_lhs",
    [
        ("x", Variable(0, "x")),
        ("x[y]", Subscript(Variable(0, "x"), VariableExpr(Variable(0, "y")))),
    ],
)
@pytest.mark.parametrize("spacing", ["", " "])
@pytest.mark.parametrize("op", AssignmentOp)
def test_assignment_statement(
    lhs_string: str, exp_lhs: Union[Variable, Subscript], spacing: str, op: AssignmentOp
) -> None:
    string = "foo(): {}{}{}{}b".format(lhs_string, spacing, op.value, spacing)
    ast = parse(string)
    (function,) = ast.functions
    (assignment_stmt,) = function.body
    assert isinstance(assignment_stmt, AssignmentStmt)
    assert assignment_stmt.offset == 7

    assert equal_ignoring_offsets(assignment_stmt.variable, exp_lhs)
    assert assignment_stmt.op == op
    assert equal_ignoring_offsets(assignment_stmt.value, VariableExpr(Variable(0, "b")))


def parse_expr(string: str) -> Expr:
    ast = parse("foo(): return {}".format(string))
    (function,) = ast.functions
    return_stmt = cast(ReturnStmt, function.body[0])
    return return_stmt.value


@pytest.mark.parametrize("spacing", ["", " "])
@pytest.mark.parametrize("op", UnaryOp)
def test_unary_expr(spacing: str, op: UnaryOp) -> None:
    expr = parse_expr("{}{}foo".format(op.value, spacing))
    assert isinstance(expr, UnaryExpr)

    assert equal_ignoring_offsets(
        expr, UnaryExpr(0, op, VariableExpr(Variable(0, "foo"))),
    )


@pytest.mark.parametrize("op", UnaryOp)
def test_unary_expr_right_associativity(op: UnaryOp) -> None:
    expr = parse_expr("{}{}foo".format(op.value, op.value))
    assert isinstance(expr, UnaryExpr)

    assert expr.offset == expr.value.offset - 1
    assert equal_ignoring_offsets(
        expr, UnaryExpr(0, op, UnaryExpr(0, op, VariableExpr(Variable(0, "foo"))))
    )


@pytest.mark.parametrize("spacing", ["", " "])
@pytest.mark.parametrize("op", BinaryOp)
def test_binary_expr(spacing: str, op: BinaryOp) -> None:
    expr = parse_expr("foo{}{}{}bar".format(spacing, op.value, spacing))
    assert isinstance(expr, BinaryExpr)

    assert equal_ignoring_offsets(
        expr,
        BinaryExpr(
            VariableExpr(Variable(0, "foo")), op, VariableExpr(Variable(0, "bar")),
        ),
    )


@pytest.mark.parametrize("op", BinaryOp)
def test_binary_expr_left_associativity(op: BinaryOp) -> None:
    expr = parse_expr("foo {} bar {} baz".format(op.value, op.value))
    assert isinstance(expr, BinaryExpr)
    assert equal_ignoring_offsets(
        expr,
        BinaryExpr(
            BinaryExpr(
                VariableExpr(Variable(0, "foo")), op, VariableExpr(Variable(0, "bar")),
            ),
            op,
            VariableExpr(Variable(0, "baz")),
        ),
    )


# Ordered from highest to lowest precedence
OPERATOR_PRECEDENCE_TABLE = [
    {"*", "//"},
    {"+", "-"},
    {"<<", ">>"},
    {"&"},
    {"^"},
    {"|"},
    {"<=", ">=", "<", ">"},
    {"==", "!="},
]


@pytest.mark.parametrize(
    "target_op", [op for ops in OPERATOR_PRECEDENCE_TABLE for op in ops]
)
def test_binary_operator_precedence(target_op: str) -> None:
    same_or_lower = dropwhile(
        lambda row: target_op not in row, OPERATOR_PRECEDENCE_TABLE
    )
    same_precedence = next(same_or_lower)
    lower_precedence = {op for ops in same_or_lower for op in ops}

    # Check when combined with same precidence, we're left associative
    for other_op in same_precedence:
        for op1, op2 in [(target_op, other_op), (other_op, target_op)]:
            expr = parse_expr("a {} b {} c".format(op1, op2))
            assert equal_ignoring_offsets(
                expr,
                BinaryExpr(
                    BinaryExpr(
                        VariableExpr(Variable(0, "a")),
                        BinaryOp(op1),
                        VariableExpr(Variable(0, "b")),
                    ),
                    BinaryOp(op2),
                    VariableExpr(Variable(0, "c")),
                ),
            )

    for other_op in lower_precedence:
        # Check when combined with lower precidence, this operator has higher
        # precidence
        expr = parse_expr("a {} b {} c".format(target_op, other_op))
        assert equal_ignoring_offsets(
            expr,
            BinaryExpr(
                BinaryExpr(
                    VariableExpr(Variable(0, "a")),
                    BinaryOp(target_op),
                    VariableExpr(Variable(0, "b")),
                ),
                BinaryOp(other_op),
                VariableExpr(Variable(0, "c")),
            ),
        )
        expr = parse_expr("a {} b {} c".format(other_op, target_op))
        assert equal_ignoring_offsets(
            expr,
            BinaryExpr(
                VariableExpr(Variable(0, "a")),
                BinaryOp(other_op),
                BinaryExpr(
                    VariableExpr(Variable(0, "b")),
                    BinaryOp(target_op),
                    VariableExpr(Variable(0, "c")),
                ),
            ),
        )
        # Except when perentheses are used
        expr = parse_expr("(a {} b) {} c".format(other_op, target_op))
        assert equal_ignoring_offsets(
            expr,
            BinaryExpr(
                BinaryExpr(
                    VariableExpr(Variable(0, "a")),
                    BinaryOp(other_op),
                    VariableExpr(Variable(0, "b")),
                ),
                BinaryOp(target_op),
                VariableExpr(Variable(0, "c")),
            ),
        )
        expr = parse_expr("a {} (b {} c)".format(target_op, other_op))
        assert equal_ignoring_offsets(
            expr,
            BinaryExpr(
                VariableExpr(Variable(0, "a")),
                BinaryOp(target_op),
                BinaryExpr(
                    VariableExpr(Variable(0, "b")),
                    BinaryOp(other_op),
                    VariableExpr(Variable(0, "c")),
                ),
            ),
        )

    # Check that lower precidence than unary operators
    for unary_op in UnaryOp:
        expr = parse_expr("{} a {} b".format(unary_op.value, target_op))
        assert equal_ignoring_offsets(
            expr,
            BinaryExpr(
                UnaryExpr(0, unary_op, VariableExpr(Variable(0, "a"))),
                BinaryOp(target_op),
                VariableExpr(Variable(0, "b")),
            ),
        )


@pytest.mark.parametrize(
    "string, exp_name, exp_arguments",
    [
        # No arguments
        ("foo()", "foo", []),
        # Single argument
        ("foo(True)", "foo", [BooleanExpr(0, True)]),
        ("foo(True,)", "foo", [BooleanExpr(0, True)]),
        # Multiple arguments
        ("foo(True,False)", "foo", [BooleanExpr(0, True), BooleanExpr(0, False)]),
        ("foo(True, False)", "foo", [BooleanExpr(0, True), BooleanExpr(0, False)]),
        ("foo(True, False, )", "foo", [BooleanExpr(0, True), BooleanExpr(0, False)]),
        # Spacing
        (
            "foo ( True , False , )",
            "foo",
            [BooleanExpr(0, True), BooleanExpr(0, False)],
        ),
        # Too many commas
        ("foo(,)", None, None),
        ("foo(,,)", None, None),
        ("foo(True,,)", None, None),
        ("foo(True,False,,)", None, None),
        ("foo(True,,False)", None, None),
        # Missing brackets
        ("foo)", None, None),
        ("foo(", None, None),
    ],
)
def test_function_call_expr(
    string: str, exp_name: Optional[str], exp_arguments: Optional[List[Expr]]
) -> None:
    if exp_name is not None and exp_arguments is not None:
        expr = parse_expr(string)
        assert equal_ignoring_offsets(
            expr, FunctionCallExpr(0, exp_name, exp_arguments)
        )
    else:
        with pytest.raises(PseudocodeParseError):
            parse_expr(string)


@pytest.mark.parametrize(
    "string, exp_variable_expr",
    [
        # Simple variable
        ("foo", VariableExpr(Variable(0, "foo"))),
        # Subscripted
        (
            "foo[True]",
            VariableExpr(Subscript(Variable(0, "foo"), BooleanExpr(0, True))),
        ),
        # Nested subscripts
        (
            "foo[1][2][3]",
            VariableExpr(
                Subscript(
                    Subscript(
                        Subscript(Variable(0, "foo"), NumberExpr(0, 1),),
                        NumberExpr(0, 2),
                    ),
                    NumberExpr(0, 3),
                ),
            ),
        ),
        # Reserved word
        ("return", None),
        # Unmatched subscript brackets
        ("foo[", None),
        ("foo]", None),
        # Too many brackets
        ("foo[[1]]", None),
        # Empty subscript
        ("foo[]", None),
    ],
)
def test_variable_expr(string: str, exp_variable_expr: Optional[VariableExpr]) -> None:
    if exp_variable_expr is not None:
        expr = parse_expr(string)
        assert equal_ignoring_offsets(expr, exp_variable_expr)
    else:
        with pytest.raises(PseudocodeParseError):
            parse_expr(string)


@pytest.mark.parametrize(
    "string, exp_success",
    [
        # Simple case
        ("{}", True),
        # With space
        ("{  }", True),
        # Unmatched
        ("{", False),
        ("}", False),
        # Something inside
        ("{what}", False),
    ],
)
def test_empty_map_expr(string: str, exp_success: bool) -> None:
    if exp_success:
        expr = parse_expr(string)
        assert expr == EmptyMapExpr(14)
    else:
        with pytest.raises(PseudocodeParseError):
            parse_expr(string)


@pytest.mark.parametrize(
    "string, exp_value",
    [
        # Valid booleans
        ("True", True),
        ("False", False),
        # Wrong case
        ("true", None),
        ("false", None),
    ],
)
def test_boolean_expr(string: str, exp_value: Optional[bool]) -> None:
    if exp_value is not None:
        expr = parse_expr(string)
        assert expr == BooleanExpr(14, exp_value)
    else:
        with pytest.raises(PseudocodeParseError):
            parse_expr(string)


@pytest.mark.parametrize(
    "string, exp_value, exp_display_base",
    [
        # Decimal
        ("0", 0, 10),
        ("000", 0, 10),
        ("1", 1, 10),
        ("1234567890", 1234567890, 10),
        ("0123", 123, 10),
        # Hex
        ("0x0", 0x0, 16),
        ("0x1", 0x1, 16),
        ("0x1234567890ABCDEF", 0x1234567890ABCDEF, 16),
        ("0X1234567890ABCDEF", 0x1234567890ABCDEF, 16),
        ("0X1234567890abcdef", 0x1234567890ABCDEF, 16),
        ("0x1234567890abcdef", 0x1234567890ABCDEF, 16),
        # Binary
        ("0b0", 0b0, 2),
        ("0b1", 0b1, 2),
        ("0b10100", 0b10100, 2),
        ("0B10100", 0b10100, 2),
        # Non-numeric char in decimal
        ("0123b", None, None),
        # Too-high digit in binary
        ("0b123", None, None),
        # Invalid char in hex
        ("0x10g", None, None),
        # Too many leading zeros
        ("00b100", None, None),
        ("00x100", None, None),
        # Space
        ("1 2 3", None, None),
        ("0x 100", None, None),
        ("0 x100", None, None),
        ("0 x 100", None, None),
        ("0b 100", None, None),
        ("0 b100", None, None),
        ("0 b 100", None, None),
    ],
)
def test_number_expr(
    string: str, exp_value: Optional[int], exp_display_base: Optional[int]
) -> None:
    if exp_value is not None and exp_display_base is not None:
        expr = parse_expr(string)
        assert expr == NumberExpr(14, exp_value, exp_display_base)
    else:
        with pytest.raises(PseudocodeParseError):
            parse_expr(string)


@pytest.mark.parametrize(
    "string, exp_comments",
    [
        # One-liner
        ("foo(): return 1 # Hello there", [Comment(16, "# Hello there")]),
        # One-liner no space
        ("foo(): return 1# Hello there", [Comment(15, "# Hello there")]),
        # Leading comment
        ("# Hi\nfoo(): return 1", [Comment(0, "# Hi")]),
        ("# Hi\r\nfoo(): return 1", [Comment(0, "# Hi")]),
        ("# Hi\n\rfoo(): return 1", [Comment(0, "# Hi")]),
        ("# Hi\nfoo(): return 1", [Comment(0, "# Hi")]),
        # Indented leading comment
        ("  # Hi\nfoo(): return 1", [Comment(2, "# Hi")]),
        ("  # Hi\r\nfoo(): return 1", [Comment(2, "# Hi")]),
        ("  # Hi\n\rfoo(): return 1", [Comment(2, "# Hi")]),
        ("  # Hi\nfoo(): return 1", [Comment(2, "# Hi")]),
        # Trailing comment
        ("foo(): return 1\n# Hi", [Comment(16, "# Hi")]),
        # Comment within block
        ("foo():\n  # Hi\n  return 1", [Comment(9, "# Hi")]),
        # Empty comment
        ("foo(): return 1 #", [Comment(16, "#")]),
        ("foo(): return 1 #\n", [Comment(16, "#")]),
        # Multiple comments
        (
            "# One\nfoo():\n  # Two\n  return 1\n# Three",
            [Comment(0, "# One"), Comment(15, "# Two"), Comment(32, "# Three")],
        ),
    ],
)
def test_comment_capture(string: str, exp_comments: Optional[List[Comment]]) -> None:
    if exp_comments is not None:
        listing = parse(string)
        assert listing.comments == exp_comments
    else:
        with pytest.raises(PseudocodeParseError):
            parse(string)


@pytest.mark.parametrize(
    "string, exp_error",
    [
        # Empty file
        (
            "",
            """

                    ^
                Expected <function-definition>
            """,
        ),
        # Function missing arguments
        (
            "foo",
            """
                    foo
                       ^
                Expected '('
            """,
        ),
        # Function missing open peren
        (
            "foo a",
            """
                    foo a
                        ^
                Expected '('
            """,
        ),
        # Function missing close peren
        (
            "foo(a",
            """
                    foo(a
                         ^
                Expected ')' or ','
            """,
        ),
        (
            "foo(a,",
            """
                    foo(a,
                          ^
                Expected ')' or <identifier>
            """,
        ),
        # Function missing colon
        (
            "foo(a)",
            """
                    foo(a)
                          ^
                Expected ':'
            """,
        ),
        # Function missing statement
        (
            "foo(a):",
            """
                    foo(a):
                           ^
                Expected <statement>
            """,
        ),
        # Non-single-line statement used
        (
            "foo(a): if (True): a()",
            """
                    foo(a): if (True): a()
                            ^
                Expected <newline> or <single-line-statement>
            """,
        ),
        # Block not indented
        (
            """
                foo(a):
                not_indented()
            """,
            """
                    not_indented()
                    ^
                Expected <statement> (with indentation > 0)
            """,
        ),
        # Block inconsistently indented
        (
            """
                foo(a):
                    indented()
                     badly_indented()
            """,
            """
                         badly_indented()
                         ^
                Expected <function-definition> (with indentation = 0) or <statement> (with indentation = 4)
            """,  # noqa: E501
        ),
        # Bad indent before expected statement
        (
            """
                foo(a):
                    indented()
                     badly_indented_and_not_stmt
            """,
            """
                         badly_indented_and_not_stmt
                         ^
                Expected <function-definition> (with indentation = 0) or <statement> (with indentation = 4)
            """,  # noqa: E501
        ),
        # If statement with no body
        (
            """
                foo(a):
                    if (True):
                    return 0
            """,
            """
                        return 0
                        ^
                Expected <statement> (with indentation > 4)
            """,
        ),
        # If with no peren
        (
            """
                foo(a):
                    if True:
                        return 0
            """,
            """
                        if True:
                           ^
                Expected '('
            """,
        ),
        # If with no close peren
        (
            """
                foo(a):
                    if (True:
                        return 0
            """,
            """
                        if (True:
                                ^
                Expected ')' or <operator>
            """,
        ),
        # If with no close peren (and lots of potential expression parses)
        (
            """
                foo(a):
                    if (a:
                        return 0
            """,
            """
                        if (a:
                             ^
                Expected '(' or ')' or '[' or <operator>
            """,
        ),
        # If with no colon
        (
            """
                foo(a):
                    if (a)
                        return 0
            """,
            """
                        if (a)
                              ^
                Expected ':'
            """,
        ),
        # If with no condition
        (
            """
                foo(a):
                    if
                        return 0
            """,
            """
                        if
                          ^
                Expected '('
            """,
        ),
        # Else if with no space
        (
            """
                foo(a):
                    if (True):
                        return 0
                    elseif (True):
                        return 1
            """,
            """
                        elseif (True):
                            ^
                Expected ':'
            """,
        ),
        # Else if with no condition
        (
            """
                foo(a):
                    if (True):
                        return 0
                    else if:
                        return 1
            """,
            """
                        else if:
                               ^
                Expected '('
            """,
        ),
        # Else with condition
        (
            """
                foo(a):
                    if (True):
                        return 0
                    else (True):
                        return 1
            """,
            """
                        else (True):
                             ^
                Expected ':' or 'if'
            """,
        ),
        # Else before else-if
        (
            """
                foo(a):
                    if (True):
                        return 0
                    else:
                        return 1
                    else if (True):
                        return 2
            """,
            """
                        else if (True):
                        ^
                Expected <function-definition> (with indentation = 0) or <statement> (optionally with indentation = 8)
            """,  # noqa: E501
        ),
        # For-each without space
        (
            """
                foo(a):
                    foreach x in a, b, c:
                        return x
            """,
            """
                        foreach x in a, b, c:
                           ^
                Expected <space>
            """,
        ),
        # For-each without name
        (
            """
                foo(a):
                    for each in a, b, c:
                        return x
            """,
            """
                        for each in a, b, c:
                                 ^
                Expected <identifier>
            """,
        ),
        # For-each without values
        (
            """
                foo(a):
                    for each x in :
                        return x
            """,
            """
                        for each x in :
                                      ^
                Expected <expression>
            """,
        ),
        # For-each with trailing comma
        (
            """
                foo(a):
                    for each x in a, b,:
                        return x
            """,
            """
                        for each x in a, b,:
                                           ^
                Expected <expression>
            """,
        ),
        # For with missing identifier
        (
            """
                foo(a):
                    for = 1 to 3:
                        return 0
            """,
            """
                        for = 1 to 3:
                            ^
                Expected 'each' or <identifier>
            """,
        ),
        # For with invalid identifier
        (
            """
                foo(a):
                    for 100 = 1 to 3:
                        return 0
            """,
            """
                        for 100 = 1 to 3:
                            ^
                Expected 'each' or <identifier>
            """,
        ),
        (
            """
                foo(a):
                    for a[1] = 1 to 3:
                        return a
            """,
            """
                        for a[1] = 1 to 3:
                             ^
                Expected '='
            """,
        ),
        # For with 'in' not =
        (
            """
                foo(a):
                    for a in 1 to 3:
                        return a
            """,
            """
                        for a in 1 to 3:
                              ^
                Expected '='
            """,
        ),
        # For with missing start
        (
            """
                foo(a):
                    for a = to 3:
                        return a
            """,
            """
                        for a = to 3:
                                ^
                Expected <expression>
            """,
        ),
        # For with missing end
        (
            """
                foo(a):
                    for a = 1 to :
                        return a
            """,
            """
                        for a = 1 to :
                                     ^
                Expected <expression>
            """,
        ),
        # For with missing spaces
        (
            """
                foo(a):
                    for a=1to3:
                        return a
            """,
            """
                        for a=1to3:
                               ^
                Expected <operator>
            """,
        ),
        (
            """
                foo(a):
                    for a=1 to3:
                        return a
            """,
            """
                        for a=1 to3:
                                  ^
                Expected <space>
            """,
        ),
        # While with missing brackets
        (
            """
                foo(a):
                    while True:
                        return 0
            """,
            """
                        while True:
                              ^
                Expected '('
            """,
        ),
        # While with missing close brackets
        (
            """
                foo(a):
                    while (True:
                        return 0
            """,
            """
                        while (True:
                                   ^
                Expected ')' or <operator>
            """,
        ),
        # While with empty expression
        (
            """
                foo(a):
                    while ():
                        return 0
            """,
            """
                        while ():
                               ^
                Expected <expression>
            """,
        ),
        # Return with no expression
        (
            """
                foo(a):
                    return # Nothing!
            """,
            """
                        return # Nothing!
                               ^
                Expected <expression>
            """,
        ),
        # Return with non expression
        (
            """
                foo(a):
                    return return 1
            """,
            """
                        return return 1
                               ^
                Expected <expression>
            """,
        ),
        # Assigning to non-variable
        (
            """
                foo(a):
                    a() = 123
            """,
            """
                        a() = 123
                            ^
                Expected <newline>
            """,
        ),
        # Assigning a non expression
        (
            """
                foo(a):
                    a = return 1
            """,
            """
                        a = return 1
                            ^
                Expected <expression>
            """,
        ),
        # Unmatched brackets in expression
        (
            """
                foo(a):
                    return (1 + (2 + 3)
            """,
            """
                        return (1 + (2 + 3)
                                           ^
                Expected ')' or <operator>
            """,
        ),
        # Missing half of empty map
        (
            """
                foo(a):
                    return {
            """,
            """
                        return {
                                ^
                Expected '}'
            """,
        ),
        # Malformed decimal
        (
            """
                foo(a):
                    return 123f
            """,
            """
                        return 123f
                                  ^
                Expected <operator>
            """,
        ),
        # Malformed hex
        (
            """
                foo(a):
                    return 0x123fg
            """,
            """
                        return 0x123fg
                                     ^
                Expected <operator>
            """,
        ),
        # Malformed bin
        (
            """
                foo(a):
                    return 0b102
            """,
            """
                        return 0b102
                                   ^
                Expected <operator>
            """,
        ),
        # Number prefix on own...
        (
            """
                foo(a):
                    return 0x
            """,
            """
                        return 0x
                                ^
                Expected <operator>
            """,
        ),
        # Reserved words
        (
            """
                foo(a):
                    return if
            """,
            """
                        return if
                               ^
                Expected <expression>
            """,
        ),
    ],
)
def test_parse_error_messages(string: str, exp_error: str) -> None:
    with pytest.raises(PseudocodeParseError) as exc_info:
        parse(dedent(string))
    exc = exc_info.value
    message = "\n".join(map(str.rstrip, exc.explain().splitlines()[1:]))
    assert message == dedent(exp_error[1:]).rstrip()
