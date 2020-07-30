"""
These test simultaneously test the pseudocode grammar, parser and AST
transfomer.
"""

import pytest

from typing import Any, List, Optional, Union, cast, Sequence, Type

from textwrap import indent, dedent

from dataclasses import asdict, fields

import pseudocode_samples

from vc2_pseudocode_parser.parser import (
    parse,
    ParseError,
    BinaryOp,
    UnaryOp,
    AssignmentOp,
    OPERATOR_PRECEDENCE_TABLE,
    OPERATOR_ASSOCIATIVITY_TABLE,
    Associativity,
    ASTNode,
    Expr,
    VariableExpr,
    Variable,
    LabelExpr,
    Label,
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
    ReturnStmt,
    UnaryExpr,
    BinaryExpr,
    EmptyMapExpr,
    EmptyLine,
    Comment,
    EOL,
    ASTConstructionError,
    LabelUsedAsVariableNameError,
    CannotSubscriptLabelError,
)


@pytest.mark.parametrize(
    "string, exp_leading_empty_line_offsets, exp_function_offsets",
    [
        # Empty string should fail
        ("", [], []),
        # Whitespace only should fail
        (" ", [], []),
        ("  ", [], []),
        (" \t \n \r", [], []),
        # Non-function nonsense should fail
        ("foobar", [], []),
        # One-liner function (no whitespace) should be OK
        ("foobar(): a()", [], [0]),
        # Whitespace around function should be OK
        (" foobar(): a() ", [], [1]),
        ("\n  \nfoobar(): a() \n \n ", [0, 1], [4]),
        # Multiple functions should be OK
        ("foobar(): a()\nfoobar(): a()", [], [0, 14]),
        # Empty lines' indentations should be ignored
        ("    \nfoobar(): a()\nfoobar(): a()", [0], [5, 19]),
        ("foobar(): a()\n    \nfoobar(): a()", [], [0, 19]),
        ("foobar(): a()\nfoobar(): a()\n    ", [], [0, 14]),
        # Indent must be same for functions
        ("  foobar(): a()\n  foobar(): a()", [], [2, 18]),
        ("  foobar(): a()\n   foobar(): a()", [], []),
        ("   foobar(): a()\n  foobar(): a()", [], []),
    ],
)
def test_listing(
    string: str,
    exp_leading_empty_line_offsets: List[int],
    exp_function_offsets: List[int],
) -> None:
    if exp_function_offsets:
        ast = parse(string)
        assert [f.offset for f in ast.functions] == exp_function_offsets
        assert [
            l.offset for l in ast.leading_empty_lines
        ] == exp_leading_empty_line_offsets
    else:
        with pytest.raises(ParseError):
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
        with pytest.raises(ParseError):
            parse(string)


def equal_ignoring_offsets(a: Any, b: Any) -> bool:
    def remove_offset(x: Any) -> Any:
        if isinstance(x, dict):
            return {
                k: remove_offset(v)
                for k, v in x.items()
                if k != "offset" and k != "offset_end"
            }
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


def equal_ignoring_offsets_and_eol(a: Any, b: Any) -> bool:
    def remove_offset(x: Any) -> Any:
        if isinstance(x, dict):
            return {
                k: remove_offset(v)
                for k, v in x.items()
                if (k != "offset" and k != "offset_end" and k != "eol")
            }
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
        # Equal appart from eol
        (
            ReturnStmt(0, NumberExpr(0, 3, 123), EOL(0, 0)),
            ReturnStmt(0, NumberExpr(0, 3, 123), EOL(0, 0, Comment(0, "Hello"))),
            True,
        ),
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
def test_equal_ignoring_offsets_and_eol(
    a: ASTNode, b: ASTNode, exp_equal: bool
) -> None:
    assert equal_ignoring_offsets_and_eol(a, b) is exp_equal


def fcs(name: str) -> FunctionCallStmt:
    return FunctionCallStmt(FunctionCallExpr(0, 0, name, []), EOL(0, 0))


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
        # Nesting with 'else' belonging to parent
        (
            ("if (True):\n" "    if (True):\n" "        a()\n" "else:\n" "    b()"),
            IfElseStmt(
                [
                    IfBranch(
                        0,
                        BooleanExpr(0, True),
                        [IfElseStmt([IfBranch(0, BooleanExpr(0, True), [fcs("a")])])],
                    ),
                ],
                ElseBranch(0, [fcs("b")]),
            ),
        ),
        # Nesting with 'else if' belonging to parent
        (
            (
                "if (True):\n"
                "    if (True):\n"
                "        a()\n"
                "else if (False):\n"
                "    b()"
            ),
            IfElseStmt(
                [
                    IfBranch(
                        0,
                        BooleanExpr(0, True),
                        [IfElseStmt([IfBranch(0, BooleanExpr(0, True), [fcs("a")])])],
                    ),
                    IfBranch(0, BooleanExpr(0, False), [fcs("b")],),
                ],
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

        assert equal_ignoring_offsets_and_eol(if_else_stmt, exp_if_else_stmt)

        # Sanity check ofsets
        assert if_else_stmt.offset == 11

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
        with pytest.raises(ParseError):
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
    string = "foo(a, b, c):\n{}".format(indent(dedent(string), "    "))

    if exp_variable is not None and exp_variable is not None:
        ast = parse(string)
        (function,) = ast.functions
        (for_each_stmt,) = function.body
        assert isinstance(for_each_stmt, ForEachStmt)
        assert for_each_stmt.offset == 18

        assert for_each_stmt.variable.name == exp_variable

        assert [
            cast(Variable, cast(VariableExpr, v).variable).name
            for v in for_each_stmt.values
        ] == exp_values

        assert len(for_each_stmt.body) == 1
        assert equal_ignoring_offsets_and_eol(for_each_stmt.body[0], fcs("a"))
    else:
        with pytest.raises(ParseError):
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
        assert equal_ignoring_offsets_and_eol(for_stmt.body[0], fcs("a"))
    else:
        with pytest.raises(ParseError):
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

        assert equal_ignoring_offsets_and_eol(for_stmt.condition, exp_condition)

        assert len(for_stmt.body) == 1
        assert equal_ignoring_offsets_and_eol(for_stmt.body[0], fcs("a"))
    else:
        with pytest.raises(ParseError):
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
        ("return True", ReturnStmt(0, BooleanExpr(0, True), EOL(0, 0))),
        # Extra space
        ("return   True", ReturnStmt(0, BooleanExpr(0, True), EOL(0, 0))),
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
        assert equal_ignoring_offsets_and_eol(return_stmt, exp_stmt)
    else:
        with pytest.raises(ParseError):
            parse(string)


@pytest.mark.parametrize(
    "lhs_string, exp_lhs",
    [
        ("x", Variable(0, "x")),
        ("x[y]", Subscript(0, Variable(0, "x"), LabelExpr(Label(0, "y")))),
    ],
)
@pytest.mark.parametrize("spacing", ["", " "])
@pytest.mark.parametrize("op", AssignmentOp)
def test_assignment_statement(
    lhs_string: str, exp_lhs: Union[Variable, Subscript], spacing: str, op: AssignmentOp
) -> None:
    string = "foo(x): {}{}{}{}b".format(lhs_string, spacing, op.value, spacing)
    ast = parse(string)
    (function,) = ast.functions
    (assignment_stmt,) = function.body
    assert isinstance(assignment_stmt, AssignmentStmt)
    assert assignment_stmt.offset == 8

    assert equal_ignoring_offsets_and_eol(assignment_stmt.variable, exp_lhs)
    assert assignment_stmt.op == op
    assert equal_ignoring_offsets_and_eol(
        assignment_stmt.value, LabelExpr(Label(0, "b"))
    )


def parse_expr(string: str) -> Expr:
    ast = parse("foo(): return {}".format(string))
    (function,) = ast.functions
    return_stmt = cast(ReturnStmt, function.body[0])
    return return_stmt.value


@pytest.mark.parametrize("spacing", ["", " "])
@pytest.mark.parametrize("op", UnaryOp)
def test_unary_expr(spacing: str, op: UnaryOp) -> None:
    expr = parse_expr("{}{} foo".format(op.value, spacing))
    assert isinstance(expr, UnaryExpr)

    assert equal_ignoring_offsets_and_eol(
        expr, UnaryExpr(0, op, LabelExpr(Label(0, "foo"))),
    )


@pytest.mark.parametrize("spacing", ["", " "])
@pytest.mark.parametrize("op", BinaryOp)
def test_binary_expr(spacing: str, op: BinaryOp) -> None:
    # Special case: and/or *require* whitespace
    if spacing == "" and op in (BinaryOp.logical_and, BinaryOp.logical_or):
        return

    expr = parse_expr("foo{}{}{}bar".format(spacing, op.value, spacing))
    assert isinstance(expr, BinaryExpr)

    assert equal_ignoring_offsets_and_eol(
        expr, BinaryExpr(LabelExpr(Label(0, "foo")), op, LabelExpr(Label(0, "bar")),),
    )


def test_operator_precedence_table_sanity() -> None:
    # Check all operators with same precedence are of the same type (i.e.
    # all binary or all unary)
    for score in set(OPERATOR_PRECEDENCE_TABLE.values()):
        types = set(
            type(op) for op, s in OPERATOR_PRECEDENCE_TABLE.items() if score == s
        )
        assert len(types) == 1


@pytest.mark.parametrize(
    "op", [op for op in OPERATOR_PRECEDENCE_TABLE if isinstance(op, BinaryOp)]
)
def test_binary_operators(op: BinaryOp) -> None:
    score = OPERATOR_PRECEDENCE_TABLE[op]

    # Check associativity with same precedence
    for other_op in [
        other_op
        for other_op, other_score in OPERATOR_PRECEDENCE_TABLE.items()
        if other_score == score
    ]:
        # Sanity check invariant: operators with same associativity are same
        # type
        assert isinstance(other_op, BinaryOp)

        # Sanity check invariant: operators with same associativity have same
        # associativity
        assert (
            OPERATOR_ASSOCIATIVITY_TABLE[op] == OPERATOR_ASSOCIATIVITY_TABLE[other_op]
        )
        associativity = OPERATOR_ASSOCIATIVITY_TABLE[op]

        for op1, op2 in [(op, other_op), (other_op, op)]:
            expr = parse_expr(f"a {op1.value} b {op2.value} c")
            if associativity == Associativity.left:
                assert equal_ignoring_offsets_and_eol(
                    expr,
                    BinaryExpr(
                        BinaryExpr(
                            LabelExpr(Label(0, "a")), op1, LabelExpr(Label(0, "b")),
                        ),
                        op2,
                        LabelExpr(Label(0, "c")),
                    ),
                )
            elif associativity == Associativity.right:
                assert equal_ignoring_offsets_and_eol(
                    expr,
                    BinaryExpr(
                        LabelExpr(Label(0, "a")),
                        op1,
                        BinaryExpr(
                            LabelExpr(Label(0, "b")), op2, LabelExpr(Label(0, "c")),
                        ),
                    ),
                )
            else:
                raise NotImplementedError(associativity)  # Unreachable...

    # Check when combined with lower-precedence binary op, this op has higher
    # precedence
    for other_op in [
        other_op
        for other_op, other_score in OPERATOR_PRECEDENCE_TABLE.items()
        if other_score < score and isinstance(other_op, BinaryOp)
    ]:
        expr = parse_expr(f"a {op.value} b {other_op.value} c")
        assert equal_ignoring_offsets_and_eol(
            expr,
            BinaryExpr(
                BinaryExpr(LabelExpr(Label(0, "a")), op, LabelExpr(Label(0, "b")),),
                other_op,
                LabelExpr(Label(0, "c")),
            ),
        )

        expr = parse_expr(f"a {other_op.value} b {op.value} c")
        assert equal_ignoring_offsets_and_eol(
            expr,
            BinaryExpr(
                LabelExpr(Label(0, "a")),
                other_op,
                BinaryExpr(LabelExpr(Label(0, "b")), op, LabelExpr(Label(0, "c")),),
            ),
        )

    # Check when combined with lower-precedence unary op, this op has higher
    # precedence
    for other_op in [
        other_op
        for other_op, other_score in OPERATOR_PRECEDENCE_TABLE.items()
        if other_score < score and isinstance(other_op, UnaryOp)
    ]:
        expr = parse_expr(f"{other_op.value} a {op.value} b")
        assert equal_ignoring_offsets_and_eol(
            expr,
            UnaryExpr(
                0,
                other_op,
                BinaryExpr(LabelExpr(Label(0, "a")), op, LabelExpr(Label(0, "b")),),
            ),
        )

        # Special case: the 'not' operator cannot be used on the RHS of a
        # binary expression (a quirk of the grammar, shared with other
        # languages)
        if other_op == UnaryOp.logical_not:
            with pytest.raises(ParseError):
                parse_expr(f"a {op.value} {other_op.value} b")
        else:
            expr = parse_expr(f"a {op.value} {other_op.value} b")
            assert equal_ignoring_offsets_and_eol(
                expr,
                BinaryExpr(
                    LabelExpr(Label(0, "a")),
                    op,
                    UnaryExpr(0, other_op, LabelExpr(Label(0, "b"))),
                ),
            )


@pytest.mark.parametrize(
    "op", [op for op in OPERATOR_PRECEDENCE_TABLE if isinstance(op, UnaryOp)]
)
def test_unary_operators(op: UnaryOp) -> None:
    score = OPERATOR_PRECEDENCE_TABLE[op]

    # Check right-associative with same precedence
    for other_op in [
        other_op
        for other_op, other_score in OPERATOR_PRECEDENCE_TABLE.items()
        if other_score == score
    ]:
        assert isinstance(other_op, UnaryOp)
        for op1, op2 in [(op, other_op), (other_op, op)]:
            expr = parse_expr(f"{op1.value} {op2.value} a")
            assert equal_ignoring_offsets_and_eol(
                expr, UnaryExpr(0, op1, UnaryExpr(0, op2, LabelExpr(Label(0, "a")),),),
            )

    # Check when combined with lower-precedence binary op, this op has higher
    # precedence
    for other_op in [
        other_op
        for other_op, other_score in OPERATOR_PRECEDENCE_TABLE.items()
        if other_score < score and isinstance(other_op, BinaryOp)
    ]:
        expr = parse_expr(f"{op.value} a {other_op.value} b")
        assert equal_ignoring_offsets_and_eol(
            expr,
            BinaryExpr(
                UnaryExpr(0, op, LabelExpr(Label(0, "a"))),
                other_op,
                LabelExpr(Label(0, "b")),
            ),
        )

        expr = parse_expr(f"a {other_op.value} {op.value} b")
        assert equal_ignoring_offsets_and_eol(
            expr,
            BinaryExpr(
                LabelExpr(Label(0, "a")),
                other_op,
                UnaryExpr(0, op, LabelExpr(Label(0, "b"))),
            ),
        )

    # Check when combined with other unary ops, the ops are processed
    # left-to-right regardless
    for other_op in [
        other_op
        for other_op, other_score in OPERATOR_PRECEDENCE_TABLE.items()
        if isinstance(other_op, UnaryOp)
    ]:
        # NB: the 'not' operator cannot be used on the RHS of a
        # unary expression (except with itself) (a quirk of the grammar, shared
        # with other languages)

        if op == UnaryOp.logical_not and op != other_op:
            with pytest.raises(ParseError):
                parse_expr(f"{other_op.value} {op.value} a")
        else:
            expr = parse_expr(f"{other_op.value} {op.value} a")
            assert equal_ignoring_offsets_and_eol(
                expr,
                UnaryExpr(0, other_op, UnaryExpr(0, op, LabelExpr(Label(0, "a"))),),
            )

        if other_op == UnaryOp.logical_not and op != other_op:
            with pytest.raises(ParseError):
                parse_expr(f"{op.value} {other_op.value} a")
        else:
            expr = parse_expr(f"{op.value} {other_op.value} a")
            assert equal_ignoring_offsets_and_eol(
                expr,
                UnaryExpr(0, op, UnaryExpr(0, other_op, LabelExpr(Label(0, "a"))),),
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
        assert equal_ignoring_offsets_and_eol(
            expr, FunctionCallExpr(0, 0, exp_name, exp_arguments)
        )
    else:
        with pytest.raises(ParseError):
            parse_expr(string)


@pytest.mark.parametrize(
    "string, exp_variable_expr",
    [
        # Simple variable
        ("foo", VariableExpr(Variable(0, "foo"))),
        # Subscripted
        (
            "foo[True]",
            VariableExpr(Subscript(0, Variable(0, "foo"), BooleanExpr(0, True))),
        ),
        # Nested subscripts
        (
            "foo[1][2][3]",
            VariableExpr(
                Subscript(
                    0,
                    Subscript(
                        0,
                        Subscript(0, Variable(0, "foo"), NumberExpr(0, 0, 1),),
                        NumberExpr(0, 0, 2),
                    ),
                    NumberExpr(0, 0, 3),
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
    if exp_variable_expr is None:
        with pytest.raises(ParseError):
            parse("foo(foo): return {}".format(string))
    else:
        ast = parse("foo(foo): return {}".format(string))
        (function,) = ast.functions
        return_stmt = cast(ReturnStmt, function.body[0])
        expr = return_stmt.value
        assert equal_ignoring_offsets_and_eol(expr, exp_variable_expr)


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
        assert expr == EmptyMapExpr(14, 14 + len(string))
    else:
        with pytest.raises(ParseError):
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
        with pytest.raises(ParseError):
            parse_expr(string)


@pytest.mark.parametrize(
    "string, exp_value, exp_display_base, exp_display_digits",
    [
        # Decimal
        ("0", 0, 10, 1),
        ("000", 0, 10, 3),
        ("1", 1, 10, 1),
        ("1234567890", 1234567890, 10, 10),
        ("0123", 123, 10, 4),
        # Hex
        ("0x0", 0x0, 16, 1),
        ("0x1", 0x1, 16, 1),
        ("0x1234567890ABCDEF", 0x1234567890ABCDEF, 16, 16),
        ("0X1234567890ABCDEF", 0x1234567890ABCDEF, 16, 16),
        ("0X1234567890abcdef", 0x1234567890ABCDEF, 16, 16),
        ("0x1234567890abcdef", 0x1234567890ABCDEF, 16, 16),
        # Binary
        ("0b0", 0b0, 2, 1),
        ("0b1", 0b1, 2, 1),
        ("0b10100", 0b10100, 2, 5),
        ("0B10100", 0b10100, 2, 5),
        # Non-numeric char in decimal
        ("0123b", None, None, None),
        # Too-high digit in binary
        ("0b123", None, None, None),
        # Invalid char in hex
        ("0x10g", None, None, None),
        # Too many leading zeros
        ("00b100", None, None, None),
        ("00x100", None, None, None),
        # Space
        ("1 2 3", None, None, None),
        ("0x 100", None, None, None),
        ("0 x100", None, None, None),
        ("0 x 100", None, None, None),
        ("0b 100", None, None, None),
        ("0 b100", None, None, None),
        ("0 b 100", None, None, None),
    ],
)
def test_number_expr(
    string: str,
    exp_value: Optional[int],
    exp_display_base: Optional[int],
    exp_display_digits: Optional[int],
) -> None:
    if (
        exp_value is not None
        and exp_display_base is not None
        and exp_display_digits is not None
    ):
        expr = parse_expr(string)
        assert expr == NumberExpr(
            14, 14 + len(string), exp_value, exp_display_base, exp_display_digits
        )
    else:
        with pytest.raises(ParseError):
            parse_expr(string)


@pytest.mark.parametrize(
    "string, exp_eol",
    [
        # Nothing follows
        ("foo(): return True", EOL(18, 18)),
        ("foo(): return True\n", EOL(18, 19)),
        ("foo(): return True\r", EOL(18, 19)),
        ("foo(): return True\r\n", EOL(18, 20)),
        # Comment
        ("foo(): return True # Hello", EOL(18, 26, Comment(19, "# Hello"))),
        ("foo(): return True # Hello\n", EOL(18, 26, Comment(19, "# Hello"))),
        ("foo(): return True # Hello\r", EOL(18, 26, Comment(19, "# Hello"))),
        ("foo(): return True # Hello\r\n", EOL(18, 26, Comment(19, "# Hello"))),
        # Empty lines
        ("foo(): return True\n\n", EOL(18, 20, None, [EmptyLine(19, 20)])),
        ("foo(): return True\r\r", EOL(18, 20, None, [EmptyLine(19, 20)])),
        ("foo(): return True\r\n\r\n", EOL(18, 22, None, [EmptyLine(20, 22)])),
        # Multiple empty lines
        (
            "foo(): return True\n\n\n",
            EOL(18, 21, None, [EmptyLine(19, 20), EmptyLine(20, 21)]),
        ),
        # Empty lines with comments
        (
            "foo(): return True # Foo\n# Bar\n  # Baz\n",
            EOL(
                18,
                38,
                Comment(19, "# Foo"),
                [
                    EmptyLine(25, 30, Comment(25, "# Bar")),
                    EmptyLine(31, 38, Comment(33, "# Baz")),
                ],
            ),
        ),
        # Empty lines with comments and no final newline
        (
            "foo(): return True # Foo\n# Bar\n  # Baz",
            EOL(
                18,
                38,
                Comment(19, "# Foo"),
                [
                    EmptyLine(25, 30, Comment(25, "# Bar")),
                    EmptyLine(31, 38, Comment(33, "# Baz")),
                ],
            ),
        ),
    ],
)
def test_eol_and_any_ws_and_v_space_and_comment(string: str, exp_eol: EOL) -> None:
    ast = parse(string)
    (function,) = ast.functions
    (return_stmt,) = function.body
    assert isinstance(return_stmt, ReturnStmt)
    assert return_stmt.eol == exp_eol


@pytest.mark.parametrize(
    "string, exp_eol",
    [
        # Inline form gets no EOL
        ("foo(): return True", None),
        # Newline form gets EOL (even if empty)
        ("foo():\n  return True", EOL(6, 7)),
    ],
)
def test_stmt_block_eol(string: str, exp_eol: Optional[EOL]) -> None:
    ast = parse(string)
    (fn,) = ast.functions
    assert fn.eol == exp_eol


def assert_labels_and_variable_types_correct(
    node: ASTNode,
    exp_label_suffixes: Sequence[str],
    exp_variable_suffixes: Sequence[str],
) -> None:
    """
    Asserts that label and variable names have one of the set of suffixes
    provided.
    """
    if isinstance(node, Label):
        assert any(node.name.endswith(suffix) for suffix in exp_label_suffixes), node
    elif isinstance(node, Variable):
        assert any(node.name.endswith(suffix) for suffix in exp_variable_suffixes), node
    else:
        for field in fields(node):
            child_or_child_list = getattr(node, field.name)
            children = (
                child_or_child_list
                if isinstance(child_or_child_list, list)
                else [child_or_child_list]
            )
            for child in children:
                if isinstance(child, ASTNode):
                    assert_labels_and_variable_types_correct(
                        child, exp_label_suffixes, exp_variable_suffixes,
                    )


@pytest.mark.parametrize(
    "node, exp_label_suffixes, exp_variable_suffixes, exp_pass",
    [
        # No allowed suffixies
        (Label(0, "foo_a"), [], [], False),
        (Variable(0, "foo_a"), [], [], False),
        # Labels
        (Label(0, "foo_a"), ["_a", "_b"], [], True),
        (Label(0, "foo_b"), ["_a", "_b"], [], True),
        (Label(0, "foo_c"), ["_a", "_b"], [], False),
        (Label(0, "foo_a"), [], ["_a", "_b"], False),
        # Variables
        (Variable(0, "foo_a"), [], ["_a", "_b"], True),
        (Variable(0, "foo_b"), [], ["_a", "_b"], True),
        (Variable(0, "foo_c"), [], ["_a", "_b"], False),
        (Variable(0, "foo_a"), ["_a", "_b"], [], False),
        # Descends into non-list children
        (
            Subscript(0, Variable(0, "foo_var"), LabelExpr(Label(0, "foo_lab"))),
            ["_lab"],
            ["_var"],
            True,
        ),
        (
            Subscript(0, Variable(0, "foo_var"), LabelExpr(Label(0, "foo_lab"))),
            ["_var"],
            ["_lab"],
            False,
        ),
        (
            Subscript(0, Variable(0, "foo_var"), LabelExpr(Label(0, "foo_lab"))),
            ["_lab"],
            ["_lab"],
            False,
        ),
        (
            Subscript(0, Variable(0, "foo_var"), LabelExpr(Label(0, "foo_lab"))),
            ["_var"],
            ["_var"],
            False,
        ),
        # Descends into children containing lists
        (
            FunctionCallExpr(
                0,
                0,
                "foo",
                [LabelExpr(Label(0, "a_lab")), VariableExpr(Variable(0, "b_var"))],
            ),
            ["_lab"],
            ["_var"],
            True,
        ),
        (
            FunctionCallExpr(
                0,
                0,
                "foo",
                [LabelExpr(Label(0, "a_lab")), VariableExpr(Variable(0, "b_var"))],
            ),
            ["_var"],
            ["_lab"],
            False,
        ),
        (
            FunctionCallExpr(
                0,
                0,
                "foo",
                [LabelExpr(Label(0, "a_lab")), VariableExpr(Variable(0, "b_var"))],
            ),
            ["_lab"],
            ["_lab"],
            False,
        ),
        (
            FunctionCallExpr(
                0,
                0,
                "foo",
                [LabelExpr(Label(0, "a_lab")), VariableExpr(Variable(0, "b_var"))],
            ),
            ["_var"],
            ["_var"],
            False,
        ),
    ],
)
def test_assert_labels_and_variable_types_correct(
    node: ASTNode,
    exp_label_suffixes: Sequence[str],
    exp_variable_suffixes: Sequence[str],
    exp_pass: bool,
) -> None:
    print(node)
    if exp_pass:
        assert_labels_and_variable_types_correct(
            node, exp_label_suffixes, exp_variable_suffixes
        )
    else:
        with pytest.raises(AssertionError):
            assert_labels_and_variable_types_correct(
                node, exp_label_suffixes, exp_variable_suffixes
            )


class TestInferLabels:
    def test_valid_cases(self) -> None:
        source = """
            foo(a_var):  # Arguments are always variables
                bar(a_var, b_lab)  # Never seen before value is a label
                bar(a_var, b_lab)  # ...and still is the next time its used

                # Variables in if conditions and bodies
                if (a_var):
                    bar(a_var)
                # Full if/else if/else
                if (a_var):
                    bar(a_var)
                else if (a_var):
                    bar(a_var)
                else:
                    bar(a_var)

                # Labels in if conditions and bodies
                if (c_lab):
                    bar(d_lab)
                # Full if/else if/else
                if (e_lab):
                    bar(f_lab)
                else if (g_lab):
                    bar(h_lab)
                else:
                    bar(i_lab)

                # Values defined in if stmt should live on afterwards
                if (True):
                    b_var = 123
                    bar(a_var, b_var)
                else if (True):
                    c_var = 123
                    bar(a_var, b_var, c_var)
                else:
                    d_var = 123
                    bar(a_var, b_var, c_var, d_var)
                bar(a_var, b_var, c_var, d_var)

                # For-each defines its own variable, and processes variables and
                # body
                for each j_var in a_var, k_lab:
                    e_var = bar(j_var, k_lab)
                bar(j_var)  # Iterator variable should live on
                bar(e_var)  # Internally defined variable should live on

                # For defines its own variable, and processes bounds and body
                for l_var = a_var to a_var:
                    f_var = bar(l_var, a_var)
                bar(l_var)  # Variable should live on
                bar(f_var)  # Internally defined variable should live on

                for m_var = n_lab to n_lab:
                    bar(m_var, n_lab)
                bar(m_var)  # Variable should live on

                # While variable and body
                while (a_var):
                    g_var = bar(a_var, o_lab)
                bar(g_var)  # Internally defined variable should live on
                while (p_lab):
                    bar(a_var, p_lab)

                # Function call
                bar(a_var, q_lab)

                # Return
                return a_var
                return r_lab

                # Assignment
                a_var = a_var
                a_var = s_lab
                t_var = {}
                t_var[a_var][u_lab] = a_var

                # Paren expr
                a_var = (a_var)
                a_var = (v_lab)

                # Unary expr
                a_var = -a_var
                a_var = -w_lab

                # Binary expr
                a_var = a_var - a_var
                a_var = x_lab - y_lab

                # Empty map
                a_var = {}

                # Boolean
                a_var = True

                # Number
                a_var = 123
        """
        assert_labels_and_variable_types_correct(
            parse(source), ["_lab"], ["_var"],
        )

    @pytest.mark.parametrize(
        "source, exp_error, exp_message",
        [
            # Attempting to assign to label
            (
                """
                    foo():
                        bar(a)
                        a = 123
                """,
                LabelUsedAsVariableNameError,
                """
                    At line 3 column 5:
                            a = 123
                            ^
                    The name 'a' is already in use as a label name.
                """,
            ),
            # Attempting to assign undefined variable to itself
            (
                """
                    foo():
                        a = a
                """,
                LabelUsedAsVariableNameError,
                """
                    At line 2 column 5:
                            a = a
                            ^
                    The name 'a' is already in use as a label name.
                """,
            ),
            # Attempting to subscript a label
            (
                """
                    foo():
                        bar(a)
                        return a[b]
                """,
                CannotSubscriptLabelError,
                """
                    At line 3 column 12:
                            return a[b]
                                   ^
                    Attempting to subscript label 'a'.
                """,
            ),
            # Attempting to assign to subscript of label
            (
                """
                    foo():
                        bar(a)
                        a[b] = 123
                """,
                CannotSubscriptLabelError,
                """
                    At line 3 column 5:
                            a[b] = 123
                            ^
                    Attempting to subscript label 'a'.
                """,
            ),
            # Attempting to use newly defined variable in its own subscript
            (
                """
                    foo():
                        a[a] = 123
                """,
                CannotSubscriptLabelError,
                """
                    At line 2 column 5:
                            a[a] = 123
                            ^
                    Attempting to subscript label 'a'.
                """,
            ),
        ],
    )
    def test_errors(
        self, source: str, exp_error: Type[ASTConstructionError], exp_message: str
    ) -> None:
        with pytest.raises(exp_error) as excinfo:
            parse(dedent(source).strip())

        assert str(excinfo.value) == dedent(exp_message).strip()


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
        # Function missing open paren
        (
            "foo a",
            """
                    foo a
                        ^
                Expected '('
            """,
        ),
        # Function missing close paren
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
        # If with no paren
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
        # If with no close paren
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
        # If with no close paren (and lots of potential expression parses)
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
        # Missing paren expression
        (
            """
                foo(a):
                    return ()
            """,
            """
                        return ()
                                ^
                Expected <expression>
            """,
        ),
        # Missing closing paren and expression
        (
            """
                foo(a):
                    return (
            """,
            """
                        return (
                                ^
                Expected <expression>
            """,
        ),
        # Missing closing paren
        (
            """
                foo(a):
                    return (1
            """,
            """
                        return (1
                                 ^
                Expected ')' or <operator>
            """,
        ),
        # Missing space after not
        (
            """
                foo(a):
                    return not-1
            """,
            """
                        return not-1
                                  ^
                Expected <space>
            """,
        ),
        # Missing space before and
        (
            """
                foo(a):
                    return 1and -2
            """,
            """
                        return 1and -2
                                ^
                Expected <operator>
            """,
        ),
        # Missing space after and
        (
            """
                foo(a):
                    return 1 and-2
            """,
            """
                        return 1 and-2
                                    ^
                Expected <space>
            """,
        ),
        # Missing space before or
        (
            """
                foo(a):
                    return 1or -2
            """,
            """
                        return 1or -2
                                ^
                Expected <operator>
            """,
        ),
        # Missing space after or
        (
            """
                foo(a):
                    return 1 or-2
            """,
            """
                        return 1 or-2
                                   ^
                Expected <space>
            """,
        ),
    ]
    + [
        # Missing unary-op operand
        (
            f"""
                foo(a):
                    return ({op.value} )
            """,
            f"""
                        return ({op.value} )
                                 {" "*len(op.value)}^
                Expected <expression>
            """,
        )
        for op in UnaryOp
    ]
    + [
        # Missing binary-op operand
        (
            f"""
                foo(a):
                    return (1 {op.value} )
            """,
            f"""
                        return (1 {op.value} )
                                   {" "*len(op.value)}^
                Expected <expression>
            """,
        )
        for op in BinaryOp
    ],
)
def test_parse_error_messages(string: str, exp_error: str) -> None:
    with pytest.raises(ParseError) as exc_info:
        parse(dedent(string))
    exc = exc_info.value
    message = "\n".join(map(str.rstrip, str(exc).splitlines()[1:]))
    assert message == dedent(exp_error[1:]).rstrip()


@pytest.mark.parametrize("name", pseudocode_samples.__all__)
@pytest.mark.parametrize("newlines", ["\n", "\r", "\r\n"])
def test_pseudocode_samples(name: str, newlines: str) -> None:
    source = getattr(pseudocode_samples, name)
    # Just check no parse error occurs
    parse(source.replace("\n", newlines))
