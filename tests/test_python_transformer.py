import pytest  # type: ignore

from typing import cast

import ast
import _ast

from textwrap import dedent

from vc2_pseudocode.ast import (
    Listing,
    Function,
    ReturnStmt,
    Expr,
    PerenExpr,
    UnaryOp,
    UnaryExpr,
    BinaryOp,
    BinaryExpr,
    NumberExpr,
    FunctionCallExpr,
    VariableExpr,
    Variable,
)

from vc2_pseudocode.parser import parse

from vc2_pseudocode.python_transformer import (
    UndefinedArrayOrMapError,
    InvalidPowArgumentsError,
    PYTHON_OPERATOR_PRECEDENCE_TABLE,
    expr_add_one,
    PythonTransformer,
    pseudocode_to_python,
)


def test_binary_op_precedence_table_completeness() -> None:
    assert set(PYTHON_OPERATOR_PRECEDENCE_TABLE) == set(BinaryOp) | set(UnaryOp)


class TestExprAddOne:
    def test_decimal_number(self) -> None:
        e = NumberExpr(100, 103, 123, 10)
        assert expr_add_one(e) == NumberExpr(100, 103, 124, 10)

    @pytest.mark.parametrize("display_base", [2, 16])
    def test_non_decimal_number(self, display_base: int) -> None:
        e = NumberExpr(100, 103, 123, display_base)
        assert expr_add_one(e) == BinaryExpr(
            NumberExpr(100, 103, 123, display_base),
            BinaryOp("+"),
            NumberExpr(100, 100, 1, 10),
        )

    def test_adding_decimal_number(self) -> None:
        e = BinaryExpr(
            NumberExpr(10, 11, 0, 10), BinaryOp("+"), NumberExpr(100, 103, 123, 10),
        )
        assert expr_add_one(e) == BinaryExpr(
            NumberExpr(10, 11, 0, 10), BinaryOp("+"), NumberExpr(100, 103, 124, 10),
        )

    @pytest.mark.parametrize("display_base", [2, 16])
    def test_adding_non_decimal_number(self, display_base: int) -> None:
        e = BinaryExpr(
            NumberExpr(10, 11, 0, 10),
            BinaryOp("+"),
            NumberExpr(100, 103, 123, display_base),
        )
        assert expr_add_one(e) == BinaryExpr(
            e, BinaryOp("+"), NumberExpr(10, 10, 1, 10),
        )

    def test_subtracting_decimal_number(self) -> None:
        e = BinaryExpr(
            NumberExpr(10, 11, 0, 10), BinaryOp("-"), NumberExpr(100, 103, 123, 10),
        )
        assert expr_add_one(e) == BinaryExpr(
            NumberExpr(10, 11, 0, 10), BinaryOp("-"), NumberExpr(100, 103, 122, 10),
        )

    def test_subtracting_one(self) -> None:
        e = BinaryExpr(
            NumberExpr(10, 11, 0, 10), BinaryOp("-"), NumberExpr(100, 101, 1, 10),
        )
        assert expr_add_one(e) == NumberExpr(10, 11, 0, 10)

    @pytest.mark.parametrize("display_base", [2, 16])
    def test_subtracting_non_decimal_number(self, display_base: int) -> None:
        e = BinaryExpr(
            NumberExpr(10, 11, 0, 10),
            BinaryOp("-"),
            NumberExpr(100, 103, 123, display_base),
        )
        assert expr_add_one(e) == BinaryExpr(
            e, BinaryOp("+"), NumberExpr(10, 10, 1, 10),
        )

    @pytest.mark.parametrize(
        "expr",
        [
            BinaryExpr(
                NumberExpr(10, 13, 999, 10), BinaryOp("*"), NumberExpr(20, 12, 2, 10),
            ),
            FunctionCallExpr(10, 20, "foo", []),
        ],
    )
    def test_other_values(self, expr: Expr) -> None:
        assert expr_add_one(expr) == BinaryExpr(
            expr, BinaryOp("+"), NumberExpr(10, 10, 1, 10),
        )


@pytest.mark.parametrize(
    "pseudocode, exp_python",
    [
        # A single (argument free) function
        (
            """
            foo():
                return 0
            """,
            """
            def foo():
                return 0
            """,
        ),
        # Multiple functions, with arguments
        (
            """
            foo(a):
                return 0
            bar(b, c):
                return 0
            """,
            """
            def foo(a):
                return 0


            def bar(b, c):
                return 0
            """,
        ),
        # If statement
        (
            """
            foo(x):
                if (x == 1):
                    return 100
            """,
            """
            def foo(x):
                if x == 1:
                    return 100
            """,
        ),
        # If-elif statement
        (
            """
            foo(x):
                if (x == 1):
                    return 100
                else if (x == 2):
                    return 200
                else if (x == 3):
                    return 300
            """,
            """
            def foo(x):
                if x == 1:
                    return 100
                elif x == 2:
                    return 200
                elif x == 3:
                    return 300
            """,
        ),
        # If-elif-else statement
        (
            """
            foo(x):
                if (x == 1):
                    return 100
                else if (x == 2):
                    return 200
                else:
                    return 300
            """,
            """
            def foo(x):
                if x == 1:
                    return 100
                elif x == 2:
                    return 200
                else:
                    return 300
            """,
        ),
        # For each stmt
        (
            """
            foo(x):
                for each y in 1:
                    bar(y)
                for each y in 1, 2, 3, x:
                    bar(y)
            """,
            """
            def foo(x):
                for y in (1, ):
                    bar(y)
                for y in (1, 2, 3, x):
                    bar(y)
            """,
        ),
        # For stmt
        (
            """
            foo(x):
                for y = 1 to 3:
                    bar(y)
            """,
            """
            def foo(x):
                for y in range(1, 4):
                    bar(y)
            """,
        ),
        # For stmt from zero
        (
            """
            foo(x):
                for y = 0 to 3:
                    bar(y)
            """,
            """
            def foo(x):
                for y in range(4):
                    bar(y)
            """,
        ),
        # For stmt from zero but not base 10
        (
            """
            foo(x):
                for y = 0x0 to 3:
                    bar(y)
            """,
            """
            def foo(x):
                for y in range(0x0, 4):
                    bar(y)
            """,
        ),
        # For stmt with add/sub final value
        (
            """
            foo(x):
                for y = 1 to x + 100:
                    bar(y)
                for y = 1 to x - 100:
                    bar(y)
                for y = 1 to x - 1:
                    bar(y)
            """,
            """
            def foo(x):
                for y in range(1, x + 101):
                    bar(y)
                for y in range(1, x - 99):
                    bar(y)
                for y in range(1, x):
                    bar(y)
            """,
        ),
        # For stmt with non-number final value
        (
            """
            foo(x):
                for y = 1 to x:
                    bar(y)
            """,
            """
            def foo(x):
                for y in range(1, x + 1):
                    bar(y)
            """,
        ),
        # While stmt
        (
            """
            foo(x):
                while (x != {}):
                    bar(x)
            """,
            """
            def foo(x):
                while x != {}:
                    bar(x)
            """,
        ),
        # Function call stmt
        (
            """
            foo(x):
                bar()
                bar(x)
                bar(x, x + 1, x + 2)
            """,
            """
            def foo(x):
                bar()
                bar(x)
                bar(x, x + 1, x + 2)
            """,
        ),
        # Return stmt
        (
            """
            foo(x):
                return x + 1
            """,
            """
            def foo(x):
                return x + 1
            """,
        ),
        # Assignment stmt
        (
            """
            foo(x):
                x += 1
            """,
            """
            def foo(x):
                x += 1
            """,
        ),
        # Names not in current scope are considered labels
        (
            """
            foo(a):
                b = 100
                for each c in 1, 2, 3:
                    bar(a, b, c, d, e)
                for d = 1 to 3:
                    bar(a, b, c, d, e)
                return a + b + c + d + e + a() + b() + c() + d() + e()
            """,
            """
            def foo(a):
                b = 100
                for c in (1, 2, 3):
                    bar(a, b, c, 'd', 'e')
                for d in range(1, 4):
                    bar(a, b, 'c', d, 'e')
                return a + b + 'c' + 'd' + 'e' + a() + b() + c() + d() + e()
            """,
        ),
        # Redefining names in nested scopes doesn't cause problems
        (
            """
            foo(a):
                for each a in 1, 2, 3:
                    bar(a, b)
                return a + b
            """,
            """
            def foo(a):
                for a in (1, 2, 3):
                    bar(a, 'b')
                return a + 'b'
            """,
        ),
        # Use of names in subscripts doesn't count as defining a variable
        (
            """
            foo(a):
                a[b] = c
                return a + b + c
            """,
            """
            def foo(a):
                a['b'] = 'c'
                return a + 'b' + 'c'
            """,
        ),
        # Variable is unused until it is...
        (
            """
            foo():
                a = a
            """,
            """
            def foo():
                a = 'a'
            """,
        ),
        # Perentheses are passed through (nowever unnecessary)
        (
            """
            foo(a, b):
                return (a) + ((b))
            """,
            """
            def foo(a, b):
                return (a) + ((b))
            """,
        ),
        # Unary expressions are transformed appropriately
        (
            """
            foo(a, b, c, d):
                bar(!a, +b, -c, not d)
            """,
            """
            def foo(a, b, c, d):
                bar(~a, +b, -c, not d)
            """,
        ),
        # Binary expressions (extra brackets not added as precedence rules
        # match)
        (
            """
            foo(a, b, c):
                bar(a + b + c)
                bar(a * b // c)
                bar(a + !b & c != 0 or 1)
            """,
            """
            def foo(a, b, c):
                bar(a + b + c)
                bar(a * b // c)
                bar(a + ~b & c != 0 or 1)
            """,
        ),
        # Exponentiation
        (
            """
            foo(a, b, c):
                return pow(a, b)
                return pow(a + b, c)
                return pow(a, b + c)
                return pow(-a, b)
                return pow(a, -b)
                return -pow(a, b)
                return pow(a, b) + pow(b, c)
            """,
            """
            def foo(a, b, c):
                return a ** b
                return (a + b) ** c
                return a ** (b + c)
                return (-a) ** b
                return a ** (-b)
                return -(a ** b)
                return a ** b + b ** c
            """,
        ),
        # Variable expression
        (
            """
            foo(a, b, c):
                bar(a, a[b], a[b][c])
            """,
            """
            def foo(a, b, c):
                bar(a, a[b], a[b][c])
            """,
        ),
        # Empty map expression
        (
            """
            foo():
                return {}
            """,
            """
            def foo():
                return {}
            """,
        ),
        # Boolean expressions
        (
            """
            foo():
                return True != False
            """,
            """
            def foo():
                return True != False
            """,
        ),
        # Number expressions
        (
            """
            foo():
                return 192 + 0X123abc + 0B1010
                return 00192 + 0X00123abc + 0B001010
            """,
            """
            def foo():
                return 192 + 0x123ABC + 0b1010
                return 192 + 0x00123ABC + 0b001010
            """,
        ),
        # Normalise vertical whitespace
        (
            """
            foo():
                foo()
                bar()

                baz()



                qux()
            bar():
                return 0

            baz():
                return 0




            qux():
                return 0
            """,
            """
            def foo():
                foo()
                bar()

                baz()

                qux()


            def bar():
                return 0


            def baz():
                return 0


            def qux():
                return 0
            """,
        ),
        # Comments around functions
        (
            """
            # Leading comment at start
            # With adjacent second line

            # And non-adjacent third line


            # And very non-adjacent fourth line and then a space...

            foo():
                return 0

            # And spaced between functions

            bar():
                return 0

            # And adjacent to a function
            baz():
                return 0

            qux():  # And on a function definition
                # Before an if
                if (True):  # And on an if
                    # And inside an if
                    foo()
                    # And before an else if
                else if (True):  # And on an else if
                    # And inside an else if
                    foo()
                    # And before an else
                else:  # And on an else
                    # And inside an else
                    foo()

                # Before a for each
                for each x in 1, 2, 3:  # On a for each
                    # Inside a for each
                    foo()

                # Before a for
                for x = 1 to 3:  # On a for
                    # Inside a for
                    foo()

                # Before a while
                while (False):  # On a while
                    # Inside a while
                    foo()

                # Before an assignment
                x = 100  # On an assignment

                # Before a call
                foo()  # On a call

                # Before a return
                return 0  # On a return

            quo():
                # Spaced before an if

                if (True):
                    foo()
                    # And spaced before an else if

                else if (True):
                    foo()
                    # And spaced before an else

                else:
                    foo()

                # Spaced before a for each

                for each x in 1, 2, 3:
                    foo()

                # Spaced before a for
                for x = 1 to 3:
                    foo()

                # Spaced before a while
                while (False):
                    foo()

                # Spaced before an assignment

                x = 100

                # Spaced before a call

                foo()

                # Spaced before a return

                return 0

            qac(): return 0  # Comment on one-liner function

            qiz():
                if (True): return 0  # Comment on one-liner if
                else if (True): return 0  # Comment on one-liner else if
                else: return 0  # Comment on one-liner else

                for each x in 1, 2, 3: foo()  # Comment on one-liner for each

                for x = 1 to 3: foo()  # Comment on one-liner for

                while (False): foo()  # Comment on one-liner while

            # And at the end of a file
            # Adjacent

            # And non-adjacent


            # And very non-adjacent
            """,
            # ------------------------------------------------------------------
            """
            # Leading comment at start
            # With adjacent second line

            # And non-adjacent third line

            # And very non-adjacent fourth line and then a space...


            def foo():
                return 0


            # And spaced between functions


            def bar():
                return 0


            # And adjacent to a function
            def baz():
                return 0


            def qux():  # And on a function definition
                # Before an if
                if True:  # And on an if
                    # And inside an if
                    foo()
                    # And before an else if
                elif True:  # And on an else if
                    # And inside an else if
                    foo()
                    # And before an else
                else:  # And on an else
                    # And inside an else
                    foo()

                # Before a for each
                for x in (1, 2, 3):  # On a for each
                    # Inside a for each
                    foo()

                # Before a for
                for x in range(1, 4):  # On a for
                    # Inside a for
                    foo()

                # Before a while
                while False:  # On a while
                    # Inside a while
                    foo()

                # Before an assignment
                x = 100  # On an assignment

                # Before a call
                foo()  # On a call

                # Before a return
                return 0  # On a return


            def quo():
                # Spaced before an if

                if True:
                    foo()
                    # And spaced before an else if

                elif True:
                    foo()
                    # And spaced before an else

                else:
                    foo()

                # Spaced before a for each

                for x in (1, 2, 3):
                    foo()

                # Spaced before a for
                for x in range(1, 4):
                    foo()

                # Spaced before a while
                while False:
                    foo()

                # Spaced before an assignment

                x = 100

                # Spaced before a call

                foo()

                # Spaced before a return

                return 0


            def qac():  # Comment on one-liner function
                return 0


            def qiz():
                if True:  # Comment on one-liner if
                    return 0
                elif True:  # Comment on one-liner else if
                    return 0
                else:  # Comment on one-liner else
                    return 0

                for x in (1, 2, 3):  # Comment on one-liner for each
                    foo()

                for x in range(1, 4):  # Comment on one-liner for
                    foo()

                while False:  # Comment on one-liner while
                    foo()


            # And at the end of a file
            # Adjacent

            # And non-adjacent

            # And very non-adjacent
            """,
        ),
    ],
)
def test_transformer(pseudocode: str, exp_python: str) -> None:
    listing = parse(pseudocode)
    transformer = PythonTransformer(pseudocode)
    python = transformer.transform(listing)
    assert python == dedent(exp_python).strip()

    # Verify that the output is valid Python
    assert compile(python, "<none>", "exec") is not None


@pytest.mark.parametrize(
    "pseudocode, exp_error",
    [
        # Assigning to map/array which isn't defined
        (
            """
            foo():
                a[0] = 100
            """,
            """
            At line 2 column 5:
                    a[0] = 100
                    ^
            Map or array 'a' not defined.
            """,
        ),
        # Subscripting map/array which isn't defined
        (
            """
            foo():
                return a[0]
            """,
            """
            At line 2 column 12:
                    return a[0]
                           ^
            Map or array 'a' not defined.
            """,
        ),
    ],
)
def test_undefined_array_or_map(pseudocode: str, exp_error: str) -> None:
    pseudocode = dedent(pseudocode).strip()
    listing = parse(pseudocode)
    transformer = PythonTransformer(pseudocode)
    with pytest.raises(UndefinedArrayOrMapError) as exc_info:
        transformer.transform(listing)
    assert str(exc_info.value) == dedent(exp_error).strip()


@pytest.mark.parametrize(
    "pseudocode, exp_error",
    [
        # Too-few arguments
        (
            """
            foo():
                return pow()
            """,
            """
            At line 2 column 12:
                    return pow()
                           ^
            The pow function expects exactly two arguments.
            """,
        ),
        (
            """
            foo():
                return pow(1)
            """,
            """
            At line 2 column 12:
                    return pow(1)
                           ^
            The pow function expects exactly two arguments.
            """,
        ),
        # Too many arguments
        (
            """
            foo():
                return pow(1, 2, 3)
            """,
            """
            At line 2 column 12:
                    return pow(1, 2, 3)
                           ^
            The pow function expects exactly two arguments.
            """,
        ),
    ],
)
def test_invalid_pow_arguments(pseudocode: str, exp_error: str) -> None:
    pseudocode = dedent(pseudocode).strip()
    listing = parse(pseudocode)
    transformer = PythonTransformer(pseudocode)
    with pytest.raises(InvalidPowArgumentsError) as exc_info:
        transformer.transform(listing)
    assert str(exc_info.value) == dedent(exp_error).strip()


class PythonToBracketed(ast.NodeTransformer):
    """
    Transforms simple Python expressions into fully bracketed string
    representations (e.g. ``((a + b) + c)``).
    """

    def visit_Expression(self, node: _ast.Expression) -> str:
        return cast(str, self.visit(node.body))

    def visit_Compare(self, node: _ast.Compare) -> str:
        left = cast(str, self.visit(node.left))
        for op, right in zip(node.ops, node.comparators):
            op = self.visit(op)
            right = self.visit(right)
            left = f"({left} {op} {right})"
        return left

    def visit_BoolOp(self, node: _ast.BoolOp) -> str:
        left = cast(str, self.visit(node.values[0]))
        op = self.visit(node.op)
        for right in node.values[1:]:
            right = self.visit(right)
            left = f"({left} {op} {right})"
        return left

    def visit_BinOp(self, node: _ast.BinOp) -> str:
        left = self.visit(node.left)
        op = self.visit(node.op)
        right = self.visit(node.right)
        return f"({left} {op} {right})"

    def visit_UnaryOp(self, node: _ast.UnaryOp) -> str:
        op = self.visit(node.op)
        operand = self.visit(node.operand)
        return f"({op}{operand})"

    def visit_Name(self, node: _ast.Name) -> str:
        return node.id

    def visit_Add(self, node: _ast.Add) -> str:
        return "+"

    def visit_Sub(self, node: _ast.Sub) -> str:
        return "-"

    def visit_Mult(self, node: _ast.Mult) -> str:
        return "*"

    def visit_LShift(self, node: _ast.LShift) -> str:
        return "<<"

    def visit_RShift(self, node: _ast.RShift) -> str:
        return ">>"

    def visit_BitOr(self, node: _ast.BitOr) -> str:
        return "|"

    def visit_BitXor(self, node: _ast.BitXor) -> str:
        return "^"

    def visit_BitAnd(self, node: _ast.BitAnd) -> str:
        return "&"

    def visit_FloorDiv(self, node: _ast.FloorDiv) -> str:
        return "//"

    def visit_Mod(self, node: _ast.Mod) -> str:
        return "%"

    def visit_Pow(self, node: _ast.Pow) -> str:
        return "**"

    def visit_Eq(self, node: _ast.Eq) -> str:
        return "=="

    def visit_NotEq(self, node: _ast.NotEq) -> str:
        return "!="

    def visit_Lt(self, node: _ast.Lt) -> str:
        return "<"

    def visit_LtE(self, node: _ast.LtE) -> str:
        return "<="

    def visit_Gt(self, node: _ast.Gt) -> str:
        return ">"

    def visit_GtE(self, node: _ast.GtE) -> str:
        return ">="

    def visit_And(self, node: _ast.And) -> str:
        return "and"

    def visit_Or(self, node: _ast.Or) -> str:
        return "or"

    def visit_Invert(self, node: _ast.Invert) -> str:
        return "!"

    def visit_UAdd(self, node: _ast.UAdd) -> str:
        return "+"

    def visit_USub(self, node: _ast.USub) -> str:
        return "-"

    def visit_Not(self, node: _ast.Not) -> str:
        return "not "


def strip_peren_from_pseudocode_expr(expr: Expr) -> Expr:
    """
    Removes :py:class:`PerenExpr`s from an expression.
    """
    if isinstance(expr, PerenExpr):
        return expr.value
    elif isinstance(expr, UnaryExpr):
        expr.value = strip_peren_from_pseudocode_expr(expr.value)
        return expr
    elif isinstance(expr, BinaryExpr):
        expr.lhs = strip_peren_from_pseudocode_expr(expr.lhs)
        expr.rhs = strip_peren_from_pseudocode_expr(expr.rhs)
        return expr
    elif isinstance(expr, VariableExpr):
        return expr
    elif isinstance(expr, FunctionCallExpr):
        return expr
    else:
        raise NotImplementedError(type(expr))


def pseudocode_to_bracketed(expr: Expr) -> str:
    """
    Transforms simple pseudocode expressions into fully bracketed string
    representations (e.g. ``((a + b) + c)``).
    """
    if isinstance(expr, UnaryExpr):
        op = expr.op.value
        value = pseudocode_to_bracketed(expr.value)
        space = " " if op == "not" else ""
        return f"({op}{space}{value})"
    elif isinstance(expr, BinaryExpr):
        lhs = pseudocode_to_bracketed(expr.lhs)
        op = expr.op.value
        rhs = pseudocode_to_bracketed(expr.rhs)
        return f"({lhs} {op} {rhs})"
    elif isinstance(expr, FunctionCallExpr):
        assert expr.name == "pow"
        lhs, rhs = map(pseudocode_to_bracketed, expr.arguments)
        return f"({lhs} ** {rhs})"
    elif isinstance(expr, VariableExpr):
        assert isinstance(expr.variable, Variable)
        return expr.variable.name
    else:
        raise NotImplementedError(type(expr))


@pytest.mark.parametrize(
    "expr_string",
    (
        # Check associativity of unary operations
        [f"{op.value} {op.value} a" for op in UnaryOp]
        # Check precedence of unary-vs-pow/binary operations
        + [f"{uo.value} a {bo.value} b" for uo in UnaryOp for bo in BinaryOp]
        + [
            f"a {bo.value} {uo.value} b"
            for uo in UnaryOp
            for bo in BinaryOp
            # This special case is syntactically invalid (in the pseudocode and
            # in Python!)
            if uo != UnaryOp.logical_not
        ]
        + [f"pow({uo.value} a, {uo.value} b)" for uo in UnaryOp]
        + [f"{uo.value} pow(a, b)" for uo in UnaryOp]
        # Check precedence of binary operations
        + [f"a {ao.value} b {bo.value} c" for ao in BinaryOp for bo in BinaryOp]
        + [f"(a {ao.value} b) {bo.value} c" for ao in BinaryOp for bo in BinaryOp]
        + [f"a {ao.value} (b {bo.value} c)" for ao in BinaryOp for bo in BinaryOp]
        # Check precedence of pow-vs-binary operations
        + [f"pow(a, b {op.value} c)" for op in BinaryOp]
        + [f"pow(a {op.value} b, c)" for op in BinaryOp]
        + [f"pow(a, b) {op.value} c" for op in BinaryOp]
        + [f"a {op.value} pow(b, c)" for op in BinaryOp]
        # Check correct output associativity of power function
        + ["pow(pow(a, b), c)", "pow(a, pow(b, c))"]
    ),
)
def test_operator_precedence(expr_string: str) -> None:
    pseudocode = f"""
        foo(a, b, c):
            return {expr_string}
    """
    listing = parse(pseudocode)
    transformer = PythonTransformer(pseudocode)

    # Generate a fully bracketed expression for the pseudocode parse tree
    fn = listing.functions[0]
    assert fn.name == "foo"
    assert isinstance(fn.body[0], ReturnStmt)
    pseudocode_ast = fn.body[0].value
    assert isinstance(pseudocode_ast, Expr)
    # NB: We remove explicit brackets from the parsetree as they are only there
    # for display purposes (the operator grouping should be completely
    # represented by the parse tree itself)
    pseudocode_ast = strip_peren_from_pseudocode_expr(pseudocode_ast)
    pseudocode_bracketed = pseudocode_to_bracketed(pseudocode_ast)

    # Generate a fully bracketed expression for the generated Python's parse tree
    python = transformer.transform(listing)
    python_expr = python.split("\n")[-1].partition("return ")[2]
    python_ast = ast.parse(python_expr, mode="eval")
    python_bracketed = PythonToBracketed().visit(python_ast)

    assert python_bracketed == pseudocode_bracketed


def test_generating_not_on_rhs_of_binary_op() -> None:
    transformer = PythonTransformer(" ")
    listing = Listing(
        [
            Function(
                0,
                "foo",
                [Variable(0, "a"), Variable(0, "b")],
                [
                    ReturnStmt(
                        0,
                        BinaryExpr(
                            VariableExpr(Variable(0, "a")),
                            BinaryOp("+"),
                            UnaryExpr(
                                0, UnaryOp("not"), VariableExpr(Variable(0, "b"))
                            ),
                        ),
                    ),
                ],
            ),
        ]
    )

    # Generate a fully bracketed expression for the generated Python's parse tree
    python = transformer.transform(listing)
    python_expr = python.split("\n")[-1].partition("return ")[2]
    python_ast = ast.parse(python_expr, mode="eval")
    python_bracketed = PythonToBracketed().visit(python_ast)

    assert python_bracketed == "(a + (not b))"


def test_pseudocode_to_python() -> None:
    assert pseudocode_to_python("foo(): return bar()") == "def foo():\n    return bar()"
