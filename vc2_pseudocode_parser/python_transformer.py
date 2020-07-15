'''
The :py:mod:`vc2_pseudocode_parser.python_transformer` module and
``vc2-pseudocode-to-python`` command line tool automatically translate
pseudocode listings into valid Python.

In general, the translation between pseudocode and Python is 'obvious'. The
only non-obvious part, perhaps, is that labels are translated into Python
string literals. The output is pretty-printed in a style similar to
the `Black <https://github.com/psf/black>`_ code style with comments and
vertical whitespace retained (in a semi-normalised fashion).

For example, the following pseudocode::

    add(a, b, c):
        # A function which adds three numbers together
        total = 0  # An accumulator
        for each n in a, b, c:
            total += n
        return total

    update_state(state):
        state[count] += 1

Is translated in the the following Python::

    def add(a, b, c):
        """
        A function which adds three numbers together
        """
        total = 0  # An accumulator
        for n in [a, b, c]:
            total += n
        return total


    def update_state(state):
        state["count"] += 1



Command-line utility
====================

The ``vc2-pseudocode-to-python`` command line utility is provided which can
convert a pseudocode listing into Python.

Example usage::

    $ vc2-pseudocode-to-python input.pc output.py


Python API
==========

The :py:func:`pseudocode_to_python` utility function may be used to directly
translate pseudocode into Python.

.. autofunction:: pseudocode_to_python

Example usage::

    >>> from vc2_pseudocode_parser.python_transformer import pseudocode_to_python

    >>> print(pseudocode_to_python("""
    ...     foo(state, a):
    ...         state[bar] = a + 1
    ... """))
    def foo(state, a):
        state["bar"] = a + 1

'''

from typing import List, Iterable, Union, Mapping, Tuple, Optional, cast

from textwrap import indent, dedent

from itertools import chain

from vc2_pseudocode_parser.parser.parser import parse

from vc2_pseudocode_parser.parser.operators import BinaryOp, UnaryOp, Associativity

from vc2_pseudocode_parser.parser.ast import (
    Listing,
    Function,
    Stmt,
    IfElseStmt,
    IfBranch,
    ElseBranch,
    ForEachStmt,
    ForStmt,
    WhileStmt,
    FunctionCallStmt,
    ReturnStmt,
    AssignmentStmt,
    Expr,
    FunctionCallExpr,
    ParenExpr,
    UnaryExpr,
    BinaryExpr,
    VariableExpr,
    Variable,
    Subscript,
    LabelExpr,
    Label,
    EmptyMapExpr,
    BooleanExpr,
    NumberExpr,
    EOL,
    EmptyLine,
)


PYTHON_OPERATOR_PRECEDENCE_TABLE: Mapping[Union[BinaryOp, UnaryOp], int] = {
    op: score
    for score, ops in enumerate(
        reversed(
            [
                # Shown in high-to-low order
                [BinaryOp(o) for o in ["**"]],
                [UnaryOp(o) for o in ["+", "-", "~"]],
                [BinaryOp(o) for o in ["*", "//", "%"]],
                [BinaryOp(o) for o in ["+", "-"]],
                [BinaryOp(o) for o in ["<<", ">>"]],
                [BinaryOp(o) for o in ["&"]],
                [BinaryOp(o) for o in ["^"]],
                [BinaryOp(o) for o in ["|"]],
                [BinaryOp(o) for o in ["==", "!=", "<=", ">=", "<", ">"]],
                [UnaryOp(o) for o in ["not"]],
                [BinaryOp(o) for o in ["and"]],
                [BinaryOp(o) for o in ["or"]],
            ]
        )
    )
    for op in cast(Iterable[Union[BinaryOp, UnaryOp]], ops)
}
"""
Lookup giving a precedence score for each operator. A higher score means
higher precedence.
"""

PYTHON_OPERATOR_ASSOCIATIVITY_TABLE: Mapping[
    Union[BinaryOp, UnaryOp], Associativity
] = dict(
    chain(
        [(op, Associativity.left) for op in BinaryOp if op != BinaryOp("**")],
        [(BinaryOp("**"), Associativity.right)],
        [(op, Associativity.right) for op in UnaryOp],
    )
)
"""Lookup giving operator associativity for each operator."""


def split_trailing_comments(block: str) -> Tuple[str, str]:
    """
    Split a source block into the original source code and any trailing
    whitespace, including the newline which divides the source from the
    trailer.
    """
    # Find trailing comment-only/blank lines
    lines = block.split("\n")
    first_trailing_line = len(lines)
    for i in reversed(range(len(lines))):
        dedented = lines[i].lstrip()
        if dedented.strip() == "" or dedented.startswith("#"):
            first_trailing_line = i
        else:
            break

    before = "\n".join(lines[:first_trailing_line])

    after = "\n".join(lines[first_trailing_line:])
    if 1 <= first_trailing_line < len(lines):
        after = "\n" + after

    return before, after


def dedent_trailing_comments(block: str) -> str:
    """
    De-indent any trailing comments in a Python code block.
    """
    before, after = split_trailing_comments(block)

    return before + "\n".join(s.lstrip() for s in after.split("\n"))


def remove_prefix_from_comment_block(block: str) -> str:
    """
    Remove the `#` prefix from a comment block, along with any common
    indentation.
    """
    return dedent("\n".join([line[1:] for line in block.split("\n")]))


def expr_add_one(expr: Expr) -> Expr:
    """
    Return an expression equivalent to the one provided where the equivalent
    value has had one subtracted from it.
    """
    if isinstance(expr, NumberExpr) and expr.display_base == 10:
        return NumberExpr(expr.offset, expr.offset_end, expr.value + 1)
    elif (
        isinstance(expr, BinaryExpr)
        and expr.op == BinaryOp("+")
        and isinstance(expr.rhs, NumberExpr)
        and expr.rhs.display_base == 10
    ):
        return BinaryExpr(
            expr.lhs,
            expr.op,
            NumberExpr(expr.rhs.offset, expr.rhs.offset_end, expr.rhs.value + 1),
        )
    elif (
        isinstance(expr, BinaryExpr)
        and expr.op == BinaryOp("-")
        and isinstance(expr.rhs, NumberExpr)
        and expr.rhs.display_base == 10
    ):
        if expr.rhs.value == 1:
            return expr.lhs
        else:
            return BinaryExpr(
                expr.lhs,
                expr.op,
                NumberExpr(expr.rhs.offset, expr.rhs.offset_end, expr.rhs.value - 1),
            )
    else:
        return BinaryExpr(expr, BinaryOp("+"), NumberExpr(expr.offset, expr.offset, 1))


class PythonTransformer:
    """
    A transformer which transforms from a pseudocode AST into equivalent Python
    code.

    Once constructed, a Python translation of a parsed pseudocode listing may
    be produced using the :py:meth:`transform`.

    Parameters
    ----------
    source: str
        The pseudocode source code (used to produce error messages).
    indent: str
        If provided, the string to use for block indentation. Defaults to four
        spaces.
    generate_docstrings: bool
        If True, the first block of comments in the file and each function will
        be converted into a docstring. Otherwise they'll be left as ordinary
        comments.
    add_translation_note: bool
        If True, adds a comment to the top of the generated output indicating
        that this file was automatically translated from the pseudocode.
    """

    _source: str
    """The input source code (used for error message generation."""

    _indent: str
    """String to use to indent blocks."""

    _generate_docstrings: bool
    """If True, turn opening comment blocks into docstrings. The default is False."""

    _add_translation_note: bool
    """
    Add a comment to the top of the generated file indicating that it has been
    auto-generated.
    """

    def __init__(
        self,
        source: str,
        indent: str = "    ",
        generate_docstrings: bool = True,
        add_translation_note: bool = False,
    ) -> None:
        self._source = source
        self._indent = indent
        self._generate_docstrings = generate_docstrings
        self._add_translation_note = add_translation_note

    def transform(self, listing: Listing) -> str:
        """
        Transform a parsed pseudocode AST into an equivalent Python program.
        """
        function_definitions = []
        for function in listing.functions:
            fdef, comments = split_trailing_comments(self._transform_function(function))

            if comments.strip() == "":
                comments = "\n\n\n"
            else:
                # Force a 3-line gap before the trailing comments
                comments = "\n\n\n" + comments.lstrip()

                # If comments end with whitespace, expand that to two empty
                # lines
                if comments[-1] == "\n":
                    comments += "\n\n"
                else:
                    comments += "\n"

            function_definitions.append(fdef + comments)

        functions = "".join(function_definitions).rstrip("\n")

        leading_comments = self._transform_empty_lines(
            listing.leading_empty_lines, make_first_comment_block_into_docstring=True,
        ).lstrip()
        if leading_comments:
            # Force a 3-line gap if an empty line has been left
            if leading_comments[-1] == "\n":
                leading_comments += "\n\n"
            else:
                leading_comments += "\n"

        note = ""
        if self._add_translation_note:
            note = "# This file was automatically translated from a pseudocode listing.\n\n"

        return note + leading_comments + functions

    def _transform_function(self, function: Function) -> str:
        name = function.name
        args = ", ".join(v.name for v in function.arguments)
        body = self._transform_block(function, function.body, True)
        return f"def {name}({args}):{body}"

    def _transform_block(
        self,
        container: Union[
            Function, IfBranch, ElseBranch, ForEachStmt, ForStmt, WhileStmt
        ],
        body: List[Stmt],
        make_first_comment_block_into_docstring: bool = False,
    ) -> str:
        eol = (
            self._transform_eol(
                container.eol,
                self._indent,
                True,
                make_first_comment_block_into_docstring,
            )
            if container.eol
            else ""
        )

        formatted_statements = []
        for stmt in body:
            formatted_statements.append(self._transform_stmt(stmt))
        statements = dedent_trailing_comments(
            indent("\n".join(formatted_statements), self._indent)
        )

        return f"{eol}\n{statements}"

    def _transform_eol(
        self,
        eol: EOL,
        following_indentation: str = "",
        strip_empty_leading_lines: bool = False,
        make_first_comment_block_into_docstring: bool = False,
    ) -> str:
        comment = f"  {eol.comment.string}" if eol.comment is not None else ""

        empty_lines = indent(
            self._transform_empty_lines(
                eol.empty_lines,
                strip_empty_leading_lines,
                make_first_comment_block_into_docstring,
            ),
            following_indentation,
        )

        return f"{comment}{empty_lines}"

    def _transform_empty_lines(
        self,
        empty_lines: List[EmptyLine],
        strip_empty_leading_lines: bool = False,
        make_first_comment_block_into_docstring: bool = False,
    ) -> str:
        out_lines = ["\n"] if strip_empty_leading_lines else []
        for empty_line in empty_lines:
            if empty_line.comment is None:
                # Normalise vertical spacing to allow at most one consecutive
                # empty line without a comment.
                if not out_lines or out_lines[-1] != "\n":
                    out_lines.append("\n")
            else:
                out_lines.append(f"\n{empty_line.comment.string}")

        if strip_empty_leading_lines:
            out_lines.pop(0)

        if self._generate_docstrings and make_first_comment_block_into_docstring:
            last_comment_line: Optional[int] = None
            for i, line in enumerate(out_lines):
                if line.startswith("\n#"):
                    last_comment_line = i
                else:
                    break

            if last_comment_line is not None:
                comment_block = "".join(out_lines[: last_comment_line + 1])[1:]
                comment_block = remove_prefix_from_comment_block(comment_block)
                out_lines[: last_comment_line + 1] = [
                    f"\n{line}" for line in comment_block.split("\n")
                ]
                out_lines.insert(last_comment_line + 1, '\n"""')
                out_lines.insert(0, '\n"""')

        return "".join(out_lines)

    def _transform_stmt(self, stmt: Stmt) -> str:
        if isinstance(stmt, IfElseStmt):
            return self._transform_if_else_stmt(stmt)
        elif isinstance(stmt, ForEachStmt):
            return self._transform_for_each_stmt(stmt)
        elif isinstance(stmt, ForStmt):
            return self._transform_for_stmt(stmt)
        elif isinstance(stmt, WhileStmt):
            return self._transform_while_stmt(stmt)
        elif isinstance(stmt, FunctionCallStmt):
            return self._transform_function_call_stmt(stmt)
        elif isinstance(stmt, ReturnStmt):
            return self._transform_return_stmt(stmt)
        elif isinstance(stmt, AssignmentStmt):
            return self._transform_assignment_stmt(stmt)
        else:
            raise TypeError(type(stmt))  # Unreachable

    def _transform_if_else_stmt(self, stmt: IfElseStmt) -> str:
        if_blocks = []
        for i, branch in enumerate(stmt.if_branches):
            condition = self._transform_expr(branch.condition)
            body = self._transform_block(branch, branch.body)
            if i == 0:
                prefix = "if"
            else:
                prefix = "elif"
            if_blocks.append(f"{prefix} {condition}:{body}")

        else_block = ""
        if stmt.else_branch is not None:
            body = self._transform_block(stmt.else_branch, stmt.else_branch.body)
            else_block = f"\nelse:{body}"

        return "\n".join(if_blocks) + else_block

    def _transform_for_each_stmt(self, stmt: ForEachStmt) -> str:
        variable = stmt.variable.name
        values = ", ".join(self._transform_expr(e) for e in stmt.values)
        body = self._transform_block(stmt, stmt.body)
        return f"for {variable} in [{values}]:{body}"

    def _transform_for_stmt(self, stmt: ForStmt) -> str:
        variable = stmt.variable.name

        if (
            isinstance(stmt.start, NumberExpr)
            and stmt.start.value == 0
            and stmt.start.display_base == 10
        ):
            start = ""
        else:
            start = f"{self._transform_expr(stmt.start)}, "

        end = self._transform_expr(expr_add_one(stmt.end))

        body = self._transform_block(stmt, stmt.body)

        return f"for {variable} in range({start}{end}):{body}"

    def _transform_while_stmt(self, stmt: WhileStmt) -> str:
        condition = self._transform_expr(stmt.condition)
        body = self._transform_block(stmt, stmt.body)

        return f"while {condition}:{body}"

    def _transform_function_call_stmt(self, stmt: FunctionCallStmt) -> str:
        call = self._transform_function_call_expr(stmt.call)
        eol = self._transform_eol(stmt.eol)
        return f"{call}{eol}"

    def _transform_return_stmt(self, stmt: ReturnStmt) -> str:
        value = self._transform_expr(stmt.value)
        eol = self._transform_eol(stmt.eol)
        return f"return {value}{eol}"

    def _transform_assignment_stmt(self, stmt: AssignmentStmt) -> str:
        variable = self._transform_variable(stmt.variable)
        op = stmt.op.value
        value = self._transform_expr(stmt.value)
        eol = self._transform_eol(stmt.eol)

        return f"{variable} {op} {value}{eol}"

    def _transform_expr(self, expr: Expr) -> str:
        if isinstance(expr, FunctionCallExpr):
            return self._transform_function_call_expr(expr)
        elif isinstance(expr, ParenExpr):
            return self._transform_paren_expr(expr)
        elif isinstance(expr, UnaryExpr):
            return self._transform_unary_expr(expr)
        elif isinstance(expr, BinaryExpr):
            return self._transform_binary_expr(expr)
        elif isinstance(expr, VariableExpr):
            return self._transform_variable_expr(expr)
        elif isinstance(expr, LabelExpr):
            return self._transform_label_expr(expr)
        elif isinstance(expr, EmptyMapExpr):
            return self._transform_empty_map_expr(expr)
        elif isinstance(expr, BooleanExpr):
            return self._transform_boolean_expr(expr)
        elif isinstance(expr, NumberExpr):
            return self._transform_number_expr(expr)
        else:
            raise TypeError(type(expr))  # Unreachable

    def _transform_paren_expr(self, expr: ParenExpr) -> str:
        value = self._transform_expr(expr.value)
        return f"({value})"

    def _transform_unary_expr(self, expr: UnaryExpr) -> str:
        op = expr.op.value
        space = " " if op == "not" else ""
        value = self._transform_expr(expr.value)
        if isinstance(expr.value, (BinaryExpr, UnaryExpr)) and (
            PYTHON_OPERATOR_PRECEDENCE_TABLE[expr.value.op]
            < PYTHON_OPERATOR_PRECEDENCE_TABLE[expr.op]
            or expr.value.op == BinaryOp.pow
        ):
            # We also add brackets for exponentiation (**), even though it has
            # higher precedence all other operators and therefore this isn't
            # required. This is a standard Python convention which aids
            # readability.
            return f"{op}{space}({value})"
        else:
            return f"{op}{space}{value}"

    def _transform_binary_expr(self, expr: BinaryExpr) -> str:
        lhs = self._transform_expr(expr.lhs)
        op = expr.op
        associativity = PYTHON_OPERATOR_ASSOCIATIVITY_TABLE[op]
        rhs = self._transform_expr(expr.rhs)

        # Decide parentheses for LHS
        if isinstance(expr.lhs, (BinaryExpr, UnaryExpr)):
            if associativity == Associativity.left and (
                PYTHON_OPERATOR_PRECEDENCE_TABLE[expr.lhs.op]
                < PYTHON_OPERATOR_PRECEDENCE_TABLE[expr.op]
            ):
                lhs = f"({lhs})"
            elif associativity == Associativity.right and (
                PYTHON_OPERATOR_PRECEDENCE_TABLE[expr.lhs.op]
                <= PYTHON_OPERATOR_PRECEDENCE_TABLE[expr.op]
            ):
                lhs = f"({lhs})"

        # Decide parentheses for RHS
        if isinstance(expr.rhs, (BinaryExpr, UnaryExpr)):
            if associativity == Associativity.left and (
                PYTHON_OPERATOR_PRECEDENCE_TABLE[expr.rhs.op]
                <= PYTHON_OPERATOR_PRECEDENCE_TABLE[expr.op]
            ):
                rhs = f"({rhs})"
            elif associativity == Associativity.right and (
                PYTHON_OPERATOR_PRECEDENCE_TABLE[expr.rhs.op]
                < PYTHON_OPERATOR_PRECEDENCE_TABLE[expr.op]
            ):
                rhs = f"({rhs})"

        return f"{lhs} {op.value} {rhs}"

    def _transform_function_call_expr(self, expr: FunctionCallExpr) -> str:
        name = expr.name
        args = ", ".join(self._transform_expr(a) for a in expr.arguments)
        return f"{name}({args})"

    def _transform_variable_expr(self, expr: VariableExpr) -> str:
        return self._transform_variable(expr.variable)

    def _transform_variable(self, expr: Union[Variable, Subscript]) -> str:
        if isinstance(expr, Variable):
            return expr.name
        elif isinstance(expr, Subscript):
            variable = self._transform_variable(expr.variable)
            subscript = self._transform_expr(expr.subscript)
            return f"{variable}[{subscript}]"
        else:
            raise TypeError(type(expr))  # Unreachable

    def _transform_label_expr(self, expr: LabelExpr) -> str:
        return self._transform_label(expr.label)

    def _transform_label(self, label: Label) -> str:
        return f'"{label.name}"'

    def _transform_empty_map_expr(self, expr: EmptyMapExpr) -> str:
        return "{}"

    def _transform_boolean_expr(self, expr: BooleanExpr) -> str:
        return "True" if expr.value else "False"

    def _transform_number_expr(self, expr: NumberExpr) -> str:
        if expr.display_base == 10:
            return str(expr.value)
        elif expr.display_base == 2:
            return "0b{:0{}b}".format(expr.value, expr.display_digits)
        elif expr.display_base == 16:
            return "0x{:0{}X}".format(expr.value, expr.display_digits)
        else:
            raise TypeError(expr.display_base)


def pseudocode_to_python(
    pseudocode_source: str,
    indent: str = "    ",
    generate_docstrings: bool = True,
    add_translation_note: bool = False,
) -> str:
    """
    Transform a pseudocode listing into Python.

    Will throw a :py:exc:`~vc2_pseudocode_parser.parser.ParseError`
    or :py:exc:`.ASTConstructionError` if the supplied pseudocode contains
    syntactic errors.

    Parameters
    ==========
    pseudocode_source : str
        The pseudocode source code to translate.
    indent : str
        The string to use for indentation in the generated Python source.
        Defaults to four spaces.
    generate_docstrings : bool
        If True, the first block of comments in the file and each function will
        be converted into a docstring. Otherwise they'll be left as ordinary
        comments. Defaulse to True.
    add_translation_note : bool
        If True, adds a comment to the top of the generated output indicating
        that this file was automatically translated from the pseudocode.
        Defaulse to False.
    """
    pseudocode_ast = parse(pseudocode_source)
    transformer = PythonTransformer(
        pseudocode_source, indent, generate_docstrings, add_translation_note
    )
    python_source = transformer.transform(pseudocode_ast)
    return python_source
