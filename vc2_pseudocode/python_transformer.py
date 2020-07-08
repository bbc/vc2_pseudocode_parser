"""
Transform a parsed AST representation of a pseudocode program into Python

In general, the translation between pseudocode and Python is 'obvious'. The
exceptions are as follows:

* Labels in in the pseudocode are translated into equivalent strings in Python.

The :py:func:`pseudocode_to_python` utility function may be used to directly
translate pseudocode into Python. Alternatively, the lower-level
:py:class:`PythonTransformer` may be used to transform a pseudocode AST.
"""

from typing import List, Iterable, Union, Mapping, Tuple, cast

from textwrap import indent

from contextlib import contextmanager

from dataclasses import dataclass

from itertools import chain

from vc2_pseudocode.parser import parse

from vc2_pseudocode.operators import BinaryOp, UnaryOp, Associativity

from peggie.error_message_generation import (
    offset_to_line_and_column,
    extract_line,
    format_error_message,
)

from vc2_pseudocode.ast import (
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
    PerenExpr,
    UnaryExpr,
    BinaryExpr,
    VariableExpr,
    Variable,
    Subscript,
    EmptyMapExpr,
    BooleanExpr,
    NumberExpr,
    EOL,
    EmptyLine,
)


@dataclass
class PythonTransformationError(Exception):
    line: int
    column: int
    snippet: str

    @property
    def explanation(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        return format_error_message(
            self.line, self.column, self.snippet, self.explanation
        )


@dataclass
class UndefinedArrayOrMapError(PythonTransformationError):
    variable_name: str

    @classmethod
    def from_variable_expr(
        cls, source: str, variable: Union[Variable, Subscript]
    ) -> "UndefinedArrayOrMapError":
        """
        Create an :py:class:`UndefinedArrayOrMapError` for the variable or subscript
        provided.
        """
        line, column = offset_to_line_and_column(source, variable.offset)
        snippet = extract_line(source, line)

        return cls(line, column, snippet, variable.name)

    @property
    def explanation(self) -> str:
        return f"Map or array '{self.variable_name}' not defined."


@dataclass
class VariableCalledAsFunctionError(PythonTransformationError):
    function_name: str

    @classmethod
    def from_function_call_expr(
        cls, source: str, call: FunctionCallExpr
    ) -> "VariableCalledAsFunctionError":
        """
        Create an :py:class:`VariableCalledAsFunctionError` for the function call
        provided.
        """
        line, column = offset_to_line_and_column(source, call.offset)
        snippet = extract_line(source, line)
        return cls(line, column, snippet, call.name)

    @property
    def explanation(self) -> str:
        return f"Attempted to call variable {self.function_name} as a function."


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
    """

    _source: str
    """The input source code (used for error message generation."""

    _indent: str
    """String to use to indent blocks."""

    _defined_names_stack: List[str]
    """
    A stack of names which have been assigned a value in the current scope.

    See :py:meth:`_new_scope`, :py:meth:`_add_name_to_current_scope` and
    :py:meth:`_is_name_in_scope`.

    This information is used to determine whether variable-like entities in the
    pseudocode refer to variables or are labels. In the case of labels, the
    Python translation uses strings instead.
    """

    def __init__(self, source: str, indent: str = "    ") -> None:
        self._source = source
        self._indent = indent

        self._defined_names_stack = []

    @contextmanager  # type: ignore
    def _new_scope(self) -> Iterable[None]:
        """
        Context manager which creates a nested scope in which names can be
        defined.
        """
        old_len = len(self._defined_names_stack)
        try:
            yield
        finally:
            del self._defined_names_stack[old_len:]

    def _add_name_to_current_scope(self, name: str) -> None:
        """Add a new name to the current scope."""
        self._defined_names_stack.append(name)

    def _is_name_in_scope(self, name: str) -> bool:
        """Check if a name is in the current scope."""
        return name in self._defined_names_stack

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
            listing.leading_empty_lines
        ).lstrip()
        if leading_comments:
            # Force a 3-line gap if an empty line has been left
            if leading_comments[-1] == "\n":
                leading_comments += "\n\n"
            else:
                leading_comments += "\n"

        return leading_comments + functions

    def _transform_function(self, function: Function) -> str:
        name = function.name
        args = ", ".join(v.name for v in function.arguments)

        with self._new_scope():
            for v in function.arguments:
                self._add_name_to_current_scope(v.name)
            body = self._transform_block(function, function.body)

        return f"def {name}({args}):{body}"

    def _transform_block(
        self,
        container: Union[
            Function, IfBranch, ElseBranch, ForEachStmt, ForStmt, WhileStmt
        ],
        body: List[Stmt],
    ) -> str:
        eol = (
            self._transform_eol(container.eol, self._indent, True)
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
    ) -> str:
        comment = f"  {eol.comment.string}" if eol.comment is not None else ""

        empty_lines = indent(
            self._transform_empty_lines(eol.empty_lines, strip_empty_leading_lines),
            following_indentation,
        )

        return f"{comment}{empty_lines}"

    def _transform_empty_lines(
        self, empty_lines: List[EmptyLine], strip_empty_leading_lines: bool = False,
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

        self._add_name_to_current_scope(variable)
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

        self._add_name_to_current_scope(variable)
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

        if isinstance(stmt.variable, Variable):
            self._add_name_to_current_scope(stmt.variable.name)

        if not self._is_name_in_scope(stmt.variable.name):
            raise UndefinedArrayOrMapError.from_variable_expr(
                self._source, stmt.variable
            )

        return f"{variable} {op} {value}{eol}"

    def _transform_expr(self, expr: Expr) -> str:
        if isinstance(expr, FunctionCallExpr):
            return self._transform_function_call_expr(expr)
        elif isinstance(expr, PerenExpr):
            return self._transform_peren_expr(expr)
        elif isinstance(expr, UnaryExpr):
            return self._transform_unary_expr(expr)
        elif isinstance(expr, BinaryExpr):
            return self._transform_binary_expr(expr)
        elif isinstance(expr, VariableExpr):
            return self._transform_variable_expr(expr)
        elif isinstance(expr, EmptyMapExpr):
            return self._transform_empty_map_expr(expr)
        elif isinstance(expr, BooleanExpr):
            return self._transform_boolean_expr(expr)
        elif isinstance(expr, NumberExpr):
            return self._transform_number_expr(expr)
        else:
            raise TypeError(type(expr))  # Unreachable

    def _transform_peren_expr(self, expr: PerenExpr) -> str:
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

        # Decide perentheses for LHS
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

        # Decide perentheses for RHS
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

        if self._is_name_in_scope(name):
            raise VariableCalledAsFunctionError.from_function_call_expr(
                self._source, expr
            )

        args = ", ".join(self._transform_expr(a) for a in expr.arguments)
        return f"{name}({args})"

    def _transform_variable_expr(self, expr: VariableExpr) -> str:
        if self._is_name_in_scope(expr.variable.name):
            return self._transform_variable(expr.variable)
        else:
            # The variable name being used has not been defined in the current
            # scope, as a result it must be a label (which we'll translate to a
            # string).
            if isinstance(expr.variable, Variable):
                return f'"{expr.variable.name}"'
            elif isinstance(expr.variable, Subscript):
                # Subscripting an undefined variable is not allowed
                variable: Union[Variable, Subscript] = expr.variable
                while not isinstance(variable, Variable):
                    variable = variable.variable
                raise UndefinedArrayOrMapError.from_variable_expr(
                    self._source, variable
                )
            else:
                raise TypeError(type(expr.variable))  # Unreachable

    def _transform_variable(self, expr: Union[Variable, Subscript]) -> str:
        if isinstance(expr, Variable):
            return expr.name
        elif isinstance(expr, Subscript):
            variable = self._transform_variable(expr.variable)
            subscript = self._transform_expr(expr.subscript)
            return f"{variable}[{subscript}]"
        else:
            raise TypeError(type(expr))  # Unreachable

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


def pseudocode_to_python(pseudocode_source: str) -> str:
    """
    Transform a pseudocode listing into Python.

    Will throw a :py:exc:`vc2_pseudocode.parser.ParseError` if the supplied
    pseudocode contains syntactic errors and
    :py:exc:`PythonTransformationError` if the pseudocode cannot be transformed
    into Python.
    """
    pseudocode_ast = parse(pseudocode_source)
    transformer = PythonTransformer(pseudocode_source)
    python_source = transformer.transform(pseudocode_ast)
    return python_source
