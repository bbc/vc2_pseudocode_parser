"""
Transform a parsed AST representation of a pseudocode program into Python

In general, the translation between pseudocode and Python is 'obvious'. The
exceptions are as follows:

* Labels in in the pseudocode are translated into equivalent strings in Python.

The :py:func:`pseudocode_to_python` utility function may be used to directly
translate pseudocode into Python. Alternatively, the lower-level
:py:class:`PythonTransformer` may be used to transform a pseudocode AST.
"""

from typing import List, Iterable, Union, Mapping, Optional, Tuple, cast

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
    ASTNode,
    Listing,
    Function,
    Comment,
    Stmt,
    IfElseStmt,
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
                [UnaryOp(o) for o in ["+", "-", "!"]],
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

    _offset_to_lineno: List[int]
    """For each char in the input source code, the line number it resides on."""

    _defined_names_stack: List[str]
    """
    A stack of names which have been assigned a value in the current scope.

    See :py:meth:`_new_scope`, :py:meth:`_add_name_to_current_scope` and
    :py:meth:`_is_name_in_scope`.

    This information is used to determine whether variable-like entities in the
    pseudocode refer to variables or are labels. In the case of labels, the
    Python translation uses strings instead.
    """

    _unconsumed_comments: List[Tuple[int, Comment]]
    """
    A list of line-numbers and comments which have yet to be output in the
    resulting Python code, in line-order.
    """

    def __init__(self, source: str, indent: str = "    ") -> None:
        self._source = source
        self._indent = indent

        self._offset_to_lineno = []
        for lineno, line in enumerate(source.splitlines(keepends=True)):
            self._offset_to_lineno.extend([lineno] * len(line))

        self._defined_names_stack = []
        self._unconsumed_comments = []

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

    def _adjacent_lines(self, a: ASTNode, b: ASTNode) -> bool:
        """Test if two nodes are on the same line or b is on the line after a."""
        line_a = self._offset_to_lineno[a.offset_end - 1]
        line_b = self._offset_to_lineno[b.offset]
        return line_a == line_b or line_a == line_b - 1

    def _transform_comment_block(
        self, comments: List[Comment], next_node: Optional[ASTNode]
    ) -> str:
        """
        Transform a series of comments into Python comments with normalised
        vertical whitespace. The returned string will either be empty or a
        series of comments, all ending in a newline (including the final
        comment).
        """
        lines = []
        last_comment = None
        for comment in comments:
            if last_comment is not None and not self._adjacent_lines(
                last_comment, comment
            ):
                lines.append("")
            lines.append(comment.string)
            last_comment = comment

        if (
            last_comment is not None
            and next_node is not None
            and not self._adjacent_lines(last_comment, next_node)
        ):
            lines.append("")

        if lines:
            return "\n".join(lines) + "\n"
        else:
            return ""

    def _consume_comments_on_or_before(self, node: ASTNode) -> Tuple[str, str]:
        """
        Consume the comments which appear in the source before and on the same
        line as the (start of) the provided ast element.

        Returns a tuple (comments_before, comment_on_line).

        ``comments_before`` will either be empty or contain a series of comment
        lines, ending with a newline.

        ``comment_on_line`` will either be empty or a comment, proceeded with
        two spaces, which should be appended to the end of the line 'node'
        starts on.
        """
        line = self._offset_to_lineno[node.offset]

        comments_before = []
        while self._unconsumed_comments and self._unconsumed_comments[0][0] < line:
            comments_before.append(self._unconsumed_comments.pop(0)[1])
        comment_block_before = self._transform_comment_block(comments_before, node)

        comment_on_line = ""
        if self._unconsumed_comments and self._unconsumed_comments[0][0] == line:
            comment_on_line = "  " + self._unconsumed_comments.pop(0)[1].string

        return (comment_block_before, comment_on_line)

    def transform(self, listing: Listing) -> str:
        """
        Transform a parsed pseudocode AST into an equivalent Python program.
        """
        self._unconsumed_comments = [
            (self._offset_to_lineno[c.offset], c) for c in listing.comments
        ]

        function_definitions = []
        for function in listing.functions:
            function_definitions.append(self._transform_function(function))
        functions = "\n\n\n".join(function_definitions)

        # Add all trailing comments
        comments = [c for _l, c in self._unconsumed_comments]
        trailing_comments = "\n\n\n" + self._transform_comment_block(comments, None)

        return f"{functions}{trailing_comments.rstrip()}"

    def _transform_function(self, function: Function) -> str:
        name = function.name
        args = ", ".join(v.name for v in function.arguments)

        comments_before, comment_on_line = self._consume_comments_on_or_before(function)

        with self._new_scope():
            for v in function.arguments:
                self._add_name_to_current_scope(v.name)
            body = self._transform_block(function, function.body)

        # Ensure either comment adjacent to function or there's a two line
        # space before function
        if comments_before[-2:] == "\n\n":
            comments_before += "\n"

        return f"{comments_before}def {name}({args}):{comment_on_line}{body}"

    def _transform_block(self, container: ASTNode, body: List[Stmt]) -> str:
        formatted_statements = []
        last_stmt = None
        for stmt in body:
            if last_stmt is not None and not self._adjacent_lines(last_stmt, stmt):
                formatted_statements.append("")
            formatted_statements.append(self._transform_stmt(stmt))
            last_stmt = stmt
        statements = "\n".join(formatted_statements)

        return f"\n{indent(statements, self._indent)}"

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
            comments_before, comment_on_line = self._consume_comments_on_or_before(
                branch
            )
            condition = self._transform_expr(branch.condition)
            body = self._transform_block(branch, branch.body)
            if i == 0:
                if comments_before:
                    if_blocks.append(comments_before[:-1])
                prefix = "if"
            else:
                if comments_before:
                    if_blocks.append(indent(comments_before[:-1], self._indent))
                prefix = "elif"
            if_blocks.append(f"{prefix} {condition}:{comment_on_line}{body}")

        else_block = ""
        if stmt.else_branch is not None:
            comments_before, comment_on_line = self._consume_comments_on_or_before(
                stmt.else_branch
            )
            if comments_before:
                if_blocks.append(indent(comments_before[:-1], self._indent))

            body = self._transform_block(stmt.else_branch, stmt.else_branch.body)
            else_block = f"\nelse:{comment_on_line}{body}"

        return "\n".join(if_blocks) + else_block

    def _transform_for_each_stmt(self, stmt: ForEachStmt) -> str:
        comments_before, comment_on_line = self._consume_comments_on_or_before(stmt)
        variable = stmt.variable.name
        values = ", ".join(self._transform_expr(e) for e in stmt.values)

        self._add_name_to_current_scope(variable)
        body = self._transform_block(stmt, stmt.body)

        return f"{comments_before}for {variable} in [{values}]:{comment_on_line}{body}"

    def _transform_for_stmt(self, stmt: ForStmt) -> str:
        comments_before, comment_on_line = self._consume_comments_on_or_before(stmt)
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

        return f"{comments_before}for {variable} in range({start}{end}):{comment_on_line}{body}"

    def _transform_while_stmt(self, stmt: WhileStmt) -> str:
        comments_before, comment_on_line = self._consume_comments_on_or_before(stmt)
        condition = self._transform_expr(stmt.condition)
        body = self._transform_block(stmt, stmt.body)

        return f"{comments_before}while {condition}:{comment_on_line}{body}"

    def _transform_function_call_stmt(self, stmt: FunctionCallStmt) -> str:
        comments_before, comment_on_line = self._consume_comments_on_or_before(stmt)
        call = self._transform_function_call_expr(stmt.call)
        return f"{comments_before}{call}{comment_on_line}"

    def _transform_return_stmt(self, stmt: ReturnStmt) -> str:
        comments_before, comment_on_line = self._consume_comments_on_or_before(stmt)
        value = self._transform_expr(stmt.value)
        return f"{comments_before}return {value}{comment_on_line}"

    def _transform_assignment_stmt(self, stmt: AssignmentStmt) -> str:
        comments_before, comment_on_line = self._consume_comments_on_or_before(stmt)

        variable = self._transform_variable(stmt.variable)
        op = stmt.op.value
        value = self._transform_expr(stmt.value)

        if isinstance(stmt.variable, Variable):
            self._add_name_to_current_scope(stmt.variable.name)

        if not self._is_name_in_scope(stmt.variable.name):
            raise UndefinedArrayOrMapError.from_variable_expr(
                self._source, stmt.variable
            )

        return f"{comments_before}{variable} {op} {value}{comment_on_line}"

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
        op = expr.op.value if expr.op != UnaryOp("!") else "~"
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
                return repr(expr.variable.name)
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
