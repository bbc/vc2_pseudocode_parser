"""
Abstract Syntax Tree (AST) data structures for the VC-2 specification pseudocode language.
"""

from typing import List, Union, Optional, Any, cast, Tuple, Set, Sequence

from peggie import (
    ParseTree,
    Alt,
    Regex,
    Lookahead,
    ParseTreeTransformer,
)

from peggie.error_message_generation import (
    offset_to_line_and_column,
    extract_line,
    format_error_message,
)

from vc2_pseudocode_parser.pseudocode_parser.operators import (
    BinaryOp,
    UnaryOp,
    AssignmentOp,
    OPERATOR_ASSOCIATIVITY_TABLE,
    Associativity,
)

from dataclasses import dataclass, field


__all__ = [
    "ASTNode",
    "Listing",
    "Comment",
    "EmptyLine",
    "EOL",
    "Function",
    "Stmt",
    "IfElseStmt",
    "IfBranch",
    "ElseBranch",
    "ForEachStmt",
    "ForStmt",
    "WhileStmt",
    "FunctionCallStmt",
    "ReturnStmt",
    "AssignmentStmt",
    "Variable",
    "Subscript",
    "Label",
    "Expr",
    "PerenExpr",
    "UnaryExpr",
    "BinaryExpr",
    "FunctionCallExpr",
    "VariableExpr",
    "LabelExpr",
    "EmptyMapExpr",
    "BooleanExpr",
    "NumberExpr",
    "ASTConstructionError",
    "LabelUsedAsVariableNameError",
    "CannotSubscriptLabelError",
]


@dataclass
class ASTNode:
    offset: int
    """Index of first character in the source related to this node."""

    offset_end: int
    """Index of the character after the final character related to this node."""


@dataclass
class Listing(ASTNode):
    """The root of a pseudocode AST."""

    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)

    functions: List["Function"]
    """
    List of :py:class:`Function` trees for each function defined in the tree.
    """

    leading_empty_lines: List["EmptyLine"] = field(default_factory=list)
    """
    List of :py:class:`EmptyLine` trees relating to empty (or comment-only)
    lines at the start of the source listing.
    """

    def __post_init__(self) -> None:
        self.offset = self.functions[0].offset
        self.offset_end = self.functions[-1].offset_end


@dataclass
class Comment(ASTNode):
    """An end-of-line comment."""

    offset_end: int = field(init=False, repr=False)
    string: str
    """The comment string, including leading '#' but not trailing newline."""

    def __post_init__(self) -> None:
        self.offset_end = self.offset + len(self.string)


@dataclass
class EmptyLine(ASTNode):
    """Represents an empty (or comment-only) line in the source."""

    comment: Optional[Comment] = None
    """The :py:class:`Comment` on this line, if present."""


@dataclass
class EOL(ASTNode):
    """The end-of-line terminator following a statement."""

    comment: Optional[Comment] = None
    """A :py:class:`Comment` on the same line as the statement, if present."""

    empty_lines: List[EmptyLine] = field(default_factory=list)
    """Any trailing :py:class:`EmptyLine` after this statement."""


@dataclass
class Function(ASTNode):
    """
    A function definition::

        <name>(<arguments[0]>, <arguments[1]>, <...>): <eol>
            <body>
    """

    offset_end: int = field(init=False, repr=False)

    name: str
    """The name of the function."""

    arguments: List["Variable"]
    """
    The :py:class:`Variable` objects corresponding with the arguments to this
    function.
    """

    body: List["Stmt"]
    """The list of :py:class:`Stmt` which make up this function."""

    eol: Optional[EOL] = None
    """:py:class:`EOL` at the end of the function heading."""

    def __post_init__(self) -> None:
        self.offset_end = self.body[-1].offset_end


@dataclass
class Stmt(ASTNode):
    """
    Base-class for all statement AST nodes.
    """

    pass


@dataclass
class IfElseStmt(Stmt):
    r"""
    An ``if`` statement with an arbitrary number of ``else if`` clauses and
    optional ``else`` clause.

    A ``if`` statement is broken as illustrated by the following example::

        if (condition0):       # \ if_branches[0]
            body0              # /
        else if (condition1):  # \ if_branches[1]
            body1              # /
        else if (condition2):  # \ if_branches[2]
            body2              # /
        else:                  # \ else_branch
            body3              # /
    """

    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)

    if_branches: List["IfBranch"]
    """
    The opening ``if`` clause followed by any ``else if`` clauses are
    represented as a series of :py:class:`IfBranch`.
    """

    else_branch: Optional["ElseBranch"] = None
    """
    If an ``else`` clause is present, a :py:class:`ElseBranch` giving its
    contents.
    """

    def __post_init__(self) -> None:
        self.offset = self.if_branches[0].offset
        if self.else_branch is not None:
            self.offset_end = self.else_branch.offset_end
        else:
            self.offset_end = self.if_branches[-1].offset_end


@dataclass
class IfBranch(ASTNode):
    """
    An ``if`` or ``else if`` clause in an if-else-if-else statement::

        if (<condition>): <eol>
            <body>

    Or::

        else if (<condition>): <eol>
            <body>
    """

    offset_end: int = field(init=False, repr=False)

    condition: "Expr"
    """The :py:class:`Expr` representing the condition for the branch condition."""

    body: List[Stmt]
    """The list of :py:class:`Stmt` to execute when the condition is True."""

    eol: Optional[EOL] = None
    """:py:class:`EOL` following the ``if`` or ``else if`` clause heading."""

    def __post_init__(self) -> None:
        self.offset_end = self.body[-1].offset_end


@dataclass
class ElseBranch(ASTNode):
    """
    An ``else`` clause in an if-else-if-else statement::

        else: <eol>
            <body>
    """

    offset_end: int = field(init=False, repr=False)

    body: List[Stmt]
    """The list of :py:class:`Stmt` to execute when the else branch is reached."""

    eol: Optional[EOL] = None
    """:py:class:`EOL` following the ``else`` clause heading."""

    def __post_init__(self) -> None:
        self.offset_end = self.body[-1].offset_end


@dataclass
class ForEachStmt(Stmt):
    """
    A ``for each`` loop::

        for each <variable> in <values[0]>, <values[1]>, <...>: <eol>
            <body>
    """

    offset_end: int = field(init=False, repr=False)

    variable: "Variable"
    """The loop :py:class:`Variable`."""

    values: List["Expr"]
    """The :py:class:`Expr` giving the set of values the loop will iterate over."""

    body: List[Stmt]
    """The list of :py:class:`Stmt` to execute in each iteration."""

    eol: Optional[EOL] = None
    """:py:class:`EOL` following the ``for each`` heading."""

    def __post_init__(self) -> None:
        self.offset_end = self.body[-1].offset_end


@dataclass
class ForStmt(Stmt):
    """
    A ``for`` loop::

        for <variable> = <start> to <end>: <eol>
            <body>
    """

    offset_end: int = field(init=False, repr=False)

    variable: "Variable"
    """The loop :py:class:`Variable`."""

    start: "Expr"
    """The :py:class:`Expr` giving the loop starting value."""

    end: "Expr"
    """The :py:class:`Expr` giving the (inclusive) loop ending value."""

    body: List[Stmt]
    """The list of :py:class:`Stmt` to execute in each iteration."""

    eol: Optional[EOL] = None
    """:py:class:`EOL` following the ``for`` heading."""

    def __post_init__(self) -> None:
        self.offset_end = self.body[-1].offset_end


@dataclass
class WhileStmt(Stmt):
    """
    A ``while`` loop::

        while (<condition>): <eol>
            <body>
    """

    offset_end: int = field(init=False, repr=False)

    condition: "Expr"
    """The :py:class:`Expr` representing the loop condition."""

    body: List[Stmt]
    """The list of :py:class:`Stmt` to execute in each iteration."""

    eol: Optional[EOL] = None
    """:py:class:`EOL` following the ``while`` heading."""

    def __post_init__(self) -> None:
        self.offset_end = self.body[-1].offset_end


@dataclass
class FunctionCallStmt(Stmt):
    """A statement which represents a call to a function."""

    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)

    call: "FunctionCallExpr"
    """The :py:class:`FunctionCallExpr` which defines the function call."""

    eol: EOL
    """:py:class:`EOL` following the function call."""

    def __post_init__(self) -> None:
        self.offset = self.call.offset
        self.offset_end = self.call.offset_end


@dataclass
class ReturnStmt(Stmt):
    """
    A return statement::

        return <value> <eol>
    """

    offset_end: int = field(init=False, repr=False)

    value: "Expr"
    """A :py:class:`Expr` giving the value to be returned."""

    eol: EOL
    """:py:class:`EOL` following the return statement."""

    def __post_init__(self) -> None:
        self.offset_end = self.value.offset_end


@dataclass
class AssignmentStmt(Stmt):
    """
    A simple or compound assignment statement::

        <variable> <op> <value> <eol>  # e.g. x += 1 + 2
    """

    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)

    variable: Union["Variable", "Subscript"]
    """The :py:class:`Variable` or :py:class:`Subscript` being assigned to."""

    op: AssignmentOp
    """The type of assignment being performed (:py:class:`AssignmentOp`)."""

    value: "Expr"
    """The :py:class:`Expr` giving the value being assigned."""

    eol: EOL
    """:py:class:`EOL` following the assignment statement."""

    def __post_init__(self) -> None:
        self.offset = self.variable.offset
        self.offset_end = self.value.offset_end


@dataclass
class Variable(ASTNode):
    """A use of a variable."""

    offset_end: int = field(init=False, repr=False)

    name: str
    """The name of the variable."""

    def __post_init__(self) -> None:
        self.offset_end = self.offset + len(self.name)


@dataclass
class Subscript(ASTNode):
    """A subscripted variable (e.g. ``x[1]``)."""

    offset: int = field(init=False, repr=False)

    variable: Union["Subscript", Variable]
    """
    The :py:class:`Variable` being subscripted (or :py:class:`Subscript` in the
    case of a multiply subscripted variable.
    """

    subscript: "Expr"
    """The :py:class:`Expr` giving the subscript value."""

    def __post_init__(self) -> None:
        self.offset = self.variable.offset

    @property
    def name(self) -> str:
        return self.variable.name


@dataclass
class Label(ASTNode):
    """A label value."""

    offset_end: int = field(init=False, repr=False)

    name: str
    """The label name."""

    def __post_init__(self) -> None:
        self.offset_end = self.offset + len(self.name)


@dataclass
class Expr(ASTNode):
    pass


@dataclass
class PerenExpr(Expr):
    """A parenthesised expression."""

    value: Expr
    """The parenthesised :py:class:`Expr`"""


@dataclass
class UnaryExpr(Expr):
    """A unary expression, e.g. ``-foo``."""

    offset_end: int = field(init=False, repr=False)

    op: UnaryOp
    """The operator (:py:class:`.UnaryOp`)"""

    value: Expr
    """The :py:class:`Expr` the operator applies to."""

    def __post_init__(self) -> None:
        self.offset_end = self.value.offset_end


@dataclass
class BinaryExpr(Expr):
    """A binary expression, e.g. ``a + 1``."""

    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)

    lhs: Expr
    """The :py:class:`Expr` on the left-hand side of the expression."""

    op: BinaryOp
    """The operator (:py:class:`.BinaryOp`)"""

    rhs: Expr
    """The :py:class:`Expr` on the right-hand side of the expression."""

    def __post_init__(self) -> None:
        self.offset = self.lhs.offset
        self.offset_end = self.rhs.offset_end


@dataclass
class FunctionCallExpr(Expr):
    """
    A call to a function::

        <name>(<arguments[0]>, <arguments[1]>, <...>)  # e.g. foo(1, 2, 3)
    """

    name: str
    """The name of the function to be called."""

    arguments: List[Expr]
    """The list of :py:class:`Expr` giving the arguments to the call."""


@dataclass
class VariableExpr(Expr):
    """A use of a variable or subscripted variable."""

    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)

    variable: Union[Variable, Subscript]
    """The :py:class:`Variable` or :py:class:`Subscript` used."""

    def __post_init__(self) -> None:
        self.offset = self.variable.offset
        self.offset_end = self.variable.offset_end


@dataclass
class LabelExpr(Expr):
    """A label literal."""

    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)

    label: Label
    """The :py:class:`Label` used."""

    def __post_init__(self) -> None:
        self.offset = self.label.offset
        self.offset_end = self.label.offset_end


@dataclass
class EmptyMapExpr(Expr):
    """An empty map literal (e.g. ``{}``)."""

    pass


@dataclass
class BooleanExpr(Expr):
    """A boolean literal, i.e. ``True`` and ``False``."""

    offset_end: int = field(init=False, repr=False)

    value: bool
    """The boolean value."""

    def __post_init__(self) -> None:
        self.offset_end = self.offset + (4 if self.value else 5)


@dataclass
class NumberExpr(Expr):
    """A numerical literal integer, e.g. ``123`` or ``0xF00``."""

    value: int
    """The (parsed) integer value."""

    display_base: int = 10
    """The base which the literal was encoded in."""

    display_digits: int = 1
    """
    The number of digits used in the literal, including leading zeros but
    excluding any prefix (e.g. ``0x`` or ``0b``).
    """


@dataclass
class ASTConstructionError(Exception):
    """
    Exceptions thrown during construction of an AST.
    """

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
class LabelUsedAsVariableNameError(ASTConstructionError):
    """
    Thrown when a name previously used as a label is assigned to like a
    variable.
    """

    variable_name: str

    @classmethod
    def from_variable(
        cls, source: str, variable: Variable
    ) -> "LabelUsedAsVariableNameError":
        line, column = offset_to_line_and_column(source, variable.offset)
        snippet = extract_line(source, line)
        return cls(line, column, snippet, variable.name)

    @property
    def explanation(self) -> str:
        return f"The name '{self.variable_name}' is already in use as a label name."


@dataclass
class CannotSubscriptLabelError(ASTConstructionError):
    """
    Thrown when name which has previously used as a label is subscripted like a
    variable.
    """

    label_name: str

    @classmethod
    def from_subscript(
        cls, source: str, subscript: Subscript
    ) -> "CannotSubscriptLabelError":
        line, column = offset_to_line_and_column(source, subscript.offset)
        snippet = extract_line(source, line)
        return cls(line, column, snippet, subscript.name)

    @property
    def explanation(self) -> str:
        return f"Attempting to subscript label '{self.label_name}'."


class ToAST(ParseTreeTransformer):
    """
    Transformer which transforms a :py:class:`ParseTree` resulting from parsing
    a piece of pseudocode into an Abstract Syntax Tree (AST) rooted with a
    :py:class:`Listing`.
    """

    def _transform_regex(self, regex: Regex) -> Regex:
        return regex

    def _transform_lookahead(self, lookahead: Lookahead) -> Lookahead:
        return lookahead

    def comment(self, _pt: ParseTree, comment: Regex) -> Comment:
        return Comment(comment.start, comment.string.rstrip("\r\n"))

    def v_space(self, _pt: ParseTree, newline: Regex) -> EmptyLine:
        return EmptyLine(newline.start, newline.end)

    def any_ws(self, _pt: ParseTree, children: Any) -> List[EmptyLine]:
        out: List[EmptyLine] = []
        empty_line_offset: Optional[int] = None
        for child in children:
            if isinstance(child, Comment):  # comment
                out.append(
                    EmptyLine(
                        empty_line_offset
                        if empty_line_offset is not None
                        else child.offset,
                        child.offset_end,
                        child,
                    )
                )
                empty_line_offset = None
            elif isinstance(child, EmptyLine):  # v_space
                out.append(
                    EmptyLine(
                        empty_line_offset
                        if empty_line_offset is not None
                        else child.offset,
                        child.offset_end,
                    )
                )
                empty_line_offset = None
            elif isinstance(child, Regex):  # h_space
                if empty_line_offset is None:
                    empty_line_offset = child.start

        return out

    def eol(self, _pt: ParseTree, children: Any) -> EOL:
        h_space, comment_or_v_space_or_eof, any_ws = children

        empty_lines = cast(List[EmptyLine], any_ws)

        offset: int
        if h_space is not None:
            offset = h_space.start
        elif isinstance(comment_or_v_space_or_eof, (Comment, EmptyLine)):
            offset = comment_or_v_space_or_eof.offset
        elif isinstance(comment_or_v_space_or_eof, Lookahead):  # i.e. EOF
            offset = comment_or_v_space_or_eof.offset
        else:
            raise NotImplementedError()  # Unreachable

        offset_end: int
        if empty_lines:
            offset_end = empty_lines[-1].offset_end
        elif isinstance(comment_or_v_space_or_eof, Lookahead):  # i.e. EOF
            offset_end = comment_or_v_space_or_eof.offset
        elif isinstance(comment_or_v_space_or_eof, (Comment, EmptyLine)):
            offset_end = comment_or_v_space_or_eof.offset_end
        elif h_space is not None:
            offset_end = h_space.end

        comment: Optional[Comment]
        if isinstance(comment_or_v_space_or_eof, Comment):
            comment = comment_or_v_space_or_eof
        else:
            comment = None

        return EOL(offset, offset_end, comment, empty_lines)

    def start(self, parse_tree: Alt, children: Any) -> Listing:
        any_ws, functions, _eof = children

        return Listing(cast(List[Function], functions), cast(List[EmptyLine], any_ws),)

    def stmt_block(
        self, parse_tree: Alt, children: Any
    ) -> Tuple[Optional[EOL], List[Union[Stmt]]]:
        if parse_tree.choice_index == 0:  # One-liner
            _colon, _ws, stmt = children
            return None, [cast(Stmt, stmt)]
        elif parse_tree.choice_index == 1:  # Multi-line form
            _colon, eol, body = children
            return (cast(EOL, eol), cast(List[Stmt], body))
        else:
            raise TypeError(parse_tree.choice_index)  # Unreachable

    def function(self, _pt: ParseTree, children: Any) -> Function:
        name, _ws, arguments, _ws, body = children
        eol, stmts = body
        return Function(
            name.start,
            cast(str, name.string),
            cast(List[Variable], arguments),
            cast(List[Stmt], stmts),
            cast(Optional[EOL], eol),
        )

    def function_arguments(self, _pt: ParseTree, children: Any) -> List[Variable]:
        _open, _ws1, maybe_args, _ws2, _close = children
        if maybe_args is None:
            return []

        first, _ws, rest, _comma = maybe_args

        arguments = cast(
            List[Regex], [first] + [var for _comma, _ws1, var, _ws2 in rest]
        )

        return [Variable(a.start, a.string) for a in arguments]

    def if_else_stmt(self, _pt: ParseTree, children: Any) -> IfElseStmt:
        if_block, else_if_blocks, else_block = children

        if_branches = []

        if_, _ws1, condition, _ws2, body = if_block
        eol, stmts = body
        if_branches.append(
            IfBranch(
                cast(Regex, if_).start,
                cast(Expr, condition),
                cast(List[Stmt], stmts),
                cast(Optional[EOL], eol),
            )
        )

        for else_, _ws1, _if, _ws2, condition, _ws3, body in else_if_blocks:
            eol, stmts = body
            if_branches.append(
                IfBranch(
                    cast(Regex, else_).start,
                    cast(Expr, condition),
                    cast(List[Stmt], stmts),
                    cast(Optional[EOL], eol),
                )
            )

        else_branch: Optional[ElseBranch] = None
        if else_block is not None:
            else_, _ws, body = else_block
            eol, stmts = body
            else_branch = ElseBranch(
                cast(Regex, else_).start,
                cast(List[Stmt], stmts),
                cast(Optional[EOL], eol),
            )

        return IfElseStmt(if_branches, else_branch)

    def for_each_stmt(self, _pt: ParseTree, children: Any) -> ForEachStmt:
        (
            for_,
            _ws1,
            _each_,
            _ws2,
            identifier,
            _ws3,
            _in,
            _ws4,
            values,
            _ws5,
            body,
        ) = children
        eol, stmts = body
        return ForEachStmt(
            for_.start,
            Variable(identifier.start, identifier.string),
            cast(List[Expr], values),
            cast(List[Stmt], stmts),
            cast(Optional[EOL], eol),
        )

    def for_each_list(self, _pt: ParseTree, children: Any) -> List[Expr]:
        first, rest = children
        values = [first] + [expr for _ws1, _comma, _ws2, expr in rest]
        return cast(List[Expr], values)

    def for_stmt(self, _pt: ParseTree, children: Any) -> ForStmt:
        (
            for_,
            _ws1,
            identifier,
            _ws2,
            _eq,
            _ws3,
            start,
            _ws4,
            _to,
            _ws5,
            end,
            _ws6,
            body,
        ) = children
        eol, stmts = body
        return ForStmt(
            for_.start,
            Variable(identifier.start, identifier.string),
            cast(Expr, start),
            cast(Expr, end),
            cast(List[Stmt], stmts),
            cast(Optional[EOL], eol),
        )

    def while_stmt(self, _pt: ParseTree, children: Any) -> WhileStmt:
        while_, _ws1, condition, _ws2, body = children
        eol, stmts = body
        return WhileStmt(
            while_.start,
            cast(Expr, condition),
            cast(List[Stmt], stmts),
            cast(Optional[EOL], eol),
        )

    def function_call_stmt(self, _pt: ParseTree, children: Any) -> FunctionCallStmt:
        function_call, eol = children
        return FunctionCallStmt(cast(FunctionCallExpr, function_call), cast(EOL, eol),)

    def return_stmt(self, _pt: ParseTree, children: Any) -> ReturnStmt:
        return_, _ws, expr, eol = children

        return ReturnStmt(return_.start, cast(Expr, expr), cast(EOL, eol))

    def assignment_stmt(self, _pt: ParseTree, children: Any) -> AssignmentStmt:
        variable, _ws1, op, _ws2, expr, eol = children
        return AssignmentStmt(
            cast(Union[Variable, Subscript], variable),
            AssignmentOp(op.string),
            cast(Expr, expr),
            cast(EOL, eol),
        )

    def condition(self, _pt: ParseTree, children: Any) -> Expr:
        _open, _ws1, expr, _ws2, _close = children
        return cast(Expr, expr)

    def maybe_unary_expr(self, parse_tree: Alt, children: Any) -> Expr:
        if parse_tree.choice_index == 0:
            op, _ws, expr = children
            return UnaryExpr(op.start, UnaryOp(op.string), cast(Expr, expr))
        elif parse_tree.choice_index == 1:
            return cast(Expr, children)
        else:
            raise TypeError(parse_tree.choice_index)  # Unreachable

    maybe_log_not_expr = maybe_unary_expr

    def binary_expr(self, _pt: ParseTree, children: Any) -> Expr:
        lhs, rhss = cast(Tuple[Expr, Any], children)

        if len(rhss) == 0:
            return lhs

        values = [lhs] + [cast(Expr, rhs) for _ws1, _op, _ws2, rhs in rhss]
        ops = [BinaryOp(op.string) for _ws1, op, _ws2, _rhs in rhss]

        # NB: This function will only be called with a string of operators of
        # the same precedence. The ``test_operator_associativity_table_sanity``
        # test in ``tests/test_parser.py`` verifies that in this case, all
        # operators have the same associativity.
        associativity = OPERATOR_ASSOCIATIVITY_TABLE[ops[0]]

        if associativity == Associativity.left:
            lhs = values[0]
            for op, rhs in zip(ops, values[1:]):
                lhs = BinaryExpr(lhs, op, rhs)
            return lhs
        elif associativity == Associativity.right:
            rhs = values[-1]
            for op, lhs in zip(reversed(ops), reversed(values[:-1])):
                rhs = BinaryExpr(lhs, op, rhs)
            return rhs
        else:
            raise TypeError(associativity)  # Unreachable

    maybe_log_or_expr = binary_expr
    maybe_log_and_expr = binary_expr
    maybe_cmp_expr = binary_expr
    maybe_or_expr = binary_expr
    maybe_xor_expr = binary_expr
    maybe_and_expr = binary_expr
    maybe_shift_expr = binary_expr
    maybe_arith_expr = binary_expr
    maybe_prod_expr = binary_expr
    maybe_pow_expr = binary_expr

    def maybe_peren_expr(self, parse_tree: Alt, children: Any) -> Expr:
        if parse_tree.choice_index == 0:  # Perentheses
            open_, _ws1, expr, _ws2, close_ = children
            return PerenExpr(open_.start, close_.end, cast(Expr, expr))
        elif parse_tree.choice_index == 1:  # Pass-through
            return cast(Expr, children)
        else:
            raise TypeError(parse_tree.choice_index)  # Unreachable

    def atom(self, parse_tree: Alt, children: Any) -> Expr:
        if parse_tree.choice_index == 1:  # Variable
            return VariableExpr(cast(Union[Variable, Subscript], children))
        elif parse_tree.choice_index in (0, 2, 3, 4):  # call, map, bool, num
            return cast(Expr, children)  # Already Expr types
        else:
            raise TypeError(parse_tree.choice_index)  # Unreachable

    def function_call(self, _pt: ParseTree, children: Any) -> FunctionCallExpr:
        identifier, _ws, (arguments, offset_end) = children
        return FunctionCallExpr(
            identifier.start,
            cast(int, offset_end),
            cast(str, identifier.string),
            cast(List[Expr], arguments),
        )

    def function_call_arguments(
        self, _pt: ParseTree, children: Any
    ) -> Tuple[List[Expr], int]:
        _open, _ws1, maybe_args, _ws2, close_ = children
        offset_end = close_.end
        if maybe_args is None:
            return ([], offset_end)
        else:
            first, _ws, rest, _comma = maybe_args
            arguments = [first] + [expr for _comma, _ws, expr, _ws in rest]
            return (cast(List[Expr], arguments), offset_end)

    def variable(self, _pt: ParseTree, children: Any) -> Union[Variable, Subscript]:
        identifier, ws_and_subscripts = children

        variable: Union[Variable, Subscript] = Variable(
            identifier.start, identifier.string,
        )
        offset_end = identifier.end
        for _ws, (expr, offset_end) in ws_and_subscripts:
            variable = Subscript(offset_end, variable, cast(Expr, expr))

        return variable

    def subscript(self, _pt: ParseTree, children: Any) -> Tuple[Expr, int]:
        open_, _ws1, expr, _ws2, close_ = children
        offset_end = close_.end
        return (cast(Expr, expr), offset_end)

    def empty_map(self, _pt: ParseTree, children: Any) -> EmptyMapExpr:
        open_, _ws, close_ = children
        return EmptyMapExpr(open_.start, close_.end)

    def boolean(self, _pt: ParseTree, value: Regex) -> BooleanExpr:
        return BooleanExpr(value.start, value.string == "True")

    def number(self, _pt: ParseTree, number: Regex) -> NumberExpr:
        offset = number.start
        offset_end = number.end
        string = number.string
        if string.startswith("0b") or string.startswith("0B"):
            return NumberExpr(offset, offset_end, int(string, 2), 2, len(string) - 2)
        elif string.startswith("0x") or string.startswith("0X"):
            return NumberExpr(offset, offset_end, int(string, 16), 16, len(string) - 2)
        else:
            return NumberExpr(offset, offset_end, int(string), 10, len(string))

    def identifier(self, _pt: ParseTree, children: Any) -> str:
        _la, identifier = children
        return cast(str, identifier)


def infer_labels(source: str, node: Listing) -> None:
    """
    Replace :py:class:`Variables <Variable>` whose names have no definition
    with :py:class:`Labels <Label>`. Operates in-place.
    """
    _variables: Set[str] = set()
    _labels: Set[str] = set()

    def declare_variable(variable: Variable) -> None:
        if variable.name in _labels:
            raise LabelUsedAsVariableNameError.from_variable(source, variable)
        else:
            _variables.add(variable.name)

    def transform_list(nodes: Sequence[ASTNode]) -> List[ASTNode]:
        return [transform(node) for node in nodes]

    def transform(node: ASTNode) -> ASTNode:
        if isinstance(node, Listing):
            node.functions = cast(List[Function], transform_list(node.functions))
            return node
        elif isinstance(node, Function):
            # NB: New scope is started for each function
            _variables.clear()
            _labels.clear()
            for variable in node.arguments:
                declare_variable(variable)
            node.body = cast(List[Stmt], transform_list(node.body))
            return node
        elif isinstance(node, IfElseStmt):
            node.if_branches = cast(List[IfBranch], transform_list(node.if_branches))
            if node.else_branch is not None:
                node.else_branch = cast(ElseBranch, transform(node.else_branch))
            return node
        elif isinstance(node, IfBranch):
            node.condition = cast(Expr, transform(node.condition))
            node.body = cast(List[Stmt], transform_list(node.body))
            return node
        elif isinstance(node, ElseBranch):
            node.body = cast(List[Stmt], transform_list(node.body))
            return node
        elif isinstance(node, ForEachStmt):
            # NB: Values processed before variable comes into scope
            node.values = cast(List[Expr], transform_list(node.values))
            declare_variable(node.variable)
            node.body = cast(List[Stmt], transform_list(node.body))
            return node
        elif isinstance(node, ForStmt):
            # NB: Endpoints processed before variable comes into scope
            node.start = cast(Expr, transform(node.start))
            node.end = cast(Expr, transform(node.end))
            declare_variable(node.variable)
            node.body = cast(List[Stmt], transform_list(node.body))
            return node
        elif isinstance(node, WhileStmt):
            node.condition = cast(Expr, transform(node.condition))
            node.body = cast(List[Stmt], transform_list(node.body))
            return node
        elif isinstance(node, FunctionCallStmt):
            node.call = cast(FunctionCallExpr, transform(node.call))
            return node
        elif isinstance(node, ReturnStmt):
            node.value = cast(Expr, transform(node.value))
            return node
        elif isinstance(node, AssignmentStmt):
            # NB: Value and subscripts processed before variable comes into scope
            node.value = cast(Expr, transform(node.value))

            if isinstance(node.variable, Subscript):
                node.variable = cast(Subscript, transform(node.variable))
            elif isinstance(node.variable, Variable):
                declare_variable(node.variable)
                node.variable = cast(
                    Variable, transform(node.variable)
                )  # Not really necessary...

            return node
        elif isinstance(node, Variable):
            if node.name not in _variables:
                _labels.add(node.name)
                return Label(node.offset, node.name)
            else:
                return node
        elif isinstance(node, Subscript):
            variable_or_label = transform(node.variable)
            if isinstance(variable_or_label, (Variable, Subscript)):
                node.variable = variable_or_label
            else:
                raise CannotSubscriptLabelError.from_subscript(source, node)

            node.subscript = cast(Expr, transform(node.subscript))

            return node
        elif isinstance(node, Label):
            _labels.add(node.name)
            # NB: The following situation will only occur if a hand-made AST
            # contains Labels which shaddow variables. Consequently no 'pretty'
            # exception has been defined.
            assert node.name not in _variables
            return node
        elif isinstance(node, PerenExpr):
            node.value = cast(Expr, transform(node.value))
            return node
        elif isinstance(node, UnaryExpr):
            node.value = cast(Expr, transform(node.value))
            return node
        elif isinstance(node, BinaryExpr):
            node.lhs = cast(Expr, transform(node.lhs))
            node.rhs = cast(Expr, transform(node.rhs))
            return node
        elif isinstance(node, FunctionCallExpr):
            node.arguments = cast(List[Expr], transform_list(node.arguments))
            return node
        elif isinstance(node, VariableExpr):
            variable_or_label = cast(
                Union[Variable, Subscript, Label], transform(node.variable)
            )
            if isinstance(variable_or_label, Label):
                return LabelExpr(variable_or_label)
            else:
                return VariableExpr(variable_or_label)
        elif isinstance(node, LabelExpr):
            node.label = cast(Label, transform(node.label))
            return node
        elif isinstance(node, EmptyMapExpr):
            return node
        elif isinstance(node, BooleanExpr):
            return node
        elif isinstance(node, NumberExpr):
            return node
        else:
            raise TypeError(type(node))  # Unreachable

    transform(node)
