"""
Abstract Syntax Tree (AST) data structures for the VC-2 specification pseudocode language.
"""

from typing import List, Union, Optional, Any, cast, Tuple

from peggie.transformer import ParseTreeTransformer

from peggie.parser import ParseTree, Alt, Regex, Lookahead

from vc2_pseudocode.operators import (
    BinaryOp,
    UnaryOp,
    AssignmentOp,
    OPERATOR_ASSOCIATIVITY_TABLE,
    Associativity,
)

from dataclasses import dataclass, field


@dataclass
class ASTNode:
    offset: int
    offset_end: int
    """Character offset (start=inclusive, end=exclusive) into the source."""


@dataclass
class Listing(ASTNode):
    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)
    functions: List["Function"]
    leading_empty_lines: List["EmptyLine"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.offset = self.functions[0].offset
        self.offset_end = self.functions[-1].offset_end


@dataclass
class Comment(ASTNode):
    offset_end: int = field(init=False, repr=False)
    string: str

    def __post_init__(self) -> None:
        self.offset_end = self.offset + len(self.string)


@dataclass
class EmptyLine(ASTNode):
    comment: Optional[Comment] = None


@dataclass
class EOL(ASTNode):
    comment: Optional[Comment] = None
    empty_lines: List[EmptyLine] = field(default_factory=list)


@dataclass
class Function(ASTNode):
    offset_end: int = field(init=False, repr=False)
    name: str
    arguments: List["Variable"]
    body: List["Stmt"]
    eol: Optional[EOL] = None

    def __post_init__(self) -> None:
        self.offset_end = self.body[-1].offset_end


@dataclass
class Stmt(ASTNode):
    pass


@dataclass
class IfElseStmt(Stmt):
    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)
    if_branches: List["IfBranch"]
    else_branch: Optional["ElseBranch"] = None

    def __post_init__(self) -> None:
        self.offset = self.if_branches[0].offset
        if self.else_branch is not None:
            self.offset_end = self.else_branch.offset_end
        else:
            self.offset_end = self.if_branches[-1].offset_end


@dataclass
class IfBranch(ASTNode):
    offset_end: int = field(init=False, repr=False)
    condition: "Expr"
    body: List[Stmt]
    eol: Optional[EOL] = None

    def __post_init__(self) -> None:
        self.offset_end = self.body[-1].offset_end


@dataclass
class ElseBranch(ASTNode):
    offset_end: int = field(init=False, repr=False)
    body: List[Stmt]
    eol: Optional[EOL] = None

    def __post_init__(self) -> None:
        self.offset_end = self.body[-1].offset_end


@dataclass
class ForEachStmt(Stmt):
    offset_end: int = field(init=False, repr=False)
    variable: "Variable"
    values: List["Expr"]
    body: List[Stmt]
    eol: Optional[EOL] = None

    def __post_init__(self) -> None:
        self.offset_end = self.body[-1].offset_end


@dataclass
class ForStmt(Stmt):
    offset_end: int = field(init=False, repr=False)
    variable: "Variable"
    start: "Expr"
    end: "Expr"
    body: List[Stmt]
    eol: Optional[EOL] = None

    def __post_init__(self) -> None:
        self.offset_end = self.body[-1].offset_end


@dataclass
class WhileStmt(Stmt):
    offset_end: int = field(init=False, repr=False)
    condition: "Expr"
    body: List[Stmt]
    eol: Optional[EOL] = None

    def __post_init__(self) -> None:
        self.offset_end = self.body[-1].offset_end


@dataclass
class FunctionCallStmt(Stmt):
    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)
    call: "FunctionCallExpr"
    eol: EOL

    def __post_init__(self) -> None:
        self.offset = self.call.offset
        self.offset_end = self.call.offset_end


@dataclass
class ReturnStmt(Stmt):
    offset_end: int = field(init=False, repr=False)
    value: "Expr"
    eol: EOL

    def __post_init__(self) -> None:
        self.offset_end = self.value.offset_end


@dataclass
class AssignmentStmt(Stmt):
    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)
    variable: Union["Variable", "Subscript"]
    op: AssignmentOp
    value: "Expr"
    eol: EOL

    def __post_init__(self) -> None:
        self.offset = self.variable.offset
        self.offset_end = self.value.offset_end


@dataclass
class Variable(ASTNode):
    offset_end: int = field(init=False, repr=False)
    name: str

    def __post_init__(self) -> None:
        self.offset_end = self.offset + len(self.name)


@dataclass
class Subscript(ASTNode):
    offset: int = field(init=False, repr=False)
    variable: Union["Subscript", Variable]
    subscript: "Expr"

    def __post_init__(self) -> None:
        self.offset = self.variable.offset

    @property
    def name(self) -> str:
        return self.variable.name


@dataclass
class Expr(ASTNode):
    pass


@dataclass
class PerenExpr(Expr):
    value: Expr


@dataclass
class UnaryExpr(Expr):
    offset_end: int = field(init=False, repr=False)
    op: UnaryOp
    value: Expr

    def __post_init__(self) -> None:
        self.offset_end = self.value.offset_end


@dataclass
class BinaryExpr(Expr):
    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)
    lhs: Expr
    op: BinaryOp
    rhs: Expr

    def __post_init__(self) -> None:
        self.offset = self.lhs.offset
        self.offset_end = self.rhs.offset_end


@dataclass
class FunctionCallExpr(Expr):
    name: str
    arguments: List[Expr]


@dataclass
class VariableExpr(Expr):
    offset: int = field(init=False, repr=False)
    offset_end: int = field(init=False, repr=False)
    variable: Union[Variable, Subscript]

    def __post_init__(self) -> None:
        self.offset = self.variable.offset
        self.offset_end = self.variable.offset_end


@dataclass
class EmptyMapExpr(Expr):
    pass


@dataclass
class BooleanExpr(Expr):
    offset_end: int = field(init=False, repr=False)
    value: bool

    def __post_init__(self) -> None:
        self.offset_end = self.offset + (4 if self.value else 5)


@dataclass
class NumberExpr(Expr):
    value: int
    display_base: int = 10
    display_digits: int = 1


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
