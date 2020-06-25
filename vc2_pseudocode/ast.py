"""
Abstract Syntax Tree (AST) data structures for the VC-2 specification pseudocode language.
"""

from typing import List, Union, Optional, Any, cast

from peggie.transformer import ParseTreeTransformer

from peggie.parser import ParseTree, Alt, Regex

from enum import Enum

from dataclasses import dataclass, field


@dataclass
class ASTNode:
    offset: int
    """Character offset into the source. Optional."""


@dataclass
class Listing(ASTNode):
    offset: int = field(default=0, init=False, repr=False)
    functions: List["Function"]
    comments: List["Comment"]


@dataclass
class Function(ASTNode):
    name: str
    arguments: List["Variable"]
    body: List["Stmt"]


@dataclass
class Stmt(ASTNode):
    pass


@dataclass
class IfElseStmt(Stmt):
    offset: int = field(init=False, repr=False)
    if_branches: List["IfBranch"]
    else_branch: Optional["ElseBranch"] = None

    def __post_init__(self) -> None:
        self.offset = self.if_branches[0].offset


@dataclass
class IfBranch(ASTNode):
    condition: "Expr"
    body: List[Stmt]


@dataclass
class ElseBranch(ASTNode):
    body: List[Stmt]


@dataclass
class ForEachStmt(Stmt):
    variable: "Variable"
    values: List["Expr"]
    body: List[Stmt]


@dataclass
class ForStmt(Stmt):
    variable: "Variable"
    start: "Expr"
    end: "Expr"
    body: List[Stmt]


@dataclass
class WhileStmt(Stmt):
    condition: "Expr"
    body: List[Stmt]


@dataclass
class FunctionCallStmt(Stmt):
    offset: int = field(init=False, repr=False)
    call: "FunctionCallExpr"

    def __post_init__(self) -> None:
        self.offset = self.call.offset


@dataclass
class ReturnStmt(Stmt):
    value: "Expr"


class AssignmentOp(Enum):
    assign = "="
    add_assign = "+="
    sub_assign = "-="
    mul_assign = "*="
    idiv_assign = "//="
    and_assign = "&="
    xor_assign = "^="
    or_assign = "|="
    lsh_assign = "<<="
    rsh_assign = ">>="


@dataclass
class AssignmentStmt(Stmt):
    offset: int = field(init=False, repr=False)
    variable: Union["Variable", "Subscript"]
    op: AssignmentOp
    value: "Expr"

    def __post_init__(self) -> None:
        self.offset = self.variable.offset


@dataclass
class Variable(ASTNode):
    name: str


@dataclass
class Subscript(ASTNode):
    offset: int = field(init=False, repr=False)
    variable: Union["Subscript", Variable]
    subscript: "Expr"

    def __post_init__(self) -> None:
        self.offset = self.variable.offset


@dataclass
class Expr(ASTNode):
    pass


class UnaryOp(Enum):
    plus = "+"
    minus = "-"
    bitwise_not = "!"


@dataclass
class UnaryExpr(Expr):
    op: UnaryOp
    value: Expr


class BinaryOp(Enum):
    eq = "=="
    ne = "!="
    lt = "<"
    le = "<="
    gt = ">"
    ge = ">="
    or_ = "|"
    xor = "^"
    and_ = "&"
    lsh = "<<"
    rsh = ">>"
    add = "+"
    sub = "-"
    mul = "*"
    idiv = "//"


@dataclass
class BinaryExpr(Expr):
    offset: int = field(init=False, repr=False)
    lhs: Expr
    op: BinaryOp
    rhs: Expr

    def __post_init__(self) -> None:
        self.offset = self.lhs.offset


@dataclass
class FunctionCallExpr(Expr):
    name: str
    arguments: List[Expr]


@dataclass
class VariableExpr(Expr):
    offset: int = field(init=False, repr=False)
    variable: Union[Variable, Subscript]

    def __post_init__(self) -> None:
        self.offset = self.variable.offset


@dataclass
class EmptyMapExpr(Expr):
    pass


@dataclass
class BooleanExpr(Expr):
    value: bool


@dataclass
class NumberExpr(Expr):
    value: int
    display_base: int = 10


@dataclass
class Comment(ASTNode):
    string: str


class ToAST(ParseTreeTransformer):
    """
    Transformer which transforms a :py:class:`ParseTree` resulting from parsing
    a piece of pseudocode into an Abstract Syntax Tree (AST) rooted with a
    :py:class:`Listing`.
    """

    _comments: List[Comment]

    def _transform_regex(self, regex: Regex) -> Regex:
        return regex

    def start_enter(self, _pt: ParseTree) -> None:
        self._comments = []

    def comment(self, _pt: ParseTree, comment: Regex) -> Regex:
        self._comments.append(Comment(comment.start, comment.string.rstrip("\r\n"),))
        return comment

    def start(self, parse_tree: Alt, children: Any) -> Listing:
        _ws1, functions_and_ws, _eof = children

        return Listing(
            [cast(Function, function) for function, _ws in functions_and_ws],
            self._comments,
        )

    def function(self, _pt: ParseTree, children: Any) -> Function:
        name, _ws, arguments, _ws, body = children
        return Function(
            name.start,
            cast(str, name.string),
            cast(List[Variable], arguments),
            cast(List[Stmt], body),
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

    def stmt_block(self, parse_tree: Alt, children: Any) -> List[Stmt]:
        if parse_tree.choice_index == 0:  # One-liner
            _colon, _ws, stmt = children
            return [cast(Stmt, stmt)]
        elif parse_tree.choice_index == 1:  # Multi-line form
            _colon, _eol, stmts = children
            return cast(List[Stmt], stmts)
        else:
            raise TypeError(parse_tree.choice_index)  # Unreachable

    def if_else_stmt(self, _pt: ParseTree, children: Any) -> IfElseStmt:
        if_block, else_if_blocks, else_block = children

        if_branches = []

        if_, _ws1, condition, _ws2, body = if_block
        if_branches.append(
            IfBranch(
                cast(Regex, if_).start, cast(Expr, condition), cast(List[Stmt], body),
            )
        )

        for else_, _ws1, _if, _ws2, condition, _ws3, body in else_if_blocks:
            if_branches.append(
                IfBranch(
                    cast(Regex, else_).start,
                    cast(Expr, condition),
                    cast(List[Stmt], body),
                )
            )

        else_branch: Optional[ElseBranch] = None
        if else_block is not None:
            else_, _ws, body = else_block
            else_branch = ElseBranch(cast(Regex, else_).start, cast(List[Stmt], body),)

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
        return ForEachStmt(
            for_.start,
            Variable(identifier.start, identifier.string),
            cast(List[Expr], values),
            cast(List[Stmt], body),
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
        return ForStmt(
            for_.start,
            Variable(identifier.start, identifier.string),
            cast(Expr, start),
            cast(Expr, end),
            cast(List[Stmt], body),
        )

    def while_stmt(self, _pt: ParseTree, children: Any) -> WhileStmt:
        while_, _ws1, condition, _ws2, stmts = children
        return WhileStmt(while_.start, cast(Expr, condition), cast(List[Stmt], stmts))

    def function_call_stmt(self, _pt: ParseTree, children: Any) -> FunctionCallStmt:
        function_call, _eol = children
        return FunctionCallStmt(cast(FunctionCallExpr, function_call))

    def return_stmt(self, _pt: ParseTree, children: Any) -> ReturnStmt:
        return_, _ws, expr, _eol = children
        return ReturnStmt(return_.start, cast(Expr, expr))

    def assignment_stmt(self, _pt: ParseTree, children: Any) -> AssignmentStmt:
        variable, _ws1, op, _ws2, expr, _eol = children
        return AssignmentStmt(
            cast(Union[Variable, Subscript], variable),
            AssignmentOp(op.string),
            cast(Expr, expr),
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

    def binary_expr(self, _pt: ParseTree, children: Any) -> Expr:
        lhs, rhss = children
        for _ws1, op, _ws2, rhs in rhss:
            lhs = BinaryExpr(cast(Expr, lhs), BinaryOp(op.string), cast(Expr, rhs))
        return cast(Expr, lhs)

    maybe_eq_ne_expr = binary_expr
    maybe_lt_gt_expr = binary_expr
    maybe_or_expr = binary_expr
    maybe_xor_expr = binary_expr
    maybe_and_expr = binary_expr
    maybe_shift_expr = binary_expr
    maybe_arith_expr = binary_expr
    maybe_prod_expr = binary_expr

    def maybe_peren_expr(self, parse_tree: Alt, children: Any) -> Expr:
        if parse_tree.choice_index == 0:  # Perentheses
            _open, _ws1, expr, _ws2, _close = children
            return cast(Expr, expr)
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
        identifier, _ws, arguments = children
        return FunctionCallExpr(
            identifier.start, cast(str, identifier.string), cast(List[Expr], arguments),
        )

    def function_call_arguments(self, _pt: ParseTree, children: Any) -> List[Expr]:
        _open, _ws1, maybe_args, _ws2, _close = children
        if maybe_args is None:
            return []
        else:
            first, _ws, rest, _comma = maybe_args
            arguments = [first] + [expr for _comma, _ws, expr, _ws in rest]
            return cast(List[Expr], arguments)

    def variable(self, _pt: ParseTree, children: Any) -> Union[Variable, Subscript]:
        identifier, _ws1, subscripts_and_ws = children

        variable: Union[Variable, Subscript] = Variable(
            identifier.start, identifier.string,
        )
        for expr, _ws in subscripts_and_ws:
            variable = Subscript(variable, cast(Expr, expr))

        return variable

    def subscript(self, _pt: ParseTree, children: Any) -> Expr:
        open_, _ws1, expr, _ws2, _close = children
        return cast(Expr, expr)

    def empty_map(self, _pt: ParseTree, children: Any) -> EmptyMapExpr:
        open_, _ws, _close = children
        return EmptyMapExpr(open_.start)

    def boolean(self, _pt: ParseTree, value: Regex) -> BooleanExpr:
        return BooleanExpr(value.start, value.string == "True")

    def number(self, _pt: ParseTree, number: Regex) -> NumberExpr:
        offset = number.start
        string = number.string
        if string.startswith("0b") or string.startswith("0B"):
            return NumberExpr(offset, int(string, 2), 2)
        elif string.startswith("0x") or string.startswith("0X"):
            return NumberExpr(offset, int(string, 16), 16)
        else:
            return NumberExpr(offset, int(string), 10)

    def identifier(self, _pt: ParseTree, children: Any) -> str:
        _la, identifier = children
        return cast(str, identifier)
