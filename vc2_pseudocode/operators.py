"""
Pseudocode language operator information.
"""

from typing import Mapping, Union, Iterable, cast

from enum import Enum

from itertools import chain


class UnaryOp(Enum):
    plus = "+"
    minus = "-"
    bitwise_not = "~"
    logical_not = "not"


class BinaryOp(Enum):
    logical_or = "or"
    logical_and = "and"
    eq = "=="
    ne = "!="
    lt = "<"
    le = "<="
    gt = ">"
    ge = ">="
    bitwise_or_ = "|"
    bitwise_xor = "^"
    bitwise_and_ = "&"
    lsh = "<<"
    rsh = ">>"
    add = "+"
    sub = "-"
    mul = "*"
    idiv = "//"
    mod = "%"
    pow = "**"


OPERATOR_PRECEDENCE_TABLE: Mapping[Union[BinaryOp, UnaryOp], int] = {
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
"""Operator precedence scores. Higher scores mean higher precedence."""


class Associativity(Enum):
    """Operator associativity."""

    left = "left"
    right = "right"


OPERATOR_ASSOCIATIVITY_TABLE: Mapping[Union[BinaryOp, UnaryOp], Associativity] = dict(
    chain(
        [(op, Associativity.left) for op in BinaryOp if op != BinaryOp("**")],
        [(BinaryOp("**"), Associativity.right)],
        [(op, Associativity.right) for op in UnaryOp],
    )
)
"""Operator associativities."""


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
