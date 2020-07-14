from vc2_pseudocode.pseudocode_parser.operators import (
    BinaryOp,
    UnaryOp,
    OPERATOR_PRECEDENCE_TABLE,
    OPERATOR_ASSOCIATIVITY_TABLE,
    Associativity,
)


def test_operator_precedence_table_completeness() -> None:
    assert set(OPERATOR_PRECEDENCE_TABLE) == set(BinaryOp) | set(UnaryOp)


def test_operator_associativity_table_completeness() -> None:
    assert set(OPERATOR_ASSOCIATIVITY_TABLE) == set(BinaryOp) | set(UnaryOp)


def test_operator_associativity_table_sanity() -> None:
    # For sanity, all unary ops must be right-associative
    for uo in UnaryOp:
        assert OPERATOR_ASSOCIATIVITY_TABLE[uo] == Associativity.right

    # All operators with the same associativity must have same precedence (the
    # binary_expr method in the AST generator relies on this property).
    for op, score in OPERATOR_PRECEDENCE_TABLE.items():
        for other_op, other_score in OPERATOR_PRECEDENCE_TABLE.items():
            if score == other_score:
                assert (
                    OPERATOR_ASSOCIATIVITY_TABLE[op]
                    == OPERATOR_ASSOCIATIVITY_TABLE[other_op]
                )
