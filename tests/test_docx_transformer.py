import pytest  # type: ignore

from typing import Any

from dataclasses import fields

import pseudocode_samples

from vc2_pseudocode.parser import parse

from vc2_pseudocode.ast import ASTNode

from vc2_pseudocode.docx_generator import ListingTable, RunStyle

from vc2_pseudocode.docx_transformer import DocxTransformer, pseudocode_to_docx


def assert_equal_ignoring_offset_and_whitespace(a: ASTNode, b: ASTNode) -> None:
    assert type(a) == type(b), (a, b)

    for field in fields(a):
        if field.name not in ("offset", "offset_end", "eol", "leading_empty_lines"):
            va = getattr(a, field.name)
            vb = getattr(b, field.name)

            assert type(va) == type(vb), (a, b)

            vas = va if isinstance(va, list) else [va]
            vbs = vb if isinstance(vb, list) else [vb]

            assert len(vas) == len(vbs), (a, b)

            for vae, vbe in zip(vas, vbs):
                assert type(vae) == type(vbe), (a, b)
                if isinstance(vae, ASTNode):
                    assert_equal_ignoring_offset_and_whitespace(vae, vbe)
                else:
                    assert vae == vbe, (a, b)


@pytest.mark.parametrize(
    "a_str, b_str, exp_equal",
    [
        # Completely identical
        ("foo(): bar()", "foo(): bar()", True),
        # Different
        ("foo(): bar()", "foo(): baz()", False),
        ("foo(): bar()", "foo(): bar()\nbar(): baz()", False),
        # Differ only in offsets
        ("foo(): bar()", "foo  ( )  :  bar ( ) ", True),
        # Differ only in whitespace/comments
        ("foo(): bar()", "foo(): # OK \n \n bar() # Yep", True),
        # Differ only in leading whitespace
        ("# Hello\n\nfoo(): bar()", "foo(): bar()", True),
    ],
)
def test_assert_equal_ignoring_offset_and_whitespace(
    a_str: str, b_str: str, exp_equal: bool
) -> None:
    a = parse(a_str)
    b = parse(b_str)

    if exp_equal:
        assert_equal_ignoring_offset_and_whitespace(a, b)
    else:
        with pytest.raises(AssertionError):
            assert_equal_ignoring_offset_and_whitespace(a, b)


@pytest.mark.parametrize("sample_name", pseudocode_samples.__all__)
def test_document_code_has_equivalent_ast(sample_name: str) -> None:
    # Check that the generated pseudocode in the docx produces the same parse
    # tree (modulo comments and offsets) as the input pseudocode
    pseudocode = getattr(pseudocode_samples, sample_name)
    ast = parse(pseudocode)

    transformer = DocxTransformer()
    document = transformer.transform(ast)

    extracted_pseudocode = ""
    for paragraph_or_table in document.body:
        if isinstance(paragraph_or_table, ListingTable):
            for row in paragraph_or_table.rows:
                extracted_pseudocode += f"{str(row.code)}\n"

    extracted_ast = parse(extracted_pseudocode)

    assert_equal_ignoring_offset_and_whitespace(ast, extracted_ast)


@pytest.mark.parametrize("sample_name", pseudocode_samples.__all__)
def test_document_code_is_all_styled(sample_name: str) -> None:
    # Check that the generated pseudocode is completely styled with pseudocode
    # text styles
    pseudocode = getattr(pseudocode_samples, sample_name)
    ast = parse(pseudocode)

    transformer = DocxTransformer()
    document = transformer.transform(ast)

    for paragraph_or_table in document.body:
        if isinstance(paragraph_or_table, ListingTable):
            for row in paragraph_or_table.rows:
                for run in row.code.runs:
                    assert run.style in (
                        RunStyle.pseudocode,
                        RunStyle.pseudocode_fdef,
                        RunStyle.pseudocode_keyword,
                        RunStyle.pseudocode_label,
                    )


def test_pseudocode_to_docx(tmpdir: Any) -> None:
    # Just a sanity check...
    pseudocode_to_docx("foo(): bar()", str(tmpdir.join("out.docx")))
