from typing import Any

from vc2_pseudocode.docx_generator import (
    ListingDocument,
    Paragraph,
    Run,
    RunStyle,
    ListingTable,
    ListingLine,
)


class TestParagraph:
    def test_constructor(self) -> None:
        p = Paragraph()
        assert p.runs == []

        p = Paragraph([])
        assert p.runs == []

        p = Paragraph([Run("hi"), Run(" there")])
        assert p.runs == [Run("hi"), Run(" there")]

        p = Paragraph(Run("Hello"))
        assert p.runs == [Run("Hello")]

        p = Paragraph("World")
        assert p.runs == [Run("World")]

    def test_bool(self) -> None:
        assert bool(Paragraph()) is False
        assert bool(Paragraph("")) is False
        assert bool(Paragraph("hi")) is True

    def test_str(self) -> None:
        assert str(Paragraph()) == ""
        assert str(Paragraph("")) == ""
        assert str(Paragraph("Foo")) == "Foo"
        assert str(Paragraph([Run("Hello"), Run(" world")])) == "Hello world"

    def test_add(self) -> None:
        # __add__
        assert Paragraph() + Paragraph() == Paragraph()
        assert Paragraph("foo") + Paragraph("bar") == Paragraph(
            [Run("foo"), Run("bar")]
        )
        assert Paragraph("foo") + Run("bar") == Paragraph([Run("foo"), Run("bar")])
        assert Paragraph("foo") + "bar" == Paragraph([Run("foo"), Run("bar")])

        # __radd__
        assert Run("foo") + Paragraph("bar") == Paragraph([Run("foo"), Run("bar")])
        assert "foo" + Paragraph("bar") == Paragraph([Run("foo"), Run("bar")])


class TestRun:
    def test_bool(self) -> None:
        assert bool(Run("")) is False
        assert bool(Run("hi")) is True

    def test_str(self) -> None:
        assert str(Run("Foo")) == "Foo"


def test_sanity(tmpdir: Any) -> None:
    # Just writes a document and checks nothing crashes...
    doc = ListingDocument(
        [
            # Paragraph
            Paragraph("Hello"),
            # Paragrpah with runs with multiple styles
            Paragraph([Run(style.name, style) for style in RunStyle]),
            # Empty paragraph
            Paragraph(),
            # Table with comments
            ListingTable(
                [
                    ListingLine(Paragraph("Code 1"), Paragraph("Comment 1")),
                    ListingLine(Paragraph("Code 2"), Paragraph("Comment 2")),
                    ListingLine(Paragraph("Code 3"), Paragraph("Comment 3")),
                ]
            ),
            # Table with no comments
            ListingTable(
                [
                    ListingLine(Paragraph("Code 1"), Paragraph("")),
                    ListingLine(Paragraph("Code 2"), Paragraph("")),
                    ListingLine(Paragraph("Code 3"), Paragraph("")),
                ]
            ),
            # Empty table
            ListingTable([]),
        ]
    )

    filename = str(tmpdir.join("test.docx"))
    docx_doc = doc.make_docx_document()
    docx_doc.save(filename)
