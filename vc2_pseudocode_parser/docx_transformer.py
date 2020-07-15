"""
The :py:mod:`vc2_pseudocode_parser.docx_transformer` module and
``vc2-pseudocode-to-docx`` command line tool automatically translate pseudocode
listings into syntax-highlighted SMPTE-style listings tables in a Word
document.

As an example the following input::

    padding_data(state):                            # Ref
        # Read a padding data block
        for i = 1 to state[next_parse_offset]-13:   # 10.5
            # NB: data is just discarded
            read_byte()                             # A.2.2

Is transformed into the following output:

.. image:: /_static/example_docx_table.png

Note that:

* Syntax highlighting has been applied
    * Keywords are in bold (e.g. ``for`` and ``to``)
    * Labels are italicised (e.g. ``next_parse_offset``)
    * Variables and other values are in normal print
* Spacing is normalised (e.g. around the ``-`` operator)
* End-of-line comments are shown in a right-hand column
* Comments appearing on their own are omitted


Dependencies
============

To generate word documents the `python-docx
<https://python-docx.readthedocs.io/en/latest/>`_ library is used. This is an
optional dependency of the :py:mod:`vc2_pseudocode_parser` software and must be
installed separately, e.g. using::

    $ pip install python-docx


Command-line utility
====================

The ``vc2-pseudocode-to-docx`` command line utility is provided which can
convert a pseudocode listing into a Word document.

Example usage::

    $ vc2-pseudocode-to-docx input.pc output.docx


Python API
==========

The :py:func:`pseudocode_to_docx` utility function may be used to directly
translate pseudocode into a Word document.

.. autofunction:: pseudocode_to_docx

Example usage::

    >>> from vc2_pseudocode_parser.docx_transformer import pseudocode_to_docx

    >>> pseudocode_source = '''
    ...     foo(state, a):
    ...         state[bar] = a + 1
    ... '''
    >>> pseudocode_to_docx(pseudocode_source, "/path/to/output.docx")  # doctest: +SKIP

"""

from typing import Optional, List, Union

from vc2_pseudocode_parser.parser import (
    parse,
    UnaryOp,
    BinaryOp,
    AssignmentOp,
    Listing,
    Function,
    Stmt,
    IfElseStmt,
    ForEachStmt,
    ForStmt,
    WhileStmt,
    FunctionCallStmt,
    AssignmentStmt,
    ReturnStmt,
    Variable,
    Label,
    Subscript,
    Expr,
    ParenExpr,
    UnaryExpr,
    BinaryExpr,
    FunctionCallExpr,
    VariableExpr,
    LabelExpr,
    EmptyMapExpr,
    BooleanExpr,
    NumberExpr,
    EOL,
)

from vc2_pseudocode_parser.docx_generator import (
    ListingDocument,
    Paragraph,
    Run,
    RunStyle,
    ListingTable,
    ListingLine,
)


def code(string: str) -> Paragraph:
    return Paragraph(Run(string, RunStyle.pseudocode))


def keyword(string: str) -> Paragraph:
    return Paragraph(Run(string, RunStyle.pseudocode_keyword))


def fdef(string: str) -> Paragraph:
    return Paragraph(Run(string, RunStyle.pseudocode_fdef))


def label(string: str) -> Paragraph:
    return Paragraph(Run(string, RunStyle.pseudocode_label))


UNARY_OP_TO_PARAGRAPH = {
    UnaryOp("+"): code("+"),
    UnaryOp("-"): code("-"),
    UnaryOp("~"): code("~"),
    UnaryOp("not"): keyword("not"),
}
"""Paragraph to use for each unary operator."""

BINARY_OP_TO_PARAGRPAH = {
    BinaryOp("or"): keyword("or"),
    BinaryOp("and"): keyword("and"),
    BinaryOp("=="): code("=="),
    BinaryOp("!="): code("!="),
    BinaryOp("<"): code("<"),
    BinaryOp("<="): code("<="),
    BinaryOp(">"): code(">"),
    BinaryOp(">="): code(">="),
    BinaryOp("|"): code("|"),
    BinaryOp("^"): code("^"),
    BinaryOp("&"): code("&"),
    BinaryOp("<<"): code("<<"),
    BinaryOp(">>"): code(">>"),
    BinaryOp("+"): code("+"),
    BinaryOp("-"): code("-"),
    BinaryOp("*"): code("*"),
    BinaryOp("//"): code("//"),
    BinaryOp("%"): code("%"),
    BinaryOp("**"): code("**"),
}
"""Paragraph to use for each binary operator."""

ASSIGNMENT_OP_TO_PARAGRAPH = {op: code(op.value) for op in AssignmentOp}
"""Paragraph to use for each assignment operator."""


class DocxTransformer:

    _indent: str

    def __init__(self, indent: str = "   ") -> None:
        self._indent = indent

    def _indent_listing_lines(self, lines: List[ListingLine]) -> List[ListingLine]:
        """Indent the code in each row."""
        return [
            ListingLine(code=code(self._indent) + line.code, comment=line.comment,)
            for line in lines
        ]

    def transform(self, listing: Listing) -> ListingDocument:
        return ListingDocument(
            [
                paragraph_or_table
                for function in listing.functions
                for paragraph_or_table in self._transform_function(function)
            ]
        )

    def _transform_eol_comment(self, eol: Optional[EOL]) -> Paragraph:
        if eol is None or eol.comment is None or eol.comment.string.lstrip("# ") == "":
            return Paragraph("")
        else:
            return Paragraph(eol.comment.string.lstrip("# ").rstrip())

    def _transform_function(
        self, function: Function
    ) -> List[Union[Paragraph, ListingTable]]:
        rows = []

        # Function definiton
        args = ", ".join(a.name for a in function.arguments)
        rows.append(
            ListingLine(
                fdef(function.name) + code(f"({args}):"),
                self._transform_eol_comment(function.eol),
            )
        )

        # Function body
        for stmt in function.body:
            rows.extend(self._indent_listing_lines(self._transform_stmt(stmt)))

        return [ListingTable(rows), Paragraph()]

    def _transform_stmt(self, stmt: Stmt) -> List[ListingLine]:
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
        elif isinstance(stmt, AssignmentStmt):
            return self._transform_assignment_stmt(stmt)
        elif isinstance(stmt, ReturnStmt):
            return self._transform_return_stmt(stmt)
        else:
            raise TypeError(type(stmt))  # Unreachable

    def _transform_if_else_stmt(self, stmt: IfElseStmt) -> List[ListingLine]:
        out = []
        for i, if_branch in enumerate(stmt.if_branches):
            prefix = keyword("if" if i == 0 else "else if")
            condition = self._transform_expr(if_branch.condition)

            block_start = prefix + code(" (") + condition + code("):")

            out.append(
                ListingLine(block_start, self._transform_eol_comment(if_branch.eol))
            )
            for substmt in if_branch.body:
                out.extend(self._indent_listing_lines(self._transform_stmt(substmt)))

        if stmt.else_branch is not None:
            block_start = keyword("else") + code(":")

            out.append(
                ListingLine(
                    block_start, self._transform_eol_comment(stmt.else_branch.eol)
                )
            )
            for substmt in stmt.else_branch.body:
                out.extend(self._indent_listing_lines(self._transform_stmt(substmt)))

        return out

    def _transform_for_each_stmt(self, stmt: ForEachStmt) -> List[ListingLine]:

        for_each_line = keyword("for each")
        for_each_line += code(f" {stmt.variable.name}")
        for_each_line += code(" ") + keyword("in") + code(" ")
        for i, e in enumerate(stmt.values):
            if i != 0:
                for_each_line += code(", ")
            for_each_line += self._transform_expr(e)
        for_each_line += code(":")

        out = []
        out.append(ListingLine(for_each_line, self._transform_eol_comment(stmt.eol)))
        for substmt in stmt.body:
            out.extend(self._indent_listing_lines(self._transform_stmt(substmt)))

        return out

    def _transform_for_stmt(self, stmt: ForStmt) -> List[ListingLine]:
        for_line = keyword("for")
        for_line += code(f" {stmt.variable.name}")
        for_line += code(" =")
        for_line += code(" ") + self._transform_expr(stmt.start)
        for_line += code(" ") + keyword("to")
        for_line += code(" ") + self._transform_expr(stmt.end)
        for_line += code(":")

        out = []
        out.append(ListingLine(for_line, self._transform_eol_comment(stmt.eol)))
        for substmt in stmt.body:
            out.extend(self._indent_listing_lines(self._transform_stmt(substmt)))

        return out

    def _transform_while_stmt(self, stmt: WhileStmt) -> List[ListingLine]:
        while_line = keyword("while")
        while_line += code(" (") + self._transform_expr(stmt.condition) + code("):")

        out = []
        out.append(ListingLine(while_line, self._transform_eol_comment(stmt.eol)))
        for substmt in stmt.body:
            out.extend(self._indent_listing_lines(self._transform_stmt(substmt)))

        return out

    def _transform_function_call_stmt(
        self, stmt: FunctionCallStmt
    ) -> List[ListingLine]:
        call = self._transform_expr(stmt.call)
        return [ListingLine(call, self._transform_eol_comment(stmt.eol))]

    def _transform_assignment_stmt(self, stmt: AssignmentStmt) -> List[ListingLine]:
        out = self._transform_variable(stmt.variable)
        out += code(" ") + ASSIGNMENT_OP_TO_PARAGRAPH[stmt.op] + code(" ")
        out += self._transform_expr(stmt.value)
        return [ListingLine(out, self._transform_eol_comment(stmt.eol))]

    def _transform_return_stmt(self, stmt: ReturnStmt) -> List[ListingLine]:
        return_ = keyword("return")
        value = self._transform_expr(stmt.value)
        return [
            ListingLine(
                return_ + code(" ") + value, self._transform_eol_comment(stmt.eol)
            )
        ]

    def _transform_expr(self, expr: Expr) -> Paragraph:
        if isinstance(expr, ParenExpr):
            return self._transform_paren_expr(expr)
        elif isinstance(expr, UnaryExpr):
            return self._transform_unary_expr(expr)
        elif isinstance(expr, BinaryExpr):
            return self._transform_binary_expr(expr)
        elif isinstance(expr, FunctionCallExpr):
            return self._transform_function_call_expr(expr)
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

    def _transform_paren_expr(self, expr: ParenExpr) -> Paragraph:
        return code("(") + self._transform_expr(expr.value) + code(")")

    def _transform_unary_expr(self, expr: UnaryExpr) -> Paragraph:
        # NB: We assume that ParenExprs have been used to enfore the correct
        # operator precidence rules
        op = UNARY_OP_TO_PARAGRAPH[expr.op]
        space = code(" " if expr.op == UnaryOp.logical_not else "")
        value = self._transform_expr(expr.value)
        return op + space + value

    def _transform_binary_expr(self, expr: BinaryExpr) -> Paragraph:
        # NB: We assume that ParenExprs have been used to enfore the correct
        # operator precidence rules
        lhs = self._transform_expr(expr.lhs)
        op = BINARY_OP_TO_PARAGRPAH[expr.op]
        rhs = self._transform_expr(expr.rhs)
        return lhs + code(" ") + op + code(" ") + rhs

    def _transform_function_call_expr(self, expr: FunctionCallExpr) -> Paragraph:
        out = code(expr.name)
        out += code("(")
        for i, arg in enumerate(expr.arguments):
            if i != 0:
                out += code(", ")
            out += self._transform_expr(arg)
        out += code(")")
        return out

    def _transform_variable_expr(self, expr: VariableExpr) -> Paragraph:
        return self._transform_variable(expr.variable)

    def _transform_label_expr(self, expr: LabelExpr) -> Paragraph:
        return self._transform_label(expr.label)

    def _transform_empty_map_expr(self, expr: EmptyMapExpr) -> Paragraph:
        return code("{}")

    def _transform_boolean_expr(self, expr: BooleanExpr) -> Paragraph:
        return keyword("True" if expr.value else "False")

    def _transform_number_expr(self, expr: NumberExpr) -> Paragraph:
        prefix = {2: "0b", 10: "", 16: "0x"}[expr.display_base]
        format_char = {2: "b", 10: "d", 16: "X"}[expr.display_base]
        digits = "{:0{}{}}".format(expr.value, expr.display_digits, format_char)
        return code(f"{prefix}{digits}")

    def _transform_variable(self, var: Union[Variable, Subscript]) -> Paragraph:
        if isinstance(var, Variable):
            return code(var.name)
        elif isinstance(var, Subscript):
            base = self._transform_variable(var.variable)
            subscript = code("[") + self._transform_expr(var.subscript) + code("]")
            return base + subscript
        else:
            raise TypeError(type(var))  # Unreachable

    def _transform_label(self, lbl: Label) -> Paragraph:
        return label(lbl.name)


def pseudocode_to_docx(pseudocode_source: str, filename: str) -> None:
    """
    Transform a pseudocode listing into a Word (docx) document.

    Will throw a :py:exc:`~vc2_pseudocode_parser.parser.ParseError`
    :py:exc:`.ASTConstructionError` if the supplied pseudocode contains errors.
    """
    pseudocode_ast = parse(pseudocode_source)
    transformer = DocxTransformer()
    listing_document = transformer.transform(pseudocode_ast)
    listing_document.make_docx_document().save(filename)
