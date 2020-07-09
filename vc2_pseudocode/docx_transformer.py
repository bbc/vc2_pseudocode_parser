"""
Transform a parsed AST representation of a pseudocode program into a Word
document (`*.docx`) with syntax-highlighted, pretty-printed code listings, in
the style of the VC-2 SMPTE standards documents.

NB: only line-end comments are retained in the generated format.
"""

from typing import Optional, List, Union

from vc2_pseudocode.parser import parse

from vc2_pseudocode.operators import UnaryOp, BinaryOp, AssignmentOp

from vc2_pseudocode.docx_generator import (
    ListingDocument,
    Paragraph,
    Run,
    RunStyle,
    ListingTable,
    ListingLine,
)

from vc2_pseudocode.ast import (
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
    PerenExpr,
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
                fdef(function.name) + f"({args}):",
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

            block_start = prefix + " (" + condition + "):"

            out.append(
                ListingLine(block_start, self._transform_eol_comment(if_branch.eol))
            )
            for substmt in if_branch.body:
                out.extend(self._indent_listing_lines(self._transform_stmt(substmt)))

        if stmt.else_branch is not None:
            block_start = keyword("else") + ":"

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
        for_each_line += " " + stmt.variable.name
        for_each_line += " " + keyword("in") + " "
        for i, e in enumerate(stmt.values):
            if i != 0:
                for_each_line += ", "
            for_each_line += self._transform_expr(e)
        for_each_line += ":"

        out = []
        out.append(ListingLine(for_each_line, self._transform_eol_comment(stmt.eol)))
        for substmt in stmt.body:
            out.extend(self._indent_listing_lines(self._transform_stmt(substmt)))

        return out

    def _transform_for_stmt(self, stmt: ForStmt) -> List[ListingLine]:
        for_line = keyword("for")
        for_line += " " + stmt.variable.name
        for_line += " ="
        for_line += " " + self._transform_expr(stmt.start)
        for_line += " " + keyword("to")
        for_line += " " + self._transform_expr(stmt.end)
        for_line += ":"

        out = []
        out.append(ListingLine(for_line, self._transform_eol_comment(stmt.eol)))
        for substmt in stmt.body:
            out.extend(self._indent_listing_lines(self._transform_stmt(substmt)))

        return out

    def _transform_while_stmt(self, stmt: WhileStmt) -> List[ListingLine]:
        while_line = keyword("while")
        while_line += " (" + self._transform_expr(stmt.condition) + "):"

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
        out += " " + ASSIGNMENT_OP_TO_PARAGRAPH[stmt.op] + " "
        out += self._transform_expr(stmt.value)
        return [ListingLine(out, self._transform_eol_comment(stmt.eol))]

    def _transform_return_stmt(self, stmt: ReturnStmt) -> List[ListingLine]:
        return_ = keyword("return")
        value = self._transform_expr(stmt.value)
        return [
            ListingLine(return_ + " " + value, self._transform_eol_comment(stmt.eol))
        ]

    def _transform_expr(self, expr: Expr) -> Paragraph:
        if isinstance(expr, PerenExpr):
            return self._transform_peren_expr(expr)
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

    def _transform_peren_expr(self, expr: PerenExpr) -> Paragraph:
        return "(" + self._transform_expr(expr.value) + ")"

    def _transform_unary_expr(self, expr: UnaryExpr) -> Paragraph:
        # NB: We assume that PerenExprs have been used to enfore the correct
        # operator precidence rules
        op = UNARY_OP_TO_PARAGRAPH[expr.op]
        space = " " if expr.op == UnaryOp.logical_not else ""
        value = self._transform_expr(expr.value)
        return op + space + value

    def _transform_binary_expr(self, expr: BinaryExpr) -> Paragraph:
        # NB: We assume that PerenExprs have been used to enfore the correct
        # operator precidence rules
        lhs = self._transform_expr(expr.lhs)
        op = BINARY_OP_TO_PARAGRPAH[expr.op]
        rhs = self._transform_expr(expr.rhs)
        return lhs + " " + op + " " + rhs

    def _transform_function_call_expr(self, expr: FunctionCallExpr) -> Paragraph:
        out = code(expr.name)
        out += "("
        for i, arg in enumerate(expr.arguments):
            if i != 0:
                out += ", "
            out += self._transform_expr(arg)
        out += ")"
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
            subscript = "[" + self._transform_expr(var.subscript) + "]"
            return base + subscript
        else:
            raise TypeError(type(var))  # Unreachable

    def _transform_label(self, lbl: Label) -> Paragraph:
        return label(lbl.name)


def pseudocode_to_docx(pseudocode_source: str, filename: str) -> None:
    """
    Transform a pseudocode listing into a word document.

    Will throw a :py:exc:`vc2_pseudocode.parser.ParseError`
    :py:exc:`vc2_pseudocode.ast.ASTConstructionError` if the supplied
    pseudocode contains errors.
    """
    pseudocode_ast = parse(pseudocode_source)
    transformer = DocxTransformer()
    listing_document = transformer.transform(pseudocode_ast)
    listing_document.make_docx_document().save(filename)
