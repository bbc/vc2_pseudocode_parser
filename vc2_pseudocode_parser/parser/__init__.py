"""
The :py:mod:`vc2_pseudocode_parser.parser` module contains a parser and
associated Abstract Syntax Tree (AST) representation for the pseudocode
language used within the VC-2 specifications [VC2]_.

A quick-start example illustrating much of the pseudocode syntax and basic
usage of this module is given below::

    >>> from vc2_pseudocode_parser.parser import parse

    >>> source = '''
    ...     some_function(arg1, arg2, arg3):
    ...         # Assignments
    ...         hex_foo = 0xF00
    ...         sum = arg1 + arg2 + arg3
    ...
    ...         # If statements
    ...         if (sum > 0):
    ...             sign = 1
    ...         else if (sum < 0):
    ...             sign = -1
    ...         else:
    ...             sign = 0
    ...
    ...         # For-each: loop over fixed set of values
    ...         sum2 = 0
    ...         for each value in arg1, arg2, arg3:
    ...             sum2 += value
    ...
    ...         # For: loop over range of integers
    ...         sum_1_to_100 = 0
    ...         for n = 1 to 100:
    ...             sum_1_to_100 += n
    ...
    ...         # While loop
    ...         count = 0
    ...         while (sum > 0):
    ...             sum //= 2
    ...             count += 1
    ...
    ...         # Maps (like Python's dictionaries
    ...         map = {}
    ...
    ...         # Maps subscripted with labels
    ...         map[foo] = 123
    ...         map[bar] = 321
    ...
    ...         # Labels are first-class values (and are defined by their first use)
    ...         label = baz
    ...         map[label] = 999
    ...
    ...         # Function calls
    ...         foo(map)
    ...
    ...         # Return from functions
    ...         return count
    ... '''

    >>> ast = parse(source)

    >>> ast.functions[0].name
    'some_function'

    >>> assignment = ast.functions[0].body[0]
    >>> assignment.variable
    Variable(offset=68, name='hex_foo')
    >>> assignment.value
    NumberExpr(offset=78, offset_end=83, value=3840, display_base=16, display_digits=3)


Parser
======

A pseudocode snippet may be parsed into an Abstract Syntax Tree (AST) using the
:py:func:`.parse` function:

.. autofunction:: vc2_pseudocode_parser.parser.parse

Parsing failures will result of one of the exceptions below being raised. In
all cases, the :py:class:`str` representation of these errors produces a
user-friendly description of the problem.

For example::

    >>> from vc2_pseudocode_parser.parser import ParseError, ASTConstructionError

    >>> try:
    ...     ast = parse("foo(): return (a + 3")
    ... except (ParseError, ASTConstructionError) as e:
    ...     print(str(e))
    At line 1 column 21:
        foo(): return (a + 3
                            ^
    Expected ')' or <operator>

.. py:exception:: ParseError

    Re-exported from :py:exc:`.peggie.ParseError`.

.. autoexception:: ASTConstructionError

.. autoexception:: LabelUsedAsVariableNameError
    :show-inheritance:

.. autoexception:: CannotSubscriptLabelError
    :show-inheritance:

Abstract Syntax Tree (AST)
==========================

The parser generates a fairly detailed AST containing both semantic and some
non-semantic features of the source such as explicit uses of parentheses,
comments and vertical whitespace. Nodes in the AST also include character
indices of the corresponding source code enabling the construction of helpful
error messages.

Every node in the AST is a subclass of the :py:class:`.ASTNode` base class:

.. autoclass:: vc2_pseudocode_parser.parser.ast.ASTNode
    :members:
    :undoc-members:


AST Root (:py:class:`.Listing`)
-------------------------------

The root element of a complete AST is :py:class:`.Listing`:

.. autoclass:: vc2_pseudocode_parser.parser.ast.Listing
    :members:
    :exclude-members: offset, offset_end

This in turn is principally made up of a list of :py:class:`.Function` nodes:

.. autoclass:: vc2_pseudocode_parser.parser.ast.Function
    :members:
    :exclude-members: offset, offset_end

Statements
----------

AST nodes representing statements in the AST are subclasses of
:py:class:`.Stmt`.

.. autoclass:: vc2_pseudocode_parser.parser.ast.Stmt
    :members:
    :exclude-members: offset, offset_end

If-else-if-else statements are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.IfElseStmt
    :members:
    :exclude-members: offset, offset_end

.. autoclass:: vc2_pseudocode_parser.parser.ast.IfBranch
    :members:
    :exclude-members: offset, offset_end

.. autoclass:: vc2_pseudocode_parser.parser.ast.ElseBranch
    :members:
    :exclude-members: offset, offset_end

For-each loops are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.ForEachStmt
    :members:
    :exclude-members: offset, offset_end

For loops are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.ForStmt
    :members:
    :exclude-members: offset, offset_end

While loops are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.WhileStmt
    :members:
    :exclude-members: offset, offset_end

Function call statements are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.FunctionCallStmt
    :members:
    :exclude-members: offset, offset_end

Return statements are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.ReturnStmt
    :members:
    :exclude-members: offset, offset_end

Assignment statements are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.AssignmentStmt
    :members:
    :exclude-members: offset, offset_end

The following assignment operators are defined:

.. autoclass:: vc2_pseudocode_parser.parser.operators.AssignmentOp
    :members:
    :undoc-members:


Expressions
-----------

AST nodes representing expressions in the AST are subclasses of
:py:class:`~vc2_pseudocode_parser.parser.ast.Expr`.

.. autoclass:: vc2_pseudocode_parser.parser.ast.Expr
    :members:
    :exclude-members: offset, offset_end

Unary expressions are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.UnaryExpr
    :members:
    :exclude-members: offset, offset_end

With unary operators enumerated as:

.. autoclass:: vc2_pseudocode_parser.parser.operators.UnaryOp
    :members:
    :undoc-members:

Binary expressions are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.BinaryExpr
    :members:
    :exclude-members: offset, offset_end

With binary operators enumerated as:

.. autoclass:: vc2_pseudocode_parser.parser.operators.BinaryOp
    :members:
    :undoc-members:

Calls to functions are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.FunctionCallExpr
    :members:
    :exclude-members: offset, offset_end

Uses of variables and subscripted variables are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.VariableExpr
    :members:
    :exclude-members: offset, offset_end

Value literals are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.BooleanExpr
    :members:
    :exclude-members: offset, offset_end

.. autoclass:: vc2_pseudocode_parser.parser.ast.NumberExpr
    :members:
    :exclude-members: offset, offset_end

.. autoclass:: vc2_pseudocode_parser.parser.ast.EmptyMapExpr
    :members:
    :exclude-members: offset, offset_end

.. autoclass:: vc2_pseudocode_parser.parser.ast.LabelExpr
    :members:
    :exclude-members: offset, offset_end

For parenthesised expressions, e.g. ``(1 + 2)``, the presence of the
parentheses is explicitly marked in the AST. While this is not semantically
important (since evaluation order is explicit in an AST) it may be helpful in
retaining parentheses added for human legibility when translating the
pseudocode into other forms.

.. autoclass:: vc2_pseudocode_parser.parser.ast.ParenExpr
    :members:
    :exclude-members: offset, offset_end


Variables and subscripts
------------------------

A :py:class:`.Variable` is defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.Variable
    :members:
    :exclude-members: offset, offset_end

Variables may be subscripted (multiple times) and this is represented by a
nesting of :py:class:`.Subscript` objects:

.. autoclass:: vc2_pseudocode_parser.parser.ast.Subscript
    :members:
    :exclude-members: offset, offset_end

Labels
------

Labels are defined as follows:

.. autoclass:: vc2_pseudocode_parser.parser.ast.Label
    :members:
    :exclude-members: offset, offset_end

.. note::

    Labels and variables are syntactically ambiguous in the pseudocode language
    but are disambiguated in the AST. Names in the pseudocode are deemed to be
    variables if they correspond with function arguments, loop variables or
    assignment targets within a function's namespace. All other names are
    deemed to be labels.

Comments and vertical whitespace
--------------------------------

All comments and vertical whitespace (i.e. blank lines) are captured by the
AST. This enables these non-semantic components to be retained in language
translations.

.. autoclass:: vc2_pseudocode_parser.parser.ast.Comment
    :members:
    :exclude-members: offset, offset_end

.. autoclass:: vc2_pseudocode_parser.parser.ast.EmptyLine
    :members:
    :exclude-members: offset, offset_end

Simple statements are syntactically terminated by an optional comment followed
by a newline and then a number of empty or comment-only lines. These details
are captured by the :py:class:`.EOL` node.

.. autoclass:: vc2_pseudocode_parser.parser.ast.EOL
    :members:
    :exclude-members: offset, offset_end

For example, given a snippet as follows::

    foo() # Comment 0

    bar() # Comment 1
    # Comment 2

    # Comment 3

    baz()  # Comment 4

Here the function call ``bar()`` would be captured by a
:py:class:`.FunctionCallStmt`. The :py:attr:`.FunctionCallStmt.eol` would
contain an :py:class:`.EOL` with :py:attr:`.EOL.comment` set to a
:py:class:`.Comment` containing ``# Comment 1`` and :py:attr:`.EOL.empty_lines`
the following four lines (and their comments).

Function declarations, ``if``, ``else if``, ``else``, ``for each``, ``for`` and
``while`` headings are separated from their bodies by a ``:`` and optional
:py:class:`.EOL`. When the ``:`` is followed by a newline, an :py:class:`.EOL`
object will be given, but for in-line definitions, the :py:class:`.EOL` will be
omitted. For example::

    func1():
        foo()

    func2(): foo()

Here the :py:class:`.Function` for ``func1`` will have :py:attr:`.Function.eol`
set to an :py:class:`.EOL` node for ``func2`` it will be ``None``.


Grammar
=======

The pseudocode language grammar can be described by its
:py:mod:`~peggie` grammar:

..
    NB: The 'include' below is relative to the /docs/source directory, an
    unfortunate wrinkle in Sphinx/RST...

.. include:: ../../vc2_pseudocode_parser/parser/grammar.peg
    :literal:


Operator precedence and associativity tables
============================================

A table of operator precedence and associativities is also provided which may
be useful for, for example, producing pretty-printed outputs (with excess
parentheses removed).

.. autodata:: vc2_pseudocode_parser.parser.operators.OPERATOR_PRECEDENCE_TABLE
    :annotation: = {operator: int, ...}

.. autodata:: vc2_pseudocode_parser.parser.operators.OPERATOR_ASSOCIATIVITY_TABLE
    :annotation: = {operator: associativity, ...}

.. autoclass:: vc2_pseudocode_parser.parser.operators.Associativity
    :members:
    :undoc-members:

"""

from vc2_pseudocode_parser.parser.grammar import *
from vc2_pseudocode_parser.parser.operators import *
from vc2_pseudocode_parser.parser.ast import *
from vc2_pseudocode_parser.parser.parser import *

# NB: These names are explicitly re-exported here because mypy in strict mode
# does not allow implicit re-exports. The completeness of this list is tested
# by the test suite.
__all__ = [  # noqa: F405
    # parser.*
    "parse",
    "ParseError",
    # ast.*
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
    "ParenExpr",
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
    # operators.*
    "UnaryOp",
    "BinaryOp",
    "OPERATOR_PRECEDENCE_TABLE",
    "Associativity",
    "OPERATOR_ASSOCIATIVITY_TABLE",
    "AssignmentOp",
    # grammar.*
    "grammar_source",
    "grammar",
]
