import os

from peggie import compile_grammar

__all__ = [
    "grammar_source",
    "grammar",
]


grammar_source = open(
    os.path.join(os.path.dirname(__file__), "grammar.peg"), "r"
).read()
"""The PEG grammar source for the pseudocode language."""

grammar = compile_grammar(grammar_source)
"""The compiled PEG grammar for the pseudocode language."""
