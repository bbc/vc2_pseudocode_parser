VC-2 Pseudocode Parsing Software
================================

This manual describes the VC-2 pseudocode parsing software. This software
provides facilities for parsing, translating and formatting the pseudocode
language described in SMPTE ST 2042-1 (VC-2) series of standards documents.

In :ref:`parser` the :py:mod:`vc2_pseudocode_parser.parser` module is
introduced which implements a parser and Abstract Syntax Tree (AST) for the
pseudocode language. This forms the basis of the other tools provided by this
software and also may be used directly if desired.

In :ref:`pseudocode-to-python`, the ``vc2-pseudocode-to-python`` command (and
associated :py:mod:`vc2_pseudocode_parser.python_transformer` Python module)
are introduced. These produce automatic translations of VC-2 pseudocode
listings into valid Python.

In :ref:`pseudocode-to-docx`, the ``vc2-pseudocode-to-docx`` command (and
associated :py:mod:`vc2_pseudocode_parser.docx_transformer` Python module)
are introduced. These generate Word (docx) documents containing
pretty-printed and syntax highlighted versions of a VC-2 pseudocode
listing. Suplimentrary to this, :ref:`docx-generator` gives additional details
of the Word document generation process.

Finally, you can find the source code for :py:mod:`vc2_pseudocode_parser` `on
GitHub <https://github.com/bbc/vc2_pseudocode_parser/>`_.

.. only:: not latex

    .. note::
    
        This documentation is also `available in PDF format
        <https://bbc.github.io/vc2_pseudocode_parser/vc2_pseudocode_parser_manual.pdf>`_.

.. only:: not html

    .. note::
    
        This documentation is also `available to browse online in HTML format
        <https://bbc.github.io/vc2_pseudocode_parser/>`_.

.. toctree::
   :hidden:

   bibliography.rst

.. toctree::
    :maxdepth: 2
    
    parser.rst
    python_transformer.rst
    docx_transformer.rst
    docx_generator.rst
