SMPTE ST 2042-1 VC-2 Pseudocode Language Parsing Software
=========================================================

This repository contains software tools for parsing the pseudocode language
used in the [SMPTE ST 2042-1 (VC-2) professional video
codec](https://www.bbc.co.uk/rd/projects/vc-2) specification documents.

As well as a parser capable of generating Abstract Syntax Trees (ASTs) for
pseudocode listings, tools are also provided for translating the pseudocode
language into Python and also pretty-printed, syntax highlighted Word documents
for inclusions in specifications.

Installation
------------

To install from [PyPI](https://pypi.org/) use:

    $ pip install vc2_pseudocode_parser

Alternatively, to install directly from a checkout of this repository:

    $ python setup.py install --user

This software requires Python 3.6 or later.

To enable support for generating Word documents, `python-docx` must also be
installed (e.g. using `pip install python-docx`).


Example usage
-------------

As an example, consider the following pseudocode listing:


    color_spec(state, video_parameters):
        # (11.4.10.1)
        custom_color_spec_flag = read_bool(state)
        if (custom_color_spec_flag):
            index = read_uint(state)
            preset_color_spec(video_parameters, index)
            # NB: index 0 is 'custom'
            if (index == 0):
                color_primaries(state, video_parameters)  # 11.3.9.1
                color_matrix(state, video_parameters)  # 11.3.9.2
                transfer_function(state, video_parameters)  # 11.3.9.3

This may be converted into Python as follows:

    $ vc2-pseudocode-to-python listing.pc listing.py
    $ cat listing.py
    # This file was automatically translated from a pseudocode listing.

    def color_spec(state, video_parameters):
        """
        (11.4.10.1)
        """
        custom_color_spec_flag = read_bool(state)
        if custom_color_spec_flag:
            index = read_uint(state)
            preset_color_spec(video_parameters, index)
            # NB: index 0 is 'custom'
            if index == 0:
                color_primaries(state, video_parameters)  # 11.3.9.1
                color_matrix(state, video_parameters)  # 11.3.9.2
                transfer_function(state, video_parameters)  # 11.3.9.3

Or into a Word document with tabular listings as follows:

    $ vc2-pseudocode-to-docx listing.pc listing.docx

![Table showing source listing](docs/source/_static/example_docx_table_2.png)

For a more thorough introduction, see the [`vc2_pseudocode_parser`
manual](https://bbc.github.io/vc2_pseudocode_parser/) (also available in [PDF
format](https://bbc.github.io/vc2_pseudocode_parser/vc2_pseudocode_parser_manual.pdf)).


Development
-----------

Testing dependencies may be installed using:

    $ pip install -r requirements-test.txt

Tests are then run using:

    $ py.test

Though typechecking (using [MyPy](https://mypy.readthedocs.io/)) is performed
as part of the test suite, it may be run manually using:

    $ ./run_mypy.sh

Documentation build dependencies can be obtained using:

    $ pip install -r requirements-docs.txt

And the documentation built using:

    $ make -C docs html latexpdf


Experimental software
---------------------

The tools in this repository are experimental in nature. They are part of an
informal thread of work to see how enhanced machine readability of pseudocode
specifications may prove valuable to implementers and specification authors
alike. Contact [Jonathan Heathcote](mailto:jonathan.heathcote@bbc.co.uk) or
[John Fletcher](mailto:john.fletcher@bbc.co.uk) for more information.


License
-------

This software is distributed under the [GNU General Public License version
3](./LICENSE.txt), &copy; BBC 2021.
