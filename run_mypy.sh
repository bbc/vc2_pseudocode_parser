#!/bin/bash

shopt -s globstar

DIR="$(dirname "$0")"

mypy \
    --config-file \
    $DIR/mypy.ini \
    $DIR/tests/**/*.py \
    $DIR/vc2_pseudocode_parser \
    "$@"
