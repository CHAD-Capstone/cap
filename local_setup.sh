#!/bin/bash

# This script can be sourced to add cap/src to the PYTHONPATH

# Get the folder of the current script
FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_PATH=$FILE_DIR/src

# Add the src folder to the python path
export PYTHONPATH=$SRC_PATH:$PYTHONPATH