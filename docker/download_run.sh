#!/bin/bash

SCRIPT_LOC=$1
SCRIPT_NAME=$(basename $SCRIPT_LOC)


aws s3 cp $SCRIPT_LOC $SCRIPT_NAME

shift

PYTHONUNBUFFERED=1 python ./$SCRIPT_NAME $*
