#!/bin/bash

SCRIPT_LOC=$1
SCRIPT_NAME=$(basename $SCRIPT_LOC)


if echo $SCRIPT_LOC | grep '^s3://' > /dev/null
then
   aws s3 cp $SCRIPT_LOC $SCRIPT_NAME
elif echo $SCRIPT_LOC | grep '^http\(s\|\)://' > /dev/null
then
   wget $SCRIPT_LOC -O $SCRIPT_NAME
fi

shift

PYTHONUNBUFFERED=1 python ./$SCRIPT_NAME $*
