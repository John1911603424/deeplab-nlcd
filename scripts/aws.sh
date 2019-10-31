#!/bin/bash

MODEL=$1
shift

aws s3 cp $1 /tmp/image.tif
shift

OUTPUT_PREFIX=$1
shift

mkdir -p /tmp/output /tmp/input
for uri in $*
do
    aws s3 cp $uri /tmp/input/
done

/scripts/local.sh $MODEL /tmp/image.tif /tmp/output/output- $(ls /tmp/input/*.*json)

aws s3 sync /tmp/output/ $OUTPUT_PREFIX
