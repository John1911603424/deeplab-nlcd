#!/bin/bash

ln /tmp/deeplab_"$1".pth /tmp/deeplab.pth
shift

aws s3 cp "$1" /tmp/mul.tif
shift

/scripts/deeplab_inference.py --architecture resnet18-binary --classes 2 --bands 1 2 3 4 5 6 7 8 9 10 11 12 --libchips ??? --model ??? --inference-img ???

mkdir -p /tmp/output /tmp/input

OUTPUT_PREFIX = $1
shift

for uri in $*
do
    aws s3 cp $uri /tmp/input/
done

/scripts/elaborate.py --raster /tmp/pred.tif --output-prefix /tmp/output/ --geojson $(ls /tmp/input/*.*json)

aws s3 sync /tmp/output/ $OUTPUT_PREFIX
