#!/bin/bash

ln /tmp/deeplab_"$1".pth /tmp/deeplab.pth
shift

cp "$1" /tmp/mul.tif
shift

/scripts/deeplab_inference.py --architecture resnet18-binary --classes 2 --bands 1 2 3 4 5 6 7 8 9 10 11 12 --libchips ??? --model ??? --inference-img ???

OUTPUT_PREFIX = $1
shift

/scripts/elaborate.py --raster /tmp/pred.tif --output-prefix $OUTPUT_PREFIX --geojson $*
