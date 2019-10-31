#!/bin/bash

MODEL=$1
shift
ln /tmp/deeplab_"$MODEL".pth /tmp/deeplab.pth

cp "$1" /tmp/mul.tif
shift

if [ "$MODEL" == "sample_water1" ]; then
    /scripts/deeplab_inference.py --architecture resnet18-binary --bands 1 2 3 4 5 6 7 8 9 10 11 12 --libchips ??? --model ??? --inference-img ??? --window-size 128 --classes 1
elif [ "$MODEL" == "sample_water2" ]; then
    /scripts/deeplab_inference.py --architecture cheaplab-binary --bands 1 2 3 4 5 6 7 8 9 10 11 12 --libchips ??? --model ??? --inference-img ??? --warmup-window-size 128 --window-size 512 --classes 1
elif [ "$MODEL" == "sample_buildings" ]; then
    /scripts/deeplab_inference.py --architecture resnet18-binary --bands 2 3 4 6 8 --libchips ??? --model ??? --inference-img ??? --window-size 128 --classes 1
fi

OUTPUT_PREFIX=$1
shift

/scripts/elaborate.py --raster /tmp/pred-raw.tif --output-prefix "$OUTPUT_PREFIX" --geojson $*
