#!/usr/bin/env python3

# The MIT License (MIT)
# =====================
#
# Copyright © 2020 Azavea
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.


import argparse
import copy
import os

import numpy as np

import rasterio as rio
import scipy.ndimage


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--n', required=False, default=2, type=int)
    parser.add_argument('--inverse-mask', required=False, type=str)
    parser.add_argument('--threshold', required=False, type=float, default=0.0)
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()

    command = 'aws s3 cp {} /tmp/raw.tif'.format(args.input)
    os.system(command)
    with rio.open('/tmp/raw.tif', 'r') as ds:
        profile = copy.copy(ds.profile)
        data = ds.read()

    # Threshold
    data = (data > args.threshold).astype(np.uint8)

    # Close
    element = np.ones((args.n, args.n))
    data[0] = scipy.ndimage.binary_dilation(data[0], structure=element)
    data[0] = scipy.ndimage.binary_erosion(data[0], structure=element)

    # Apply inverse mask
    if args.inverse_mask is not None:

        command = 'aws s3 cp {} /tmp/mask.geojson'.format(args.inverse_mask)
        os.system(command)

        profile['compress'] = None
        profile['dtype'] = 'uint8'
        with rio.open('/tmp/uncompressed.tif', 'w', **profile) as ds:
            ds.write(data)
        command = 'gdal_rasterize -burn 1 -i /tmp/mask.geojson /tmp/uncompressed.tif'
        os.system(command)

        with rio.open('/tmp/uncompressed.tif', 'r') as ds:
            data = ds.read()

    # Write final image
    profile['compress'] = 'lzw'
    profile['predictor'] = 2
    profile['dtype'] = 'uint8'
    with rio.open('/tmp/compressed.tif', 'w', **profile) as ds:
        ds.write(data)
    command = 'aws s3 cp /tmp/compressed.tif {}'.format(args.output)
    os.system(command)
