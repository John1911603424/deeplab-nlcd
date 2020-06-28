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

import numpy as np
import rasterio as rio


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', required=True, nargs='+', type=str)
    parser.add_argument('--prediction-nd', required=False, type=int)
    parser.add_argument('--imagery', required=True, nargs='+', type=str)
    parser.add_argument('--imagery-nd', required=False, type=int)
    parser.add_argument('--data', required=True, type=str)
    return parser


# Given predictions, imagery, and the means and standard deviations of
# the two prediction classes, compute the percentage of pixels
# predicted to be foreground.  The imagery is needed solely for its
# nodata mask.
if __name__ == '__main__':
    args = cli_parser().parse_args()

    background_count = 0
    foreground_count = 0

    args.data = list(map(float, args.data.split(' ')))
    [μ0, σ0, μ1, σ1] = args.data
    midpoint = (μ0*σ1 + μ1*σ0)/(σ0 + σ1)
    print('midpoint={}'.format(midpoint))

    for (imagery, prediction) in zip(args.imagery, args.prediction):

        print('imagery={} prediction={}'.format(imagery, prediction))

        # Open "predictions"
        with rio.open(prediction, 'r') as ds:
            prediction_profile = copy.deepcopy(ds.profile)
            prediction_data = ds.read().astype(np.float64)
        if args.prediction_nd is not None:
            nodata_mask = (prediction_data[0] ==
                           args.prediction_nd).astype(np.uint8)
        else:
            nodata_mask = np.zeros_like(prediction_data[0], dtype=np.uint8)

        # Open imagery (for nodata mask)
        if args.imagery_nd is not None:
            with rio.open(imagery, 'r') as ds:
                nodata_mask += (ds.read(1) == args.imagery_nd).astype(np.uint8)
        else:
            nodata_mask += np.zeros_like(
                prediction_data[0].shape, dtype=np.uint8)

        data_mask = (nodata_mask == 0)
        background_count += (data_mask*(prediction_data < midpoint)).sum()
        foreground_count += (data_mask*(prediction_data >= midpoint)).sum()

    foreground_count = float(foreground_count)
    percentage = foreground_count / (foreground_count + background_count)

    print('percentage={}'.format(percentage))
