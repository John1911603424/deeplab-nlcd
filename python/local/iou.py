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

import numpy as np
import rasterio as rio


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, type=str)
    parser.add_argument('--ground-truth', required=True, type=str)
    return parser


# Given predictions and ground-truth, compute the IOU and other
# statistics.
if __name__ == '__main__':
    args = cli_parser().parse_args()

    with rio.open(args.predictions, 'r') as ds1, rio.open(args.ground_truth, 'r') as ds2:
        data1 = (ds1.read() >= 1).astype(np.int8)
        data2 = ds2.read()
        not_nodata = (data2 != 0)
        data2 = (data2 > 1).astype(np.int8)

    tp = ((data1 * data2 * not_nodata) > 0).sum()
    fp = ((data1 * (data2 == 0) * not_nodata) > 0).sum()
    fn = (((data1 == 0) * data2 * not_nodata) > 0).sum()
    iou = float(tp) / (((data1 + data2) * not_nodata) > 0).sum()
    # om = float(((data2 - data1)*not_nodata >= 1).sum()) / (data2 * not_nodata).sum()
    # com = float(((data1 - data2)*not_nodata >= 1).sum()) / (data2 * not_nodata).sum()
    recall = float(tp)/(tp + fn)
    precision = float(tp)/(tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    print('| | {} | {} | {} | {} |'.format(recall, precision, f1, iou))
