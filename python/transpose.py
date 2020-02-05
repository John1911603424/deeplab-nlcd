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
import json
from typing import *

import shapely.geometry  # type: ignore
import shapely.ops  # type: ignore


def footprint_flip(x: float, y: float) -> Tuple[float, float]:
    """A transform function to swap the coordinates of a shapely geometry

    Arguments:
        x {float} -- The original x-coordinate
        y {float} -- The original y-coordinate

    Returns:
        Tuple[float, float] -- A tuple with the coordinates swapped
    """
    v = (y, x)
    return v


def cli_parser() -> argparse.ArgumentParser:
    """Return a command line argument parser

    Returns:
        argparse.ArgumentParser -- A parser object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()

    with open(args.input, 'r') as f:
        feature_collection = json.load(f)

    for feature in feature_collection['features']:
        geometry = copy.copy(feature['geometry'])
        shape = shapely.geometry.shape(geometry)
        shape = shapely.ops.transform(footprint_flip, shape)
        geometry = shapely.geometry.mapping(shape)
        feature['geometry'] = geometry

    with open(args.output, 'w') as f:
        json.dump(feature_collection, f)
