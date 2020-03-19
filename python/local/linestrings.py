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
import ctypes
import json
import os
import sys

import numpy as np

from shapely.geometry import Polygon, LineString, mapping, shape
from shapely.ops import cascaded_union, transform


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--simplified', required=False,
                        default='/tmp/simplified.geojson', type=str)
    parser.add_argument('--dn', required=False, type=int, default=255)
    parser.add_argument('--libmedial',
                        default='/workdir/src/libmedial/libmedial.so', type=str)
    return parser


def geojson_to_shapely(feature):
    polygon = Polygon(shape(feature['geometry']).exterior).buffer(0)
    return polygon.simplify(0.01, preserve_topology=True).buffer(0.1).buffer(-0.1)


def polygon_to_linestrings(shape):
    # https://stackoverflow.com/questions/4213095/python-and-ctypes-how-to-correctly-pass-pointer-to-pointer-into-dll
    # https://docs.python.org/3/library/ctypes.html
    libmedial = ctypes.CDLL(libmedial_path)
    libmedial.get_skeleton.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
    ]

    libc = ctypes.CDLL('libc.so.6')
    libc.free.argtypes = [ctypes.c_void_p]

    f = 1000
    input_segments = []
    output_segments = []
    return_data = ctypes.POINTER(ctypes.c_double)()

    boundary = shape.boundary
    if boundary.geom_type == 'LineString':
        coords = list(boundary.coords)
        for i in range(0, len(coords)-1):
            input_segments.append(int(coords[i+0][0] * f))
            input_segments.append(int(coords[i+0][1] * f))
            input_segments.append(int(coords[i+1][0] * f))
            input_segments.append(int(coords[i+1][1] * f))
    elif boundary.geom_type == 'MultiLineString':
        for line in list(boundary):
            coords = list(line.coords)
            for i in range(0, len(coords)-1):
                input_segments.append(int(coords[i+0][0] * f))
                input_segments.append(int(coords[i+0][1] * f))
                input_segments.append(int(coords[i+1][0] * f))
                input_segments.append(int(coords[i+1][1] * f))
    else:
        raise Exception

    input_segments = np.array(input_segments, dtype=np.int64)
    n = libmedial.get_skeleton(
        input_segments.shape[0],
        input_segments.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(return_data))
    for i in range(0, n, 6):
        x1 = return_data[i + 0] / f
        y1 = return_data[i + 1] / f
        x2 = return_data[i + 2] / f
        y2 = return_data[i + 3] / f
        d1 = return_data[i + 4] / f
        d2 = return_data[i + 5] / f
        line_string = LineString([(x1, y1), (x2, y2)])
        if (line_string.within(shape)):
            output_segments.append(line_string)
    libc.free(return_data)

    if len(output_segments) > 0:
        return cascaded_union(output_segments)
    else:
        return None


if __name__ == '__main__':
    args = cli_parser().parse_args()

    libmedial_path = args.libmedial

    with open(args.input) as f:
        geojson_dict = json.load(f)
        features = geojson_dict.get('features')

    features = filter(lambda f: f['properties']['DN'] == args.dn, features)
    shapes = list(map(geojson_to_shapely, features))
    with open(args.simplified, 'w') as f:
        geojson_dict['features'] = [
            {'type': 'Feature', 'geometry': mapping(shape)} for shape in shapes]
        json.dump(geojson_dict, f)
    shapes = list(map(polygon_to_linestrings, shapes))
    shapes = list(filter(lambda u: u is not None, shapes))
    with open(args.output, 'w') as f:
        geojson_dict['features'] = [
            {'type': 'Feature', 'geometry': mapping(shape)} for shape in shapes]
        json.dump(geojson_dict, f)
