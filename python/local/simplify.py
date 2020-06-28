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
import json
import ast

import shapely.geometry


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--simplify', required=False, type=float)
    return parser


# Given a GeoJSON file containing geometry, simplify it (using
# shapely's simplify functionality).
if __name__ == '__main__':
    args = cli_parser().parse_args()

    with open(args.input, 'r') as f:
        geojson = json.load(f)

    def not_zero(feature):
        dn = int(feature.get('properties').get('DN'))
        return dn != 0

    features = geojson.get('features')
    features = list(filter(not_zero, features))
    for feature in features:
        shape = shapely.geometry.shape(feature.get('geometry'))
        if args.simplify:
            shape = shape.buffer(0).simplify(args.simplify)
            feature['geometry'] = shapely.geometry.mapping(shape)
    geojson['features'] = features

    with open(args.output, 'w') as f:
        json.dump(geojson, f)
