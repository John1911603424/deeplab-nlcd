#!/usr/bin/env python3

# The MIT License (MIT)
# =====================
#
# Copyright © 2019 Azavea
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
import functools
import json
from typing import *

import numpy as np  # type: ignore
import pyproj  # type: ignore
import rasterio as rio  # type: ignore
import rasterio.features  # type: ignore
import shapely.geometry  # type: ignore
import shapely.ops  # type: ignore


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug-output', action='store_true')
    parser.add_argument('--geojson-crs', default='+init=epsg:4326', type=str)
    parser.add_argument('--geojson', required=True, nargs='+', type=str)
    parser.add_argument('--output-prefix', type=str)
    parser.add_argument('--raster-band', default=1, type=int)
    parser.add_argument('--raster', required=True, type=str)
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()

    with rio.open(args.raster) as raster_ds:
        profile = copy.copy(raster_ds.profile)
        profile.update(
            dtype=rio.int32,
            count=1,
            compress='lzw'
        )

        raster_data = raster_ds.read(args.raster_band)
        raster_crs = raster_ds.crs.to_proj4()
        raster_transform = raster_ds.transform

    projection = functools.partial(
        pyproj.transform,
        pyproj.Proj(args.geojson_crs),
        pyproj.Proj(raster_crs)
    )

    for filename in args.geojson:
        shapes = []
        rasterized_shapes = np.zeros(raster_data.shape, dtype=np.int32)

        with open(filename) as infile:
            vector_data = json.load(infile)
        features = vector_data.get('features')
        for feature in features:
            s1 = shapely.geometry.shape(feature.get('geometry'))
            s2 = shapely.ops.transform(projection, s1)
            shapes.append(s2)
        shapes = list(zip(shapes, range(1, len(shapes) + 1)))

        rasterio.features.rasterize(
            shapes, out=rasterized_shapes, transform=raster_transform)

        if args.debug_output:
            with rio.open('{}.tif'.format(filename), 'w', **profile) as output_ds:
                output_ds.write(rasterized_shapes, indexes=1)

        for feature, (_, i) in zip(features, shapes):
            shape_mask = (rasterized_shapes == i)
            if not 'properties' in feature:
                feature['properties'] = {}
            properties = feature.get('properties')
            count = shape_mask.sum()
            score = (raster_data * shape_mask).sum() / float(count)
            properties['score'] = float(score)
            properties['count'] = int(count)

        if args.output_prefix:
            base_filename = filename.split('/')[-1]
            new_filename = '{}{}'.format(args.output_prefix, base_filename)
            with open(new_filename, 'w') as outfile:
                json.dump(vector_data, outfile, indent=4)
