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
import os

import rasterio as rio


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-file', required=True, type=str)
    parser.add_argument('--bbox-geojson', required=True, type=str)
    parser.add_argument('--subject-geojson', required=True, type=str)
    parser.add_argument('--correlate-geojson', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()

    command = 'gdalinfo -proj4 -json {}'.format(args.metadata_file)
    gdalinfo = json.loads(os.popen(command).read())
    [width, height] = list(map(int, gdalinfo['size']))
    [x1, y1] = list(map(float, gdalinfo['cornerCoordinates']['upperLeft']))
    [x2, y2] = list(map(float, gdalinfo['cornerCoordinates']['lowerRight']))
    [xmin, xmax] = sorted([x1, x2])
    [ymin, ymax] = sorted([y1, y2])

    if True:
        # Bounding box
        command = ''.join([
            'gdal_rasterize ',
            '-burn 1 ',
            '-init 0 ',
            '-at ',
            '-te {} {} {} {} '.format(xmin, ymin, xmax, ymax),
            '-ts {} {} '.format(width, height),
            '-ot Byte ',
            '{} /tmp/bbox.tif'.format(args.bbox_geojson)
        ])
        os.system(command)

        # Subject
        command = ''.join([
            'gdal_rasterize ',
            '-burn 1 ',
            '-init 0 ',
            '-at ',
            '-te {} {} {} {} '.format(xmin, ymin, xmax, ymax),
            '-ts {} {} '.format(width, height),
            '-ot Byte ',
            '{} /tmp/subject1.tif'.format(args.subject_geojson)
        ])
        os.system(command)

        # Buffered subject
        command = ''.join([
            'gdal_rasterize ',
            '-burn 1 ',
            '-init 0 ',
            '-at ',
            '-te {} {} {} {} '.format(xmin, ymin, xmax, ymax),
            '-ts {} {} '.format(width // 16, height // 16),
            '-ot Byte ',
            '{} /tmp/subject2.tif'.format(args.subject_geojson)
        ])
        os.system(command)

        # Full-sized buffered subject
        command = ''.join([
            'gdalwarp ',
            '-r near '
            '-ts {} {} '.format(width, height),
            '/tmp/subject2.tif /tmp/subject3.tif'
        ])
        os.system(command)

        # Correlate
        command = ''.join([
            'gdal_rasterize ',
            '-burn 1 ',
            '-init 0 ',
            '-at ',
            '-te {} {} {} {} '.format(xmin, ymin, xmax, ymax),
            '-ts {} {} '.format(width, height),
            '-ot Byte ',
            '{} /tmp/correlate1.tif'.format(args.correlate_geojson)
        ])
        os.system(command)

        # Buffered correlate
        command = ''.join([
            'gdal_rasterize ',
            '-burn 1 ',
            '-init 0 ',
            '-at ',
            '-te {} {} {} {} '.format(xmin, ymin, xmax, ymax),
            '-ts {} {} '.format(width // 32, height // 32),
            '-ot Byte ',
            '{} /tmp/correlate2.tif'.format(args.correlate_geojson)
        ])
        os.system(command)

        # Full-sized buffered correlate
        command = ''.join([
            'gdalwarp ',
            '-r near '
            '-ts {} {} '.format(width, height),
            '/tmp/correlate2.tif /tmp/correlate3.tif'
        ])
        os.system(command)

    with rio.open('/tmp/bbox.tif', 'r') as ds1, rio.open('/tmp/subject1.tif', 'r') as ds2:
        profile = copy.deepcopy(ds1.profile)
        data = ds1.read() + ds2.read()

    with rio.open('/tmp/correlate3.tif', 'r') as ds1, rio.open('/tmp/subject3.tif', 'r') as ds2:
        nodata = ((ds1.read() - ds2.read()) == 1)

    data[nodata] = 0
    profile.update(
        nodata=0,
        sparse_ok=True,
        tiled=True,
        blockxsize=32,
        blockysize=32,
        compress='deflate',
        predictor=2,
    )
    with rio.open(args.output, 'w', **profile) as ds:
        ds.write(data)
