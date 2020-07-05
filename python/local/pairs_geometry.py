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
import os

import shapely.geometry


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox-geojson', required=False, type=str)
    parser.add_argument('--bizarro-world', action='store_true')
    parser.add_argument('--correlate-geojson', required=False, type=str)
    parser.add_argument('--correlate-query', required=False,
                        type=str, default='way[highway=*]')
    parser.add_argument('--correlate-type', required=False,
                        type=str, default='lines')
    parser.add_argument('--dataset-name', required=False, type=str)
    parser.add_argument('--metadata-file', required=True, type=str)
    parser.add_argument('--output-directory',
                        required=False, type=str, default='/tmp')
    parser.add_argument('--subject-geojson', required=False, type=str)
    parser.add_argument('--subject-query', required=False,
                        type=str, default='way[building=yes]')
    parser.add_argument('--subject-type', required=False,
                        type=str, default='multipolygons')
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()

    if args.subject_geojson is None or args.correlate_geojson is None or args.bbox_geojson is None:
        assert(args.dataset_name is not None)
        if args.subject_geojson is None:
            args.subject_geojson = '{}/{}.buildings.geojson'.format(
                args.output_directory, args.dataset_name)
        if args.correlate_geojson is None:
            args.correlate_geojson = '{}/{}.roads.geojson'.format(
                args.output_directory, args.dataset_name)
        if args.bbox_geojson is None:
            args.bbox_geojson = '{}/{}.bbox.geojson'.format(
                args.output_directory, args.dataset_name)

    command = 'gdalinfo -proj4 -json {}'.format(args.metadata_file)
    gdalinfo = json.loads(os.popen(command).read())
    [x1, y1] = list(map(float, gdalinfo['cornerCoordinates']['upperLeft']))
    [x2, y2] = list(map(float, gdalinfo['cornerCoordinates']['lowerRight']))
    [xmin, xmax] = sorted([x1, x2])
    [ymin, ymax] = sorted([y1, y2])
    if args.bizarro_world:
        xdiff = xmax - xmin
        xmean = (xmax + xmin)/2.0
        ydiff = ymax - ymin
        ymean = (ymax + ymin)/2.0
        xmin = xmean - .618*xdiff*.5
        xmax = xmean + .618*xdiff*.5
        ymin = ymean - .618*ydiff*.5
        ymax = ymean + .618*ydiff*.5

    # Ensure subject data
    if not os.path.isfile(args.subject_geojson):
        if not os.path.isfile('{}.osm'.format(args.subject_geojson)):
            command = ''.join([
                'wget ',
                '-O {}.osm '.format(args.subject_geojson),
                'http://www.overpass-api.de/api/xapi_meta?',
                '{}'.format(args.subject_query),
                '[bbox={},{},{},{}]'.format(xmin, ymin, xmax, ymax)
            ])
            print(command)
            os.system(command)
        command = ''.join([
            'ogr2ogr -f GeoJSON ',
            '{} {}.osm '.format(args.subject_geojson, args.subject_geojson),
            '{}'.format(args.subject_type)
        ])
        print(command)
        os.system(command)

    # Ensure correlate data
    if not os.path.isfile(args.correlate_geojson):
        if not os.path.isfile('{}.osm'.format(args.correlate_geojson)):
            command = ''.join([
                'wget ',
                '-O {}.osm '.format(args.correlate_geojson),
                'http://www.overpass-api.de/api/xapi_meta?',
                '{}'.format(args.correlate_query),
                '[bbox={},{},{},{}]'.format(xmin, ymin, xmax, ymax)
            ])
            print(command)
            os.system(command)
        command = ''.join([
            'ogr2ogr -f GeoJSON ',
            '{} {}.osm '.format(args.correlate_geojson,
                                args.correlate_geojson),
            '{}'.format(args.correlate_type)
        ])
        print(command)
        os.system(command)

    # Ensure bbox
    if not os.path.isfile(args.bbox_geojson):
        bbox = shapely.geometry.box(xmin, ymin, xmax, ymax)
        bbox_dict = {
            'type': 'Feature',
            'properties': {},
            'geometry': shapely.geometry.mapping(bbox)
        }
        bbox_dict = {
            'type': 'FeatureCollection',
            'features': [bbox_dict]
        }
        with open(args.bbox_geojson, 'w') as f:
            json.dump(bbox_dict, f)
