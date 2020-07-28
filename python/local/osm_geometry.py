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
    parser.add_argument('--bbox-geojson', required=True, type=str)
    parser.add_argument('--bizarro-world', action='store_true')
    parser.add_argument('--subject-geojson', required=True, type=str)
    parser.add_argument('--subject-query', required=False,
                        type=str, default='way[building=yes]')
    parser.add_argument('--subject-type', required=False,
                        type=str, default='multipolygons')
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()

    with open(args.bbox_geojson, 'r') as f:
        g = json.load(f).get('features')[0].get('geometry')
        xmin, ymin, xmax, ymax = shapely.geometry.shape(g).bounds

    if args.bizarro_world:
        xdiff = (xmax - xmin)/2.0
        xmean = (xmax + xmin)/2.0
        ydiff = (ymax - ymin)/2.0
        ymean = (ymax + ymin)/2.0
        # xmin = xmean - .618*xdiff
        # xmax = xmean + .618*xdiff
        #ymin = ymean - .618*ydiff
        ymax = ymean + .80*ydiff

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

    # Ensure bbox
    bbox_geojson = '{}.bbox'.format(args.subject_geojson)
    if args.bizarro_world and not os.path.isfile(bbox_geojson):
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
        with open(bbox_geojson, 'w') as f:
            json.dump(bbox_dict, f)
