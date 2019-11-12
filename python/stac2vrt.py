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
import ctypes
import functools
import json
import os
from typing import *
from urllib.parse import urlparse

import boto3  # type: ignore
import numpy as np  # type: ignore
import pyproj  # type: ignore
import pystac  # type: ignore
import rasterio as rio  # type: ignore
import rasterio.features  # type: ignore
import requests
import shapely.geometry  # type: ignore
import shapely.ops  # type: ignore

if 'CURL_CA_BUNDLE' not in os.environ:
    os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'


def cli_parser() -> argparse.ArgumentParser:
    """Return a command line argument parser

    Returns:
        argparse.ArgumentParser -- A parser object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--geojson-crs', default='+init=epsg:4326', type=str)
    parser.add_argument('--input', required=True, type=str)
    return parser


def requests_read_method(uri: str) -> str:
    """A reader method for PyStac that supports http and s3

    Arguments:
        uri {str} -- The URI

    Returns:
        str -- The string found at that URI
    """
    parsed = urlparse(uri)
    if parsed.scheme.startswith('http'):
        return requests.get(uri).text
    elif parsed.scheme.startswith('s3'):
        parsed2 = urlparse(uri, allow_fragments=False)
        bucket = parsed2.netloc
        prefix = parsed2.path.lstrip('/')
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, prefix)
        return obj.get()['Body'].read().decode('utf-8')
    else:
        return pystac.STAC_IO.default_read_text_method(uri)


pystac.STAC_IO.read_text_method = requests_read_method


def render_label_item(item: pystac.label.LabelItem) -> None:
    assets = item.assets
    assert(len(assets) == 1)
    json_uri = next(iter(assets.values())).href
    json_str = requests_read_method(json_uri)
    label_vectors = json.loads(json_str)
    label_features = label_vectors.get('features')

    if len(label_features) > 0:
        sources = list(item.get_sources())
        assert(len(sources) == 1)
        source_assets = sources[0].assets
        assert(len(source_assets) == 1)
        imagery_uri = next(iter(source_assets.values())).href
        imagery_uri.replace('s3://', '/vsis3/')

        # Read information from imagery
        with rio.open(imagery_uri) as input_ds:
            profile = copy.copy(input_ds.profile)
            profile.update(
                dtype=np.uint8,
                count=1,
                compress='lzw',
                nodata=0
            )
            imagery_crs = input_ds.crs.to_proj4()
            imagery_transform = input_ds.transform

            projection = functools.partial(
                pyproj.transform,
                pyproj.Proj(args.geojson_crs),
                pyproj.Proj(imagery_crs)
            )

        # Rasterize and write label data
        with rio.open('/tmp/{}.tif'.format(item.id), 'w', **profile) as output_ds:
            shapes = []
            rasterized_labels = np.zeros((512, 512), dtype=np.uint8)
            for feature in label_features:
                shape = shapely.geometry.shape(feature.get('geometry'))
                shape = shapely.ops.transform(projection, shape)
                shapes.append(shape)
            rasterio.features.rasterize(
                shapes, fill=0, default_value=1, out=rasterized_labels, transform=imagery_transform)
            output_ds.write(rasterized_labels, indexes=1)
    else:
        print('skip')


if __name__ == '__main__':
    args = cli_parser().parse_args()

    catalog = pystac.Catalog.from_file(args.input)
    for collection in catalog.get_children():
        if 'imagery' in str.lower(collection.description):
            imagery_collection = collection
        elif 'label' in str.lower(collection.description):
            label_collection = collection

    label_items = label_collection.get_items()

    liboverlaps = ctypes.CDLL('/tmp/liboverlaps.so')
    liboverlaps.query.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    liboverlaps.query.restype = ctypes.c_double
    liboverlaps.insert.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]

    liboverlaps.add_tree()
    item_lists = [[]]

    for item in label_items:
        (xmin, ymin, xmax, ymax) = shapely.geometry.shape(item.geometry).bounds
        inserted = False
        for i in range(len(item_lists)):
            percentage_new = liboverlaps.query(i, xmin, ymin, xmax, ymax)
            print(percentage_new, i)
            if percentage_new > 0.95:
                liboverlaps.insert(i, xmin, ymin, ctypes.c_double(xmax), ctypes.c_double(ymax))
                item_lists[i].append(item)
                inserted = True
                break
        if not inserted:
            print(liboverlaps.add_tree())
            item_lists.append([])

    # render_label_item(item)
