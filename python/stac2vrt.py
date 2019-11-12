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
import concurrent.futures
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
    parser.add_argument('--local-prefix', default=None, type=str)
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


def render_label_item(item: pystac.label.LabelItem) -> Optional[Tuple[str, str]]:
    assets = item.assets
    assert(len(assets) == 1)
    json_uri = next(iter(assets.values())).href
    json_str = requests_read_method(json_uri)
    label_vectors = json.loads(json_str)
    label_features = label_vectors.get('features')

    if len(label_features) > 0 or len(item.geometry) > 0:
        sources = list(item.get_sources())
        assert(len(sources) == 1)
        source_assets = sources[0].assets
        assert(len(source_assets) == 1)
        imagery_uri = next(iter(source_assets.values())).href
        imagery_uri = imagery_uri.replace('s3://', '/vsis3/')

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
        filename = '/tmp/{}.tif'.format(item.id)
        with rio.open(filename, 'w', **profile) as output_ds:
            rasterized_labels = np.zeros((512, 512), dtype=np.uint8)
            shapes = []

            shape = shapely.geometry.shape(item.geometry)
            shape = shapely.ops.transform(projection, shape)
            shapes.append((shape, 1))

            for feature in label_features:
                shape = shapely.geometry.shape(feature.get('geometry'))
                shape = shapely.ops.transform(projection, shape)
                shapes.append((shape, 2))

            rasterio.features.rasterize(
                shapes, out=rasterized_labels, transform=imagery_transform)

            output_ds.write(rasterized_labels, indexes=1)

        return (imagery_uri, filename)
    else:
        return None


def render_item_list(t: Tuple[int, List[pystac.label.LabelItem]]) -> None:
    (i, item_list) = t
    imagery_txt = '/tmp/{}-imagery.txt'.format(i)
    label_txt = '/tmp/{}-label.txt'.format(i)
    with open(imagery_txt, 'w') as f, open(label_txt, 'w') as g:
        for item in item_list:
            print('rendering item {} from list {}'.format(item, i))
            (imagery, label) = render_label_item(item)
            f.write(imagery + '\n')
            g.write(label + '\n')


if __name__ == '__main__':
    args = cli_parser().parse_args()

    postfix = '/catalog.json'
    if args.local_prefix and args.input.endswith(postfix):
        local_prefix = args.local_prefix
        remote_prefix = args.input[0:-len(postfix) + 1]

        def requests_read_method_local(uri: str) -> str:
            if uri.endswith('json'):
                uri = uri.replace(remote_prefix, local_prefix)
            return requests_read_method(uri)

        pystac.STAC_IO.read_text_method = requests_read_method_local
    else:
        pystac.STAC_IO.read_text_method = requests_read_method

    catalog = pystac.Catalog.from_file(args.input)
    for collection in catalog.get_children():
        if 'label' in str.lower(collection.description):
            label_collection = collection

    label_items = label_collection.get_items()

    liboverlaps = ctypes.CDLL('/tmp/liboverlaps.so')
    liboverlaps.query.argtypes = [
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    liboverlaps.query.restype = ctypes.c_double
    liboverlaps.insert.argtypes = [
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]

    liboverlaps.add_tree()
    item_lists: List[List[pystac.label.LabelItem]] = [[]]

    for item in label_items:
        print('reading item {}'.format(item))
        (xmin, ymin, xmax, ymax) = shapely.geometry.shape(item.geometry).bounds
        inserted = False
        for i in range(len(item_lists)):
            percentage_new = liboverlaps.query(i, xmin, ymin, xmax, ymax)
            if percentage_new > 0.95:
                liboverlaps.insert(i, xmin, ymin, xmax, ymax)
                item_lists[i].append(item)
                inserted = True
                break
        if not inserted:
            liboverlaps.add_tree()
            item_lists.append([])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        zipped = zip(range(len(item_lists)), item_lists)
        executor.map(render_item_list, zipped)
