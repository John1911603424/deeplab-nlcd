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
    parser.add_argument('--imagery-only', action='store_true')
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


def decorate_item(item: Union[pystac.item.Item, pystac.label.LabelItem]) -> Union[pystac.item.Item, pystac.label.LabelItem]:
    """Decorate a label item or an imagery item with data for later use

    Arguments:
        item {Union[pystac.item.Item, pystac.label.LabelItem]} -- The item to decorate

    Returns:
        Union[pystac.item.Item, pystac.label.LabelItem] -- The decorated item
    """
    if isinstance(item, pystac.label.LabelItem):
        sources = list(item.get_sources())
        assert(len(sources) == 1)
        source = sources[0]
        source_assets = source.assets
        assert(len(source_assets) == 1)
    elif isinstance(item, pystac.item.Item):
        source = item
        source_assets = item.assets
        assert(len(source_assets) == 1)
    source_asset = next(iter(source_assets.values()))
    imagery_uri = source_asset.href
    if imagery_uri.startswith('./'):
        base = next(filter(lambda link: link.rel ==
                           'self', source.get_links())).target
        base = '/'.join(base.split('/')[0:-1]) + '/'
        item.imagery_uri = base + imagery_uri[2:len(imagery_uri)]
        item.imagery_uri = item.imagery_uri.replace('s3://', '/vsis3/')

    with rasterio.open(item.imagery_uri, 'r') as input_ds:
        item.imagery_transform = input_ds.transform
        item.imagery_crs = input_ds.crs.to_proj4()
        item.imagery_profile = copy.copy(input_ds.profile)

    return item


def render_label_item(item: pystac.label.LabelItem) -> Optional[Tuple[str, str]]:
    """Render a vector label item to a GeoTiff

    Arguments:
        item {pystac.label.LabelItem} -- The label item to render

    Returns:
        Optional[Tuple[str, str]] -- The uri of the associated imagery and the new label raster
    """
    assets = item.assets
    assert(len(assets) == 1)
    json_uri = next(iter(assets.values())).href
    if json_uri.startswith('./'):
        base = next(filter(lambda link: link.rel ==
                           'self', item.get_links())).target
        base = '/'.join(base.split('/')[0:-1]) + '/'
        json_uri = base + json_uri[2:len(json_uri)]
    json_str = pystac.STAC_IO.read_text_method(json_uri)
    label_vectors = json.loads(json_str)
    label_features = label_vectors.get('features')

    if len(label_features) > 0 or len(item.geometry) > 0:
        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj(args.geojson_crs), pyproj.Proj(item.imagery_crs))
        projection = transformer.transform

        profile = copy.copy(item.imagery_profile)
        profile.update(
            dtype=np.uint8,
            count=1,
            compress='lzw',
            nodata=0
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

            print('rendering {} features from item {} into file {}'.format(
                len(label_features), item, filename))
            rasterio.features.rasterize(
                shapes, out=rasterized_labels, transform=item.imagery_transform)

            output_ds.write(rasterized_labels, indexes=1)

        return (item.imagery_uri, filename)
    else:
        return None


def render_label_item_list(t: Tuple[int, List[pystac.label.LabelItem]]) -> None:
    """Render a list of label items

    Lists of imagery and label rasters are written which are suitable as input to gdalbuildvrt

    Arguments:
        t {Tuple[int, List[pystac.label.LabelItem]]} -- A list of label items along with the list number
    """
    (i, item_list) = t
    imagery_txt = '/tmp/imagery-{}.txt'.format(i)
    label_txt = '/tmp/labels-{}.txt'.format(i)
    with open(imagery_txt, 'w') as f, open(label_txt, 'w') as g:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            retvals = executor.map(render_label_item, item_list)
        for (imagery, label) in retvals:
            f.write(imagery + '\n')
            g.write(label + '\n')


def render_imagery_item_list(t: Tuple[int, List[pystac.item.Item]]) -> None:
    """Render a list of imagery items

    A list of imagery is written which is suitable as input to gdalbuildvrt

    Arguments:
        t {Tuple[int, List[pystac.item.Item]]} -- A list of label items along with a list number
    """
    (i, item_list) = t
    imagery_txt = '/tmp/imagery-{}.txt'.format(i)
    with open(imagery_txt, 'w') as f:
        for item in item_list:
            f.write(item.imagery_uri + '\n')


if __name__ == '__main__':
    args = cli_parser().parse_args()

    postfix = '/catalog.json'
    local_prefix = args.local_prefix
    base_prefix = args.input[0:-len(postfix) + 1]

    def requests_read_method_local(uri: str) -> str:
        if uri.startswith('./'):
            uri = base_prefix + uri[2:len(uri)]
        if local_prefix is not None and (uri.endswith('.json') or uri.endswith('.geojson')):
            uri = uri.replace(base_prefix, local_prefix)
        return requests_read_method(uri)

    pystac.STAC_IO.read_text_method = requests_read_method_local

    catalog = pystac.Catalog.from_file(args.input)
    interesting_collections = []
    for collection in catalog.get_children():
        if args.imagery_only:
            if 'imagery' in str.lower(collection.description):
                print('imagery collection {} ({}) found'.format(
                    collection, collection.description))
                interesting_collections.append(collection)
            else:
                print('collection {} ({}) rejected'.format(
                    collection, collection.description))
        else:
            if 'label' in str.lower(collection.description):
                print('label collection {} ({}) found'.format(
                    collection, collection.description))
                interesting_collections.append(collection)
            else:
                print('collection {} ({}) rejected'.format(
                    collection, collection.description))

    interesting_itemss = []
    for interesting_collection in interesting_collections:
        interesting_itemss.append(interesting_collection.get_items())

    liboverlaps = ctypes.CDLL('/tmp/liboverlaps.so')
    liboverlaps.query.argtypes = [
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    liboverlaps.query.restype = ctypes.c_double
    liboverlaps.insert.argtypes = [
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]

    liboverlaps.add_tree()
    item_lists: List[List[pystac.label.LabelItem]] = [[]]

    for interesting_items in interesting_itemss:
        for item in interesting_items:
            decorate_item(item)
            (xmin, ymin, xmax, ymax) = shapely.geometry.shape(item.geometry).bounds

            inserted = False
            for i in range(len(item_lists)):
                if len(item_lists[i]) == 0 or (item_lists[i][-1].imagery_crs == item.imagery_crs and liboverlaps.query(i, xmin, ymin, xmax, ymax) > 0.95):
                    liboverlaps.insert(i, xmin, ymin, xmax, ymax)
                    item_lists[i].append(item)
                    inserted = True
                    print('inserting item {} into list {}'.format(item, i))
                    break
            if not inserted:
                i = len(item_lists)
                liboverlaps.add_tree()
                liboverlaps.insert(i, xmin, ymin, xmax, ymax)
                item_lists.append([])
                item_lists[i].append(item)
                print('inserting item {} into new list {}'.format(item, i))

    for t in zip(range(len(item_lists)), item_lists):
        if args.imagery_only:
            render_imagery_item_list(t)
        else:
            render_label_item_list(t)
