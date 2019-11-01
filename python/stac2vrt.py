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
from typing import *
from urllib.parse import urlparse

import boto3  # type: ignore
import pystac  # type: ignore
import requests


def cli_parser() -> argparse.ArgumentParser:
    """Return a command line argument parser

    Returns:
        argparse.ArgumentParser -- A parser object
    """
    parser = argparse.ArgumentParser()
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
        parsed = urlparse(uri, allow_fragments=False)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, prefix)
        return obj.get()['Body'].read().decode('utf-8')
    else:
        return pystac.STAC_IO.default_read_text_method(uri)


pystac.STAC_IO.read_text_method = requests_read_method


if __name__ == '__main__':
    args = cli_parser().parse_args()

    catalog = pystac.Catalog.from_file(args.input)
    for collection in catalog.get_children():
        if 'imagery' in str.lower(collection.description):
            imagery_collection = collection
        elif 'label' in str.lower(collection.description):
            label_collection = collection
