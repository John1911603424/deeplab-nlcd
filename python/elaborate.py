#!/usr/bin/env python3

import argparse
from typing import *

import boto3  # type: ignore

import numpy as np  # type: ignore
import rasterio as rio  # type: ignore


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster', required=True, type=str)
    parser.add_argument('--geojson', required=True, nargs='+', type=str)
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()
    print(args)
