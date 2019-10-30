#!/usr/bin/env python3

import argparse
import copy
import functools
from typing import *

import boto3  # type: ignore

import numpy as np  # type: ignore
import pyproj
import rasterio as rio  # type: ignore


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster', required=True, type=str)
    parser.add_argument('--raster-band', default=1, type=int)
    parser.add_argument('--geojson', required=True, nargs='+', type=str)
    parser.add_argument('--geojson-crs', default='+init=epsg:4326', type=str)
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
        print(raster_crs, type(raster_crs))

    projection = functools.partial(
        pyproj.transform,
        pyproj.Proj(args.geojson_crs),
        pyproj.Proj(raster_crs)
    )

    rasterized_data = np.zeros(raster_data.shape, dtype=np.int32)
