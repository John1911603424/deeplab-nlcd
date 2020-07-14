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
import glob
import math

import numpy as np

import rasterio as rio
import rasterio.transform


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True, type=str)
    parser.add_argument('--label-dir', required=True, type=str)
    return parser


# Given a directory containing imagery chips and a directory containing
# label chips, produce one large imagery tiff and one large label
# tiff.
if __name__ == '__main__':
    args = cli_parser().parse_args()
    image_chips = sorted(glob.glob('{dir}/*.png'.format(dir=args.image_dir)))
    chip_n = len(image_chips)

    with rio.open(image_chips[0], 'r') as ds:
        chip_width = ds.profile.get('width')
        chip_height = ds.profile.get('height')

    print('{n} chips of size {w}x{h}'.format(
        n=chip_n, w=chip_width, h=chip_height))

    sqrt_chip_n = math.ceil(math.sqrt(chip_n))
    width = sqrt_chip_n * chip_width
    height = sqrt_chip_n * chip_height

    print('{w}x{h} output size'.format(w=width, h=height))

    images = np.ones((3, width, height), dtype=np.uint8) * 0x7f
    image_profile = {
        'dtype': np.uint8,
        'count': 3,
        'compress': 'deflate',
        'predictor': 2,
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'nodata': None,
        'tiled': True,
        'transform': rasterio.transform.from_bounds(31.132830, 29.978150, 31.135448, 29.980260, width, height),
        'crs': 'epsg:4326',
        'bigtiff': True,
    }

    labels = np.ones((1, width, height), dtype=np.uint8) * 0x7f
    labels_profile = {
        'dtype': np.uint8,
        'count': 1,
        'compress': 'deflate',
        'predictor': 2,
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'nodata': 0x7f,
        'tiled': True,
        'transform': rasterio.transform.from_bounds(31.132830, 29.978150, 31.135448, 29.980260, width, height),
        'crs': 'epsg:4326',
        'bigtiff': True,
    }

    i = 0
    for chip in image_chips:
        image_filename = chip
        label_filename = chip.split('/')[-1]
        label_filename = '{dir}/{filename}'.format(
            dir=args.label_dir, filename=label_filename)
        x = (i % sqrt_chip_n) * chip_width
        y = (i // sqrt_chip_n) * chip_height

        with rio.open(image_filename, 'r') as ds:
            image_chip = ds.read()
            nodata = (image_chip[0] == 0) * \
                (image_chip[1] == 0) * (image_chip[2] == 0)
            images[:, x:(x+chip_width), y:(y+chip_height)] = image_chip
        with rio.open(label_filename, 'r') as ds:
            label_chip = ds.read()
            label_chip[0][nodata] = 0x7f
            labels[:, x:(x+chip_width), y:(y+chip_height)] = label_chip

        if (i % 1031) == 33:
            print('{:03.4f}%'.format(100 * float(i) / len(image_chips)))
        i += 1

    print('Writing ...')
    with rio.open('./images.tif', 'w', **image_profile) as ds:
        ds.write(images)

    with rio.open('./labels.tif', 'w', **labels_profile) as ds:
        ds.write(labels)
