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
import csv
import glob
import math
import re

import numpy as np
import scipy.ndimage

import rasterio as rio
import rasterio.transform


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagery-dir', required=True, type=str)
    parser.add_argument('--flood-label-dir', required=True, type=str)
    parser.add_argument('--perm-label-dir', required=False, type=str)
    parser.add_argument('--test-set-csv', required=False, type=str)
    parser.add_argument('--imagery-file', required=True, type=str)
    parser.add_argument('--labels-file', required=True, type=str)
    parser.add_argument('--regexp', required=False,
                        type=str, default='[Bb]olivia')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--chip-size', required=False, type=int, default=512)
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()
    regexp = re.compile(args.regexp)
    imagery = sorted(glob.glob('{dir}/*.*'.format(dir=args.imagery_dir)))
    water = sorted(glob.glob('{dir}/*.*'.format(dir=args.flood_label_dir)))
    if args.perm_label_dir:
        perm = sorted(glob.glob('{dir}/*.*'.format(dir=args.perm_label_dir)))
    else:
        perm = None

    if not len(imagery) == len(water):
        print('WARNING: different number of imagery and water chips')
        water_set = set(map(lambda t: t.split(
            '/')[-1].replace('_NoQC', ''), water))
        imagery = list(filter(lambda t: t.split(
            '/')[-1].replace('_S1', '') in water_set, imagery))
    assert(len(imagery) == len(water))
    if perm is not None:
        assert(len(water) == len(perm))

    if perm is not None:
        filenames = list(zip(imagery, water, perm))
    else:
        filenames = list(zip(imagery, water))

    test_set = set()
    if args.test_set_csv is not None:
        with open(args.test_set_csv) as csv_file:
            for row in csv.reader(csv_file):
                test_set |= set(map(lambda r: r.split('/')[-1], row))
    if (args.mode == 'test'):
        filenames = list(filter(
            lambda t: t[0].split('/')[-1] in test_set, filenames))
    elif (args.mode == 'train'):
        filenames = list(filter(
            lambda t: not regexp.search(t[0]), filenames))
        filenames = list(filter(
            lambda t: t[0].split('/')[-1] not in test_set, filenames))

    # Dimensions
    sqrt_chip_n = int(math.ceil(math.sqrt(len(filenames))))
    width = sqrt_chip_n * args.chip_size
    height = sqrt_chip_n * args.chip_size

    # Profiles
    with rio.open(imagery[0], 'r') as ds:
        images_profile = copy.deepcopy(ds.profile)
    images_profile.update(
        compress='deflate',
        predictor=2,
        driver='GTiff',
        width=width,
        height=height,
        tiled=True,
        bigtiff=True,
        transform=rasterio.transform.from_bounds(
            31.132830, 29.978150, 31.135448, 29.980260, width, height),
        crs='epsg:4326',
    )
    labels_profile = copy.deepcopy(images_profile)
    labels_profile.update(
        count=1,
        dtype=np.uint8,
        nodata=0,
    )

    # Arrays
    images = np.zeros((images_profile.get('count'), width,
                       height), dtype=images_profile.get('dtype'))
    labels = np.zeros((1, width, height), dtype=labels_profile.get('dtype'))

    # Data
    i = 0
    for t in filenames:
        if (i % 107) == 0:
            print('.')
        x = (i // sqrt_chip_n) * args.chip_size
        y = (i % sqrt_chip_n) * args.chip_size
        with rio.open(t[0], 'r') as ds:
            chip = ds.read()
            ratiox = float(args.chip_size) / chip.shape[1]
            ratioy = float(args.chip_size) / chip.shape[2]
            if ratiox != 1.0 or ratioy != 1.0:
                chip = scipy.ndimage.zoom(
                    chip, [1.0, ratiox, ratioy], order=0, prefilter=False)
            images[:, x:(x+args.chip_size), y:(y+args.chip_size)] = chip
        with rio.open(t[1], 'r') as ds:
            chip = ds.read() + 1
            ratiox = float(args.chip_size) / chip.shape[1]
            ratioy = float(args.chip_size) / chip.shape[2]
            if ratiox != 1.0 or ratioy != 1.0:
                chip = scipy.ndimage.zoom(
                    chip, [1.0, ratiox, ratioy], order=0, prefilter=False)
            labels[:, x:(x+args.chip_size), y:(y+args.chip_size)] = chip
        if len(t) == 3:
            with rio.open(t[2], 'r') as ds:
                chip = ds.read()
                ratiox = float(args.chip_size) / chip.shape[1]
                ratioy = float(args.chip_size) / chip.shape[2]
                if ratiox != 1.0 or ratioy != 1.0:
                    chip = scipy.ndimage.zoom(
                        chip, [1.0, ratiox, ratioy], order=0, prefilter=False)
                labels[:, x:(x+args.chip_size),
                       y:(y+args.chip_size)] += chip
        i += 1

    print('Writing ...')
    with rio.open(args.imagery_file, 'w', **images_profile) as ds:
        ds.write(images)
    with rio.open(args.labels_file, 'w', **labels_profile) as ds:
        ds.write(labels)
