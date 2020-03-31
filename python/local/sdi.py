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

import numpy as np
import rasterio as rio


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagery', required=True, nargs='+', type=str)
    parser.add_argument('--imagery-nd', required=False, type=int)
    parser.add_argument('--predictions', required=True, nargs='+', type=str)
    parser.add_argument('--truth', required=True, nargs='+', type=str)
    parser.add_argument('--truth-gte', required=True, type=int)
    parser.add_argument('--truth-nd', required=False, type=int)
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()

    class0_array = []
    class1_array = []

    for (image, prediction, truth) in zip(args.imagery, args.predictions, args.truth):

        print('{} {} {}'.format(image, prediction, truth))

        with rio.open(image, 'r') as ds:
            image_profile = copy.copy(ds.profile)
            image_data = ds.read()

        with rio.open(prediction, 'r') as ds:
            prediction_profile = copy.copy(ds.profile)
            prediction_data = ds.read()

        with rio.open(truth, 'r') as ds:
            truth_profile = copy.copy(ds.profile)
            truth_data = ds.read()

        # Width and height
        assert(image_profile.get('width') == prediction_profile.get('width'))
        assert(prediction_profile.get('width') == truth_profile.get('width'))
        assert(image_profile.get('height') == prediction_profile.get('height'))
        assert(prediction_profile.get('height') == truth_profile.get('height'))
        width = image_profile.get('width')
        height = image_profile.get('height')

        # Mask
        if args.imagery_nd is not None:
            image_mask = (image_data[0] != args.imagery_nd).astype(np.uint8)
        else:
            image_mask = np.ones((height, width), dtype=np.uint8)
        if args.truth_nd is not None:
            truth_mask = (truth_data[0] != args.truth_nd).astype(np.uint8)
        else:
            truth_mask = np.ones((height, width), dtype=np.uint8)
        valid_mask = image_mask * truth_mask
        mask = truth_data >= args.truth_gte

        class0 = np.extract((mask != 1) * valid_mask, prediction_data)
        class1 = np.extract(mask * valid_mask, prediction_data)
        class0_array.append(class0)
        class1_array.append(class1)

    class0_array = np.concatenate(class0_array, axis=None)
    class1_array = np.concatenate(class1_array, axis=None)

    mu0 = np.mean(class0_array)
    sigma0 = np.std(class0_array)
    mu1 = np.mean(class1_array)
    sigma1 = np.std(class1_array)

    print('mu0 = {}, sigma0 = {}'.format(mu0, sigma0))
    print('mu1 = {}, sigma1 = {}'.format(mu1, sigma1))
    print('sdi = {}'.format(np.abs(mu0 - mu1)/(sigma0 + sigma1)))
