#!/usr/bin/env python3

# The MIT License (MIT)
# =====================
#
# Copyright © 2019-2020 Azavea
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
import ast
import codecs
import copy
import ctypes
import json
import os
import sys
import threading
from datetime import datetime
from typing import *
from urllib.parse import urlparse

import numpy as np  # type: ignore
import requests

import boto3  # type: ignore
import rasterio as rio  # type: ignore
import torch
import torchvision  # type: ignore

# S3
if True:
    def read_text(uri: str) -> str:
        """A reader function that supports http, s3, and local files

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
            with codecs.open(uri, encoding='utf-8', mode='r') as f:
                return f.read()

    def parse_s3_url(url: str) -> Tuple[str, str]:
        """Given an S3 URI, return the bucket and prefix

        Arguments:
            url {str} -- The S3 URI to parse

        Returns:
            Tuple[str, str] -- The bucket and prefix as a tuple
        """
        parsed = urlparse(url, allow_fragments=False)
        return (parsed.netloc, parsed.path.lstrip('/'))

# Inference
if True:
    def get_inference_window(libchips: ctypes.CDLL,
                             x_offset: int,
                             y_offset: int,
                             args: argparse.Namespace) -> Union[None, torch.Tensor]:
        """Read the data specified in the given plan

        Arguments:
            libchips {ctypes.CDLL} -- A shared library handle used for reading data
            x_offset {int} -- The x-offset of the desired window
            y_offset {int} -- The y-offset of the desired window
            args {argparse.Namespace} -- Arguments

        Returns:
            Union[None, torch.Tensor] -- The imagery data as a PyTorch tensor
        """
        shape = (len(args.bands), args.window_size, args.window_size)
        image = np.zeros(shape, dtype=np.float32)
        image_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if (libchips.get_inference_chip(image_ptr, x_offset, y_offset, 33) == 1):
            image_nds = np.isnan(image).sum(axis=0)
            if args.image_nd is not None:
                image_nds += (image == args.image_nd).sum(axis=0)
            for i in range(len(image)):
                image[i][image_nds > 0] = 0.0
            return torch.from_numpy(np.stack([image], axis=0))
        else:
            return None

# Arguments
if True:
    class StoreDictKeyPair(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            my_dict = {}
            for kv in values.split(','):
                k, v = kv.split(':')
                my_dict[int(k)] = int(v)
            setattr(namespace, self.dest, my_dict)

    def inference_cli_parser() -> argparse.ArgumentParser:
        """Generate a parser for command line arguments

        See: https://stackoverflow.com/questions/29986185/python-argparse-dict-arg

        Returns:
            argparse.ArgumentParser -- The parser
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--architecture',
                            help='The desired model architecture', required=True)
        parser.add_argument('--backend',
                            help="Don't use this flag unless you know what you're doing: CPU is far slower than CUDA.",
                            choices=['cpu', 'cuda'], default='cuda')
        parser.add_argument('--bands',
                            required=True, nargs='+', type=int,
                            help='list of bands to train on (1 indexed)')
        parser.add_argument('--classes',
                            required=False, type=int, default=1,
                            help='The number of prediction classes')
        parser.add_argument('--force-download',
                            type=ast.literal_eval, default=False)
        parser.add_argument('--final-prediction-img',
                            help='The location where the final prediction image should be stored')
        parser.add_argument('--image-nd',
                            default=None, type=float,
                            help='image value to ignore - must be on the first band')
        parser.add_argument('--inference-img',
                            required=True, nargs='+', type=str,
                            help='The location of the image on which to predict')
        parser.add_argument('--input-stride',
                            default=2, type=int,
                            help='consult this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md')
        parser.add_argument('--libchips',
                            required=True,
                            help='The location of libchips.so')
        parser.add_argument('--weights',
                            required=True,
                            help='The weights for the model used for preditions')
        parser.add_argument('--radius', default=10000)
        parser.add_argument('--raw-prediction-img',
                            help='The location where the raw prediction image should be stored')
        parser.add_argument('--regression-prediction-img',
                            help='The location where the regression prediction image should be stored')
        parser.add_argument('--resolution-divisor', default=1, type=int)
        parser.add_argument('--window-size', default=256, type=int)
        parser.add_argument('--threshold', required=False,
                            default=0.0, type=float)
        parser.add_argument(
            '--report', help='The location where the report will be stored')
        parser.add_argument(
            '--report-band', help='The band from which the report should be derived')
        return parser

# Architectures
if True:
    def make_model(band_count, input_stride=1, class_count=1, divisor=1, pretrained=False):
        raise Exception()

    def load_architectures(uri: str) -> None:
        arch_str = read_text(uri)
        arch_code = compile(arch_str, uri, 'exec')
        exec(arch_code, globals())


tmp_weights = '/tmp/weights.pth'
tmp_mul = '/tmp/mul.tif'
tmp_libchips = '/tmp/libchips.so'
tmp_pred_final = '/tmp/pred-final.tif'
tmp_pred_raw = '/tmp/pred-raw.tif'
tmp_pred_reg = '/tmp/pred-reg.tif'


if __name__ == '__main__':

    parser = inference_cli_parser()
    args = inference_cli_parser().parse_args()

    args.band_count = len(args.bands)

    load_architectures(args.architecture)

    # ---------------------------------
    print('MODEL')

    if args.weights.startswith('s3://'):
        if not os.path.exists(tmp_weights) or args.force_download:
            s3 = boto3.client('s3')
            bucket, prefix = parse_s3_url(args.weights)
            print('Model bucket and prefix: {}, {}'.format(bucket, prefix))
            s3.download_file(bucket, prefix, tmp_weights)
            del s3
        args.weights = tmp_weights

    # ---------------------------------
    print('DATA')

    # Look for newline-delimited lists of files
    if len(args.inference_img) == 1 and args.inference_img[0].endswith('.list'):
        text = read_text(args.inference_img[0])
        args.inference_img = list(
            filter(lambda line: len(line) > 0, text.split('\n')))

    for inference_img in args.inference_img:
        if inference_img.startswith('s3://'):
            if not os.path.exists(tmp_mul) or len(args.inference_img) > 1 or args.force_download:
                s3 = boto3.client('s3')
                bucket, prefix = parse_s3_url(inference_img)
                print('Inference image bucket and prefix: {}, {}'.format(
                    bucket, prefix))
                s3.download_file(bucket, prefix, tmp_mul)
                del s3
            inference_img = tmp_mul

        # ---------------------------------

        device = torch.device(args.backend)

        model = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=args.classes,
            divisor=args.resolution_divisor,
            pretrained=False,
        ).to(device)
        if not hasattr(model, 'no_weights'):
            model.load_state_dict(torch.load(
                args.weights, map_location=device))

        # ---------------------------------
        print('NATIVE CODE')

        if args.libchips.startswith('s3://'):
            if not os.path.exists(tmp_libchips):
                s3 = boto3.client('s3')
                bucket, prefix = parse_s3_url(args.libchips)
                print('Shared library bucket and prefix: {}, {}'.format(
                    bucket, prefix))
                s3.download_file(bucket, prefix, tmp_libchips)
                del s3
            args.libchips = tmp_libchips

        libchips = ctypes.CDLL(args.libchips)
        libchips.init()

        # ---------------------------------
        print('INFERENCE')

        libchips.start(
            1,  # Number of threads
            0,  # Number of slots
            inference_img.encode(),  # Image data
            None,  # Label data
            6,  # Make all rasters float32
            5,  # Make all labels int32
            None,  # means
            None,  # standard deviations
            args.radius,  # typical radius of component
            3,  # Inference mode
            args.window_size,
            len(args.bands),
            np.array(args.bands, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))

        with rio.open(inference_img) as ds:
            profile_final = copy.deepcopy(ds.profile)
            profile_final.update(
                dtype=rio.uint8,
                count=1,
                compress='lzw',
                nodata=None
            )
            profile_raw = copy.deepcopy(ds.profile)
            profile_raw.update(
                dtype=rio.float32,
                count=args.classes,
                compress='lzw',
                nodata=None
            )
            profile_reg = copy.deepcopy(ds.profile)
            profile_reg.update(
                dtype=rio.float32,
                count=1,
                compress='lzw',
                nodata=None
            )
        model.eval()
        start_time = datetime.now()
        with torch.no_grad():
            with rio.open(tmp_pred_final, 'w', **profile_final) as ds_final, \
                    rio.open(tmp_pred_raw, 'w', **profile_raw) as ds_raw, \
                    rio.open(tmp_pred_reg, 'w', **profile_reg) as ds_reg:
                width = libchips.get_width(0)
                height = libchips.get_height(0)
                print('x={} y={} n={}'.format(width, height, args.window_size))
                for x_offset in range(0, width, args.window_size):
                    if x_offset + args.window_size > width:
                        x_offset = width - args.window_size - 1
                    for y_offset in range(0, height, args.window_size):
                        if y_offset + args.window_size > height:
                            y_offset = height - args.window_size - 1
                        window = rio.windows.Window(
                            x_offset, y_offset, args.window_size, args.window_size)
                        tensor = get_inference_window(
                            libchips, x_offset, y_offset, copy.deepcopy(args))
                        if tensor is not None:
                            tensor = tensor.to(device)
                            out = model(tensor)
                            if isinstance(out, dict):
                                if 'reg' in out:
                                    reg_window = np.ones(
                                        (window.width, window.height), dtype=np.float32)
                                    reg: Any = out.get('reg')
                                    reg_window = reg_window * reg.item()
                                    ds_reg.write(
                                        reg_window, window=window, indexes=1)
                                out = out.get('out', out.get(
                                    'seg', out.get('2seg', None)))
                            if out is not None:
                                out_torch = out
                                out = out.cpu().numpy()
                                for i in range(0, args.classes):
                                    ds_raw.write(
                                        out[0, i], window=window, indexes=i+1)
                                if args.classes > 1:
                                    out = torch.max(out_torch, 1)[1].cpu().numpy().astype(np.uint8)
                                    out = out * (0xff // (args.classes-1))
                                    ds_final.write(
                                        out[0], window=window, indexes=1)
                                else:
                                    out = np.array(
                                        out > args.threshold, dtype=np.uint8)
                                    out = out * 0xff
                                    ds_final.write(
                                        out[0][0], window=window, indexes=1)
                    print('{:02.2f}% complete'.format(
                        (100.0 * x_offset / width)))
        finish_time = datetime.now()
        print(finish_time - start_time)

        if args.final_prediction_img is not None:
            if args.final_prediction_img.startswith('s3://'):
                s3 = boto3.client('s3')
                bucket, prefix = parse_s3_url(args.final_prediction_img)
                prefix = prefix.replace('*', inference_img.split('/')[-1])
                s3.upload_file(tmp_pred_final, bucket, prefix)
                del s3
            else:
                img = args.final_prediction_img.replace(
                    '*', inference_img.split('/')[-1])
                command = 'cp -f {} {}'.format(tmp_pred_final, img)
                os.system(command)

        if args.raw_prediction_img is not None:
            if args.raw_prediction_img.startswith('s3://'):
                s3 = boto3.client('s3')
                bucket, prefix = parse_s3_url(args.raw_prediction_img)
                prefix = prefix.replace('*', inference_img.split('/')[-1])
                s3.upload_file(tmp_pred_raw, bucket, prefix)
                del s3
            else:
                img = args.raw_prediction_img.replace(
                    '*', inference_img.split('/')[-1])
                command = 'cp -f {} {}'.format(tmp_pred_raw, img)
                os.system(command)

        if args.regression_prediction_img is not None:
            if args.regression_prediction_img.startswith('s3://'):
                s3 = boto3.client('s3')
                bucket, prefix = parse_s3_url(args.regression_prediction_img)
                prefix = prefix.replace('*', inference_img.split('/')[-1])
                s3.upload_file(tmp_pred_reg, bucket, prefix)
                del s3
            else:
                img = args.regression_prediction_img.replace(
                    '*', inference_img.split('/')[-1])
                command = 'cp -f {} {}'.format(tmp_pred_reg, img)
                os.system(command)

    libchips.stop()
    libchips.deinit()

    if args.report and args.report_band:
        command = 'gdalinfo -json {}'.format(inference_img)
        info = json.loads(os.popen(command).read())
        [x, y] = info.get('size')
        command = 'gdal_translate -b {} -co TILED=YES -co SPARSE_OK=YES {} /tmp/out0.tif'.format(
            args.report_band, inference_img)
        os.system(command)
        command = 'gdalwarp -ts {} {} -r max -co TILED=YES -co SPARSE_OK=YES /tmp/out0.tif /tmp/out1.tif'.format(
            x//4, y//4)
        os.system(command)
        command = 'gdal_polygonize.py /tmp/out1.tif -f GeoJSON /tmp/out.geojson'
        os.system(command)
        command = 'gzip -9 /tmp/out.geojson'
        os.system(command)
        command = 'aws s3 cp /tmp/out.geojson.gz {}'.format(args.report)
        os.system(command)
        command = 'rm -f /tmp/out0.tif /tmp/out1.tif /tmp/out.geojson.gz'
        os.system(command)
