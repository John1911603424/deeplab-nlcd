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
import codecs
import copy
import ctypes
import os
import sys
import threading
from typing import *
from urllib.parse import urlparse

import boto3  # type: ignore
import numpy as np  # type: ignore
import rasterio as rio  # type: ignore
import requests

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

# Warm-up
if True:
    def get_warmup_batch(libchips: ctypes.CDLL,
                         args: argparse.Namespace) -> torch.Tensor:
        """Get a warm-up batch

        Arguments:
            libchips {ctypes.CDLL} -- A shared library handled used for reading data
            args {argparse.Namespace} -- The arguments

        Returns:
            torch.Tensor -- A batch in the form of a PyTorch tensor
        """
        shape = (len(args.bands),
                 args.warmup_window_size,
                 args.warmup_window_size)
        temp1 = np.zeros(shape, dtype=np.float32)
        temp1_ptr = temp1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        rasters = []
        for _ in range(args.warmup_batch_size):
            libchips.get_next(temp1_ptr, ctypes.c_void_p(0))
            rasters.append(temp1.copy())

        raster_batch = []
        for raster in rasters:

            if not args.architecture.endswith('-binary.py'):
                for i in range(len(raster)):
                    raster[i] = (raster[i] - args.mus[i]) / args.sigmas[i]

            # NODATA from rasters
            image_nds = np.isnan(raster).sum(axis=0)
            if args.image_nd is not None:
                image_nds += (raster == args.image_nd).sum(axis=0)

            # Remove NaNs from rasters
            nodata = (image_nds > 0)
            for i in range(len(raster)):
                raster[i][nodata == True] = 0.0

            raster_batch.append(raster)

        raster_batch_tensor = torch.from_numpy(np.stack(raster_batch, axis=0))

        return raster_batch_tensor

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
            if not args.architecutre.endswith('-binary.py'):
                for i in range(len(image)):
                    image[i] = (image[i] - args.mus[i]) / args.sigmas[i]
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
                            required=True, type=int,
                            help='The number of prediction classes')
        parser.add_argument('--final-prediction-img',
                            help='The location where the final prediction image should be stored')
        parser.add_argument('--image-nd',
                            default=None, type=float,
                            help='image value to ignore - must be on the first band')
        parser.add_argument('--inference-img',
                            required=True,
                            help='The location of the image on which to predict')
        parser.add_argument('--input-stride',
                            default=2, type=int,
                            help='consult this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md')
        parser.add_argument('--libchips',
                            required=True,
                            help='The location of libchips.so')
        parser.add_argument('--model',
                            required=True,
                            help='The model to use for preditions')
        parser.add_argument('--no-warmup', action='store_true')
        parser.add_argument('--radius', default=10000)
        parser.add_argument('--raw-prediction-img',
                            help='The location where the raw prediction image should be stored')
        parser.add_argument('--statistics-img-uri',
                            help='The image from which to obtain statistics for normalization')
        parser.add_argument('--warmup-batch-size', default=16, type=int)
        parser.add_argument('--warmup-window-size', default=32, type=int)
        parser.add_argument('--window-size', default=256, type=int)
        return parser

# Architectures
if True:
    def make_model(band_count, input_stride=1, class_count=1, divisor=1, pretrained=False):
        raise Exception()

    def load_architectures(uri: str) -> None:
        arch_str = read_text(uri)
        arch_code = compile(arch_str, '/dev/null', 'exec')
        exec(arch_code, globals())


if __name__ == '__main__':

    parser = inference_cli_parser()
    args = inference_cli_parser().parse_args()

    args.mus = np.ndarray(len(args.bands), dtype=np.float64)
    args.sigmas = np.ndarray(len(args.bands), dtype=np.float64)
    mus_ptr = args.mus.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sigmas_ptr = args.sigmas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    args.band_count = len(args.bands)

    load_architectures(args.architecture)

    # ---------------------------------
    print('DATA')

    mul = '/tmp/mul.tif'
    if not os.path.exists(mul):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.inference_img)
        print('Inference image bucket and prefix: {}, {}'.format(bucket, prefix))
        s3.download_file(bucket, prefix, mul)
        del s3

    # ---------------------------------
    print('MODEL')

    if not os.path.exists('/tmp/weights.pth'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.model)
        print('Model bucket and prefix: {}, {}'.format(bucket, prefix))
        s3.download_file(bucket, prefix, '/tmp/weights.pth')
        del s3

    device = torch.device(args.backend)

    deeplab = make_model(
        args.band_count,
        input_stride=args.input_stride,
        class_count=args.classes,
        pretrained=False,
    ).to(device)
    deeplab.load_state_dict(torch.load(
        '/tmp/weights.pth', map_location=device))

    # ---------------------------------
    print('NATIVE CODE')

    if not os.path.exists('/tmp/libchips.so'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.libchips)
        print('Shared library bucket and prefix: {}, {}'.format(bucket, prefix))
        s3.download_file(bucket, prefix, '/tmp/libchips.so')
        del s3

    libchips = ctypes.CDLL('/tmp/libchips.so')
    libchips.init()

    # ---------------------------------
    print('STATISTICS')
    if args.statistics_img_uri is None:
        args.statistics_img_uri = '/tmp/mul.tif'
    libchips.get_statistics(
        args.statistics_img_uri.encode('utf-8'),
        len(args.bands),
        np.array(args.bands, dtype=np.int32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_int32)),
        mus_ptr,
        sigmas_ptr
    )
    print('MEANS={}'.format(args.mus))
    print('SIGMAS={}'.format(args.sigmas))

    # ---------------------------------
    print('WARMUP')
    # https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/2
    # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/37
    if not args.no_warmup:
        def momentum_fn(m):
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = 0.9
        deeplab.apply(momentum_fn)
        deeplab.train()
        libchips.start(
            16,  # Number of threads
            256,  # Number of slots
            b'/tmp/mul.tif',  # Image data
            None,  # Label data
            6,  # Make all rasters float32
            5,  # Make all labels int32
            None,  # means
            None,  # standard deviations
            args.radius,  # typical radius of a component
            1,  # Training mode
            args.warmup_window_size,
            len(args.bands),
            np.array(args.bands, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        with torch.no_grad():
            for i in range(107):
                batch = get_warmup_batch(libchips, copy.copy(args))
                out = deeplab(batch.to(device))
                del out
        libchips.stop()

    # ---------------------------------
    print('INFERENCE')

    libchips.start(
        1,  # Number of threads
        0,  # Number of slots
        b'/tmp/mul.tif',  # Image data
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

    with rio.open('/tmp/mul.tif') as ds:
        profile_final = copy.copy(ds.profile)
        profile_final.update(
            dtype=rio.uint8,
            count=1,
            compress='lzw'
        )
        profile_raw = copy.copy(ds.profile)
        profile_raw.update(
            dtype=rio.float32,
            count=args.classes,
            compress='lzw'
        )

    deeplab.eval()
    with torch.no_grad():
        with rio.open('/tmp/pred-final.tif', 'w', **profile_final) as ds_final, rio.open('/tmp/pred-raw.tif', 'w', **profile_raw) as ds_raw:
            width = libchips.get_width(0)
            height = libchips.get_height(0)
            print('{} {}'.format(width, height))
            for x_offset in range(0, width, args.window_size):
                if x_offset + args.window_size > width:
                    x_offset = width - args.window_size - 1
                for y_offset in range(0, height, args.window_size):
                    if y_offset + args.window_size > height:
                        y_offset = height - args.window_size - 1
                    window = rio.windows.Window(
                        x_offset, y_offset, args.window_size, args.window_size)
                    tensor = get_inference_window(
                        libchips, x_offset, y_offset, copy.copy(args))
                    if tensor is not None:
                        tensor = tensor.to(device)
                        out = deeplab(tensor)
                        if isinstance(out, dict):
                            out = out['out']
                        out = out.data.cpu().numpy()
                        for i in range(0, args.classes):
                            ds_raw.write(out[0, i], window=window, indexes=i+1)
                        if args.classes > 1:
                            out = np.apply_along_axis(np.argmax, 1, out)
                            out = np.array(out, dtype=np.uint8)
                            out = out * (0xff // (args.classes-1))
                            ds_final.write(out[0], window=window, indexes=1)
                print('{}% complete'.format(
                    (int)(100.0 * x_offset / width)))

    if args.final_prediction_img is not None:
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.final_prediction_img)
        prefix = prefix.replace('*', args.inference_img.split('/')[-1])
        s3.upload_file('/tmp/pred-final.tif', bucket, prefix)
        del s3
    if args.raw_prediction_img is not None:
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.raw_prediction_img)
        prefix = prefix.replace('*', args.inference_img.split('/')[-1])
        s3.upload_file('/tmp/pred-raw.tif', bucket, prefix)
        del s3

    libchips.stop()
    libchips.deinit()
