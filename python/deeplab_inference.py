#!/usr/bin/env python3

import argparse
import copy
import ctypes
import hashlib
import os
import re
import sys
import threading
import time
from typing import *
from urllib.parse import urlparse

import boto3  # type: ignore

import numpy as np  # type: ignore
import rasterio as rio  # type: ignore
import torch
import torchvision  # type: ignore

WATCHDOG_MUTEX: threading.Lock = threading.Lock()
WATCHDOG_TIME: float = time.time()
EVALUATIONS_BATCHES_DONE = 0


# S3
if True:
    def parse_s3_url(url: str) -> Tuple[str, str]:
        """Given an S3 URI, return the bucket and prefix

        Arguments:
            url {str} -- The S3 URI to parse

        Returns:
            Tuple[str, str] -- The bucket and prefix as a tuple
        """
        parsed = urlparse(url, allow_fragments=False)
        return (parsed.netloc, parsed.path.lstrip('/'))

    def get_matching_s3_keys(bucket: str,
                             prefix: str = '',
                             suffix: str = '') -> Generator[str, None, None]:
        """Generate all of the keys in a bucket with the given prefix and suffix

        See https://alexwlchan.net/2017/07/listing-s3-keys/

        Arguments:
            bucket {str} -- The S3 bucket

        Keyword Arguments:
            prefix {str} -- The prefix to filter by (default: {''})
            suffix {str} -- The suffix to filter by (default: {''})

        Returns:
            Generator[str, None, None] -- The list of keys
        """
        s3 = boto3.client('s3')
        kwargs = {'Bucket': bucket}

        # If the prefix is a single string (not a tuple of strings), we can
        # do the filtering directly in the S3 API.
        if isinstance(prefix, str):
            kwargs['Prefix'] = prefix

        while True:

            # The S3 API response is a large blob of metadata.
            # 'Contents' contains information about the listed objects.
            resp = s3.list_objects_v2(**kwargs)
            for obj in resp['Contents']:
                key = obj['Key']
                if key.startswith(prefix) and key.endswith(suffix):
                    yield key

            # The S3 API is paginated, returning up to 1000 keys at a time.
            # Pass the continuation token into the next response, until we
            # reach the final page (when this field is missing).
            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break

# Inference
if True:
    def get_inference_window(libchips: ctypes.CDLL,
                             x_offset: int,
                             y_offset: int,
                             band_count: int,
                             window_size: int,
                             image_nd: Union[None, Union[int, float]]) -> Union[None, torch.Tensor]:
        """Read the data specified in the given plan

        Arguments:
            libchips {ctypes.CDLL} -- A shared library handle used for reading data
            x_offset {int} -- The x-offset of the desired window
            y_offset {int} -- The y-offset of the desired window
            band_count {int} -- The number of bands in the training set
            window_size {int} -- The window size
            image_nd {Union[None, Union[int, float]]} -- The image nodata

        Returns:
            Union[None, torch.Tensor] -- The imagery data as a PyTorch tensor
        """
        shape = (band_count, window_size, window_size)
        image = np.zeros(shape, dtype=np.float32)
        image_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if (libchips.get_inference_chip(image_ptr, x_offset, y_offset, 33) == 1):
            image_nds = np.isnan(image)
            if image_nd is not None:
                image_nds += (image == image_nd)
            image[image_nds > 0] = 0.0
            return torch.from_numpy(np.stack([image], axis=0))
        else:
            return None

# Arguments
if True:
    def hash_string(string: str) -> str:
        """Return a SHA-256 hash of the given string

        See: https://gist.github.com/nmalkin/e287f71788c57fd71bd0a7eec9345add

        Arguments:
            string {str} -- The string to hash

        Returns:
            str -- The hashed string
        """
        return hashlib.sha256(string.encode('utf-8')).hexdigest()

    class StoreDictKeyPair(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            my_dict = {}
            for kv in values.split(','):
                k, v = kv.split(':')
                my_dict[int(k)] = int(v)
            setattr(namespace, self.dest, my_dict)

    def training_cli_parser() -> argparse.ArgumentParser:
        """Generate a parser for command line arguments

        See: https://stackoverflow.com/questions/29986185/python-argparse-dict-arg

        Returns:
            argparse.ArgumentParser -- The parser
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--architecture', required=True, help='The desired model architecture',
                            choices=['resnet18', 'resnet34', 'resnet101', 'stock'])
        parser.add_argument('--backend', help="Don't use this flag unless you know what you're doing: CPU is far slower than CUDA.",
                            choices=['cpu', 'cuda'], default='cuda')
        parser.add_argument('--bands', required=True,
                            help='list of bands to train on (1 indexed)', nargs='+', type=int)
        parser.add_argument(
            '--image-nd', help='image value to ignore - must be on the first band', default=None, type=float)
        parser.add_argument('--classes', required=True,
                            help='The number of prediction classes', type=int)
        parser.add_argument('--final-predictions', action='store_true')
        parser.add_argument('--inference-img', required=True,
                            help='The location of the image on which to predict')
        parser.add_argument(
            '--input-stride', help='consult this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md', default=2, type=int)
        parser.add_argument('--libchips', required=True,
                            help='The location of libchips.so')
        parser.add_argument('--model', required=True,
                            help='The model to use for preditions')
        parser.add_argument('--prediction-img', required=True,
                            help='The location where the prediction image should be stored')
        parser.add_argument('--window-size', default=256, type=int)
        return parser

# Architectures
if True:
    class DeepLabResnet18(torch.nn.Module):
        def __init__(self, band_count, input_stride, class_count):
            super(DeepLabResnet18, self).__init__()
            resnet18 = torchvision.models.resnet.resnet18(pretrained=False)
            self.backbone = torchvision.models._utils.IntermediateLayerGetter(
                resnet18, return_layers={'layer4': 'out', 'layer3': 'aux'})
            inplanes = 512
            self.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(
                inplanes, class_count)
            inplanes = 256
            self.aux_classifier = torchvision.models.segmentation.fcn.FCNHead(
                inplanes, class_count)
            self.backbone.conv1 = torch.nn.Conv2d(
                band_count, 64, kernel_size=7, stride=input_stride, padding=3, bias=False)

            if input_stride == 1:
                self.factor = 16
            else:
                self.factor = 32

        def forward(self, x):
            [w, h] = x.shape[-2:]

            features = self.backbone(torch.nn.functional.interpolate(
                x, size=[w*self.factor, h*self.factor], mode='bilinear', align_corners=False))

            result = {}

            x = features['out']
            x = self.classifier(x)
            x = torch.nn.functional.interpolate(
                x, size=[w, h], mode='bilinear', align_corners=False)
            result['out'] = x

            x = features['aux']
            x = self.aux_classifier(x)
            x = torch.nn.functional.interpolate(
                x, size=[w, h], mode='bilinear', align_corners=False)
            result['aux'] = x

            return result

    def make_model_resnet18(band_count, input_stride=1, class_count=2):
        deeplab = DeepLabResnet18(band_count, input_stride, class_count)
        return deeplab

    class DeepLabResnet34(torch.nn.Module):
        def __init__(self, band_count, input_stride, class_count):
            super(DeepLabResnet34, self).__init__()
            resnet34 = torchvision.models.resnet.resnet34(pretrained=False)
            self.backbone = torchvision.models._utils.IntermediateLayerGetter(
                resnet34, return_layers={'layer4': 'out', 'layer3': 'aux'})
            inplanes = 512
            self.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(
                inplanes, class_count)
            inplanes = 256
            self.aux_classifier = torchvision.models.segmentation.fcn.FCNHead(
                inplanes, class_count)
            self.backbone.conv1 = torch.nn.Conv2d(
                band_count, 64, kernel_size=7, stride=input_stride, padding=3, bias=False)

            if input_stride == 1:
                self.factor = 16
            else:
                self.factor = 32

        def forward(self, x):
            [w, h] = x.shape[-2:]

            features = self.backbone(torch.nn.functional.interpolate(
                x, size=[w*self.factor, h*self.factor], mode='bilinear', align_corners=False))

            result = {}

            x = features['out']
            x = self.classifier(x)
            x = torch.nn.functional.interpolate(
                x, size=[w, h], mode='bilinear', align_corners=False)
            result['out'] = x

            x = features['aux']
            x = self.aux_classifier(x)
            x = torch.nn.functional.interpolate(
                x, size=[w, h], mode='bilinear', align_corners=False)
            result['aux'] = x

            return result

    def make_model_resnet34(band_count, input_stride=1, class_count=2):
        deeplab = DeepLabResnet34(band_count, input_stride, class_count)
        return deeplab

    class DeepLabResnet101(torch.nn.Module):
        def __init__(self, band_count, input_stride, class_count):
            super(DeepLabResnet101, self).__init__()
            resnet18 = torchvision.models.resnet.resnet101(
                pretrained=False, replace_stride_with_dilation=[False, True, True])
            self.backbone = torchvision.models._utils.IntermediateLayerGetter(
                resnet18, return_layers={'layer4': 'out', 'layer3': 'aux'})
            inplanes = 2048
            self.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(
                inplanes, class_count)
            inplanes = 1024
            self.aux_classifier = torchvision.models.segmentation.fcn.FCNHead(
                inplanes, class_count)
            self.backbone.conv1 = torch.nn.Conv2d(
                band_count, 64, kernel_size=7, stride=input_stride, padding=3, bias=False)

            if input_stride == 1:
                self.factor = 4
            else:
                self.factor = 8

        def forward(self, x):
            [w, h] = x.shape[-2:]

            features = self.backbone(torch.nn.functional.interpolate(
                x, size=[w*self.factor, h*self.factor], mode='bilinear', align_corners=False))

            result = {}

            x = features['out']
            x = self.classifier(x)
            x = torch.nn.functional.interpolate(
                x, size=[w, h], mode='bilinear', align_corners=False)
            result['out'] = x

            x = features['aux']
            x = self.aux_classifier(x)
            x = torch.nn.functional.interpolate(
                x, size=[w, h], mode='bilinear', align_corners=False)
            result['aux'] = x

            return result

    def make_model_resnet101(band_count, input_stride=1, class_count=2):
        deeplab = DeepLabResnet101(band_count, input_stride, class_count)
        return deeplab

    def make_model_stock(band_count, input_stride=1, class_count=2):
        deeplab = torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=False)
        last_class = deeplab.classifier[4] = torch.nn.Conv2d(
            256, class_count, kernel_size=7, stride=1, dilation=1)
        last_class_aux = deeplab.aux_classifier[4] = torch.nn.Conv2d(
            256, class_count, kernel_size=7, stride=1, dilation=1)
        input_filters = deeplab.backbone.conv1 = torch.nn.Conv2d(
            band_count, 64, kernel_size=7, stride=input_stride, dilation=1, padding=(3, 3), bias=False)
        return deeplab


if __name__ == '__main__':

    parser = training_cli_parser()
    args = training_cli_parser().parse_args()

    # ---------------------------------
    print('DATA')

    if not os.path.exists('/tmp/mul.tif'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.inference_img)
        print('Inference image bucket and prefix: {}, {}'.format(bucket, prefix))
        s3.download_file(bucket, prefix, '/tmp/mul.tif')
        del s3

    # ---------------------------------
    print('MODEL')

    if not os.path.exists('/tmp/deeplab.pth'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.model)
        print('Model bucket and prefix: {}, {}'.format(bucket, prefix))
        s3.download_file(bucket, prefix, '/tmp/deeplab.pth')
        del s3

    if args.architecture == 'resnet18':
        make_model = make_model_resnet18
    elif args.architecture == 'resnet34':
        make_model = make_model_resnet34
    elif args.architecture == 'resnet101':
        make_model = make_model_resnet101
    elif args.architecture == 'stock':
        make_model = make_model_stock
    else:
        raise Exception

    device = torch.device(args.backend)

    deeplab = make_model(
        len(args.bands),
        input_stride=args.input_stride,
        class_count=args.classes
    ).to(device)
    deeplab.load_state_dict(torch.load(
        '/tmp/deeplab.pth', map_location=device))

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
    print('INFERENCE')

    libchips.start(
        1,  # Number of threads
        1,  # Number of slots
        b'/tmp/mul.tif',  # Image data
        ctypes.c_void_p(0),  # Label data
        6,  # Make all rasters float32
        5,  # Make all labels int32
        3,  # Training mode
        args.window_size,
        len(args.bands),
        np.array(args.bands, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))

    with rio.open('/tmp/mul.tif') as ds:
        profile = ds.profile

    if args.final_predictions:
        profile.update(
            dtype=rio.uint8,
            count=1,
            compress='lzw'
        )
    else:
        profile.update(
            dtype=rio.float32,
            count=args.classes-1,
            compress='lzw'
        )

    deeplab.eval()
    with torch.no_grad():
        with rio.open('/tmp/pred.tif', 'w', **profile) as ds:
            width = libchips.get_width()
            height = libchips.get_height()
            print('{} {}'.format(width, height))
            for x_offset in range(0, width, args.window_size):
                if x_offset + args.window_size > width:
                    x_offset = width - args.window_size - 1
                for y_offset in range(0, height, args.window_size):
                    if y_offset + args.window_size > height:
                        y_offset = height - args.window_size - 1
                    window = rio.windows.Window(
                        x_offset, y_offset, args.window_size, args.window_size)
                    tensor = get_inference_window(libchips, x_offset, y_offset, len(
                        args.bands), args.window_size, args.image_nd)
                    if tensor is not None:
                        tensor = tensor.to(device)
                        out = deeplab(tensor)
                        if isinstance(out, dict):
                            out = out['out']
                        out = out.data.cpu().numpy()
                        if args.final_predictions:
                            out = np.apply_along_axis(np.argmax, 1, out)
                            out = np.array(out, dtype=np.uint8)
                            out = out * (0xff // (args.classes-1))
                            ds.write(out[0], window=window, indexes=1)
                        else:
                            for i in range(1, args.classes):
                                ds.write(out[0, i], window=window, indexes=i)
                print('{}'.format(100.0 * x_offset / width))

    s3 = boto3.client('s3')
    bucket, prefix = parse_s3_url(args.prediction_img)
    s3.upload_file('/tmp/pred.tif', bucket, prefix)
    del s3

    libchips.stop()
    libchips.deinit()

# ./deeplab_inference.py --architecture resnet18 --bands 1 2 3 4 5 6 7 8 9 10 11 12 --inference-img www --libchips xxx --prediction-img yyy --model zzz --max-sample-windows 133 --classes 2
