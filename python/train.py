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
#
# The code in this file is under the MIT license except where
# indicted.


import argparse
import codecs
import copy
import ctypes
import glob
import hashlib
import math
import os
import random
import re
import sys
import threading
import time
from typing import *
from urllib.parse import urlparse

import boto3
import numpy as np
import requests
import torch
import torchvision
from torch.optim.lr_scheduler import OneCycleLR


WATCHDOG_MUTEX: threading.Lock = threading.Lock()
WATCHDOG_TIME: float = time.time()
EVALUATIONS_BATCHES_DONE = 0

# Bootstrap
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

    def load_code(uri: str) -> None:
        arch_str = read_text(uri)
        arch_code = compile(arch_str, uri, 'exec')
        exec(arch_code, globals())

# Stubs
if True:
    def parse_s3_url(*argv):
        raise Exception()

    def get_matching_s3_keys(*argv):
        raise Exception()

    def watchdog_thread(*argv):
        raise Exception()

    def get_batch(*argv):
        raise Exception()

    def train(*argv):
        raise Exception()

    def make_model(*argv):
        raise Exception()

    def evaluate(*argv):
        raise Exception()

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
        parser.add_argument('--architecture-code', required=True, type=str)
        parser.add_argument('--s3-code', required=False, type=str,
                            default='https://raw.githubusercontent.com/geotrellis/deeplab-nlcd/master/python/code/s3.py')
        parser.add_argument('--watchdog-code', required=False, type=str,
                            default='https://raw.githubusercontent.com/geotrellis/deeplab-nlcd/master/python/code/watchdog.py')
        parser.add_argument('--training-code', required=False, type=str,
                            default='https://raw.githubusercontent.com/geotrellis/deeplab-nlcd/master/python/code/training.py')
        parser.add_argument('--evaluation-code', required=False, type=str,
                            default='https://raw.githubusercontent.com/geotrellis/deeplab-nlcd/master/python/code/evaluation.py')
        parser.add_argument('--backend',
                            choices=['cpu', 'cuda'], default='cuda')
        parser.add_argument('--bands', required=True, nargs='+', type=int,
                            help='list of bands to train on (1 indexed)')
        parser.add_argument('--batch-size', default=16, type=int)
        parser.add_argument('--epochs1', default=0, type=int)
        parser.add_argument('--epochs2', default=13, type=int)
        parser.add_argument('--epochs3', default=0, type=int)
        parser.add_argument('--epochs4', default=33, type=int)
        parser.add_argument('--forbidden-imagery-value',
                            default=None, type=float)
        parser.add_argument('--forbidden-label-value',
                            default=None, type=int)
        parser.add_argument('--desired-label-value', default=None, type=int)
        parser.add_argument('--image-nd',
                            default=None, type=float,
                            help='image value to ignore - must be on the first band')
        parser.add_argument('--input-stride',
                            default=2, type=int,
                            help='consult this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md')
        parser.add_argument('--label-img',
                            required=True, nargs='+', type=str,
                            help='labels to train')
        parser.add_argument('--label-map',
                            required=True, help='comma separated list of mappings to apply to training labels',
                            action=StoreDictKeyPair, default=None)
        parser.add_argument('--label-nd',
                            default=None, type=int,
                            help='label to ignore')
        parser.add_argument('--learning-rate1',
                            default=1e-4, type=float,
                            help='float (probably between 10^-6 and 1) to tune SGD (see https://arxiv.org/abs/1206.5533)')
        parser.add_argument('--learning-rate2',
                            default=1e-2, type=float,
                            help='float (probably between 10^-6 and 1) to tune SGD (see https://arxiv.org/abs/1206.5533)')
        parser.add_argument('--learning-rate3',
                            default=1e-4, type=float,
                            help='float (probably between 10^-6 and 1) to tune SGD (see https://arxiv.org/abs/1206.5533)')
        parser.add_argument('--learning-rate4',
                            default=1e-2, type=float,
                            help='float (probably between 10^-6 and 1) to tune SGD (see https://arxiv.org/abs/1206.5533)')
        parser.add_argument('--libchips', required=True)
        parser.add_argument('--max-epoch-size', default=sys.maxsize, type=int)
        parser.add_argument('--max-eval-windows',
                            default=sys.maxsize, type=int,
                            help='The maximum number of windows that will be used for evaluation')
        parser.add_argument('--no-eval',
                            help='Disable evaluation after training',
                            action='store_true')
        parser.add_argument('--no-upload',
                            help='Do not upload anything to S3',
                            action='store_true')
        parser.add_argument('--bce',
                            help='Use binary cross-entropy for binary-only regression',
                            action='store_true')
        parser.add_argument('--output',
                            required=False, type=str,
                            help='Model output location')
        parser.add_argument('--optimizer', default='adam',
                            choices=['sgd', 'adam', 'adamw'])
        parser.add_argument('--radius', default=10000)
        parser.add_argument('--read-threads', type=int)
        parser.add_argument('--reroll', default=0.25, type=float)
        parser.add_argument('--resolution-divisor', default=1, type=int)
        parser.add_argument('--s3-bucket',
                            required=True,
                            help='prefix to apply when saving models to s3')
        parser.add_argument('--s3-prefix',
                            required=True,
                            help='prefix to apply when saving models to s3')
        parser.add_argument('--start-from',
                            help='The saved model to start the fourth phase from')
        parser.add_argument('--training-img',
                            required=True, nargs='+', type=str,
                            help='The input that you are training to produce labels for')
        parser.add_argument('--watchdog-seconds',
                            default=0, type=int,
                            help='The number of seconds that can pass without activity before the program is terminated (0 to disable)')
        parser.add_argument('--class-weights', nargs='+', type=float)
        parser.add_argument('--window-size-imagery', default=32, type=int)
        parser.add_argument('--window-size-labels', default=32, type=int)
        return parser


if __name__ == '__main__':

    parser = training_cli_parser()
    args = training_cli_parser().parse_args()
    hashed_args = copy.deepcopy(args)
    hashed_args.script = sys.argv[0]
    del hashed_args.backend
    del hashed_args.no_eval
    del hashed_args.no_upload
    del hashed_args.max_eval_windows
    del hashed_args.read_threads
    del hashed_args.watchdog_seconds
    arg_hash = hash_string(str(hashed_args))
    print('provided args: {}'.format(hashed_args))
    print('hash: {}'.format(arg_hash))

    assert(args.window_size_labels % args.window_size_imagery == 0)
    if args.window_size_labels != args.window_size_imagery:
        import scipy.ndimage  # Get ready to use scipy.ndimage.zoom

    tmp_mul = '/tmp/mul{}.tif'
    tmp_label = '/tmp/mask{}.tif'
    tmp_libchips = '/tmp/libchips.so.1.1'

    args.band_count = len(args.bands)

    load_code(args.s3_code)
    load_code(args.watchdog_code)
    load_code(args.training_code)
    load_code(args.evaluation_code)
    load_code(args.architecture_code)

    # ---------------------------------
    print('VALUES')

    if '-regression' in args.architecture_code and args.forbidden_imagery_value is None and args.image_nd is not None:
        print('WARNING: FORBIDDEN IMAGERY VALUE NOT SET, SETTING TO {}'.format(
            args.image_nd))
        args.forbidden_imagery_value = args.image_nd
    if '-regression' in args.architecture_code and args.forbidden_label_value is None and args.label_nd is not None:
        for k, v in args.label_map.items():
            if v == args.label_nd:
                print('WARNING: FORBIDDEN LABEL VALUE NOT SET, SETTING TO {}'.format(k))
                args.forbidden_label_value = k

    if '-regression' in args.architecture_code and args.forbidden_imagery_value is None:
        print('WARNING: PERFORMING REGRESSION WITHOUT A FORBIDDEN IMAGERY VALUE')
    if '-regression' in args.architecture_code and args.forbidden_label_value is None:
        print('WARNING: PERFORMING REGRESSION WITHOUT A FORBIDDEN LABEL VALUE')

    if not args.class_weights:
        if '-binary' in args.architecture_code:
            args.class_weights = [1.0] * 2
        else:
            args.class_weights = [1.0] * (len(args.label_map)-1)
    class_count = len(args.class_weights)

    if args.label_nd is None:
        args.label_nd = class_count
        print('\t WARNING: LABEL NODATA NOT SET, SETTING TO {}'.format(args.label_nd))

    if args.image_nd is None:
        print('\t WARNING: IMAGE NODATA NOT SET')

    if args.batch_size < 2:
        args.batch_size = 2
        print('\t WARNING: BATCH SIZE MUST BE AT LEAST 2, SETTING TO 2')

    # ---------------------------------
    print('DATA')

    # Look for newline-delimited lists of files
    if len(args.training_img) == 1 and args.training_img[0].endswith('.list'):
        text = read_text(args.training_img[0])
        args.training_img = list(
            filter(lambda line: len(line) > 0, text.split('\n')))
    if len(args.label_img) == 1 and args.label_img[0].endswith('.list'):
        text = read_text(args.label_img[0])
        args.label_img = list(
            filter(lambda line: len(line) > 0, text.split('\n')))

    # Image⨯label pairs
    args.pairs = list(zip(args.training_img, args.label_img))
    for i in range(len(args.pairs)):
        training_img = args.training_img[i]
        label_img = args.label_img[i]

        if training_img.startswith('s3://'):
            tmp_mul_local = tmp_mul.format(i)
            if not os.path.exists(tmp_mul_local):
                s3 = boto3.client('s3')
                bucket, prefix = parse_s3_url(training_img)
                print('Training image bucket and prefix: {}, {}'.format(
                    bucket, prefix))
                s3.download_file(bucket, prefix, tmp_mul_local)
                del s3
            args.training_img[i] = tmp_mul_local

        if label_img.startswith('s3://'):
            tmp_label_local = tmp_label.format(i)
            if not os.path.exists(tmp_label_local):
                s3 = boto3.client('s3')
                bucket, prefix = parse_s3_url(label_img)
                print('Training labels bucket and prefix: {}, {}'.format(
                    bucket, prefix))
                s3.download_file(bucket, prefix, tmp_label_local)
                del s3
            args.label_img[i] = tmp_label_local

    if not args.read_threads:
        args.read_threads = len(args.pairs)

    # ---------------------------------
    print('NATIVE CODE')

    if args.libchips.startswith('s3://'):
        if not os.path.exists(tmp_libchips):
            s3 = boto3.client('s3')
            bucket, prefix = parse_s3_url(args.libchips)
            print('shared library bucket and prefix: {}, {}'.format(bucket, prefix))
            s3.download_file(bucket, prefix, tmp_libchips)
            del s3
        args.libchips = tmp_libchips

    libchips = ctypes.CDLL(args.libchips)
    libchips.recenter.argtypes = [ctypes.c_int]
    libchips.get_next.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int)
    ]
    libchips.start.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

    libchips.init()
    libchips.start(
        args.read_threads,  # Number of threads
        args.read_threads * 2,  # Number of slots
        len(args.pairs),  # The number of pairs
        b'/tmp/mul%d.tif',  # Image data
        b'/tmp/mask%d.tif',  # Label data
        6,  # Make all rasters float32
        5,  # Make all labels int32
        None,  # means
        None,  # standard deviations
        args.radius,  # typical radius of a component
        1,  # Training mode
        args.window_size_imagery,
        args.window_size_labels,
        len(args.bands),
        np.array(args.bands, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))

    # ---------------------------------
    print('RECORDING RUN')

    with open('/tmp/args.txt', 'w') as f:
        f.write(str(args) + '\n')
        f.write(str(sys.argv) + '\n')
    if not args.no_upload:
        s3 = boto3.client('s3')
        s3.upload_file('/tmp/args.txt', args.s3_bucket,
                       '{}/{}/training_args.txt'.format(args.s3_prefix, arg_hash))
        del s3

    # ---------------------------------
    print('CONSIDERING PREVIOUS PROGRESS')

    current_epoch = 0
    current_pth = None

    if args.start_from is None:
        if not args.no_upload:
            for pth in get_matching_s3_keys(
                    bucket=args.s3_bucket,
                    prefix='{}/{}/'.format(args.s3_prefix, arg_hash),
                    suffix='pth'):
                m1 = re.match('.*weights_checkpoint_(\d+).pth', pth)
                if m1:
                    print('\t found {}'.format(pth))
                    checkpoint_epoch = int(m1.group(1))
                    if checkpoint_epoch > current_epoch:
                        current_epoch = checkpoint_epoch+1
                        current_pth = pth
    elif args.start_from is not None:
        current_epoch = 1
        current_pth = args.start_from

    print('\t current_epoch = {}'.format(current_epoch))
    print('\t current_pth = {}'.format(current_pth))

    # ---------------------------------
    print('INITIALIZING')

    device = torch.device(args.backend)

    natural_epoch_size = 0.0
    for i in range(len(args.pairs)):
        natural_epoch_size = natural_epoch_size + \
            (libchips.get_width(i) * libchips.get_height(i))
    natural_epoch_size = (6.0 * natural_epoch_size) / \
        (7.0 * args.window_size_imagery * args.window_size_imagery)
    natural_epoch_size = int(natural_epoch_size)
    print('\t NATURAL EPOCH SIZE={}'.format(natural_epoch_size))
    args.max_epoch_size = min(args.max_epoch_size, natural_epoch_size)
    print('\t STEPS PER EPOCH={}'.format(args.max_epoch_size))

    obj = {
        'seg': torch.nn.CrossEntropyLoss(
            ignore_index=args.label_nd,
            weight=torch.FloatTensor(args.class_weights).to(device)
        ).to(device),
        '2seg': torch.nn.BCEWithLogitsLoss().to(device),
        'l1': torch.nn.L1Loss().to(device),
        'l2': torch.nn.MSELoss().to(device)
    }

    # ---------------------------------
    # WATCHDOG

    if args.watchdog_seconds > 0:
        print('STARTING WATCHDOG')
        w_th = threading.Thread(target=watchdog_thread,
                                args=(args.watchdog_seconds,))
        w_th.daemon = True
        w_th.start()
    else:
        print('NOT STARTING WATCHDOG')

    # ---------------------------------
    print('TRAINING')

    model = make_model(
        args.band_count,
        input_stride=args.input_stride,
        class_count=class_count,
        divisor=args.resolution_divisor,
        pretrained=True
    ).to(device)

    # Phase 1, 2, 3
    if current_epoch == 0:
        print('\t TRAINING FIRST AND LAST LAYERS (1/2)')

        for p in model.parameters():
            p.requires_grad = False
        for layer in model.input_layers + model.output_layers:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(model, 'immutable_layers'):
            for layer in model.immutable_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        ps = []
        for n, p in model.named_parameters():
            if p.requires_grad == True:
                ps.append(p)
            else:
                p.grad = None
        if args.optimizer == 'sgd':
            opt = torch.optim.SGD(
                ps, lr=args.learning_rate1, momentum=0.9)
        elif args.optimizer == 'adam':
            opt = torch.optim.Adam(ps, lr=args.learning_rate1)
        elif args.optimizer == 'adamw':
            opt = torch.optim.AdamW(ps, lr=args.learning_rate1)

        train(model,
              opt,
              None,
              obj,
              args.epochs1,
              libchips,
              device,
              copy.deepcopy(args),
              arg_hash)

        print('\t TRAINING FIRST AND LAST LAYERS (2/2)')

        for p in model.parameters():
            p.requires_grad = False
        for layer in model.input_layers + model.output_layers:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(model, 'immutable_layers'):
            for layer in model.immutable_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        ps = []
        for n, p in model.named_parameters():
            if p.requires_grad == True:
                ps.append(p)
            else:
                p.grad = None
        if args.optimizer == 'sgd':
            opt = torch.optim.SGD(ps, lr=args.learning_rate2, momentum=0.9)
        elif args.optimizer == 'adam':
            opt = torch.optim.Adam(ps, lr=args.learning_rate2)
        elif args.optimizer == 'adamw':
            opt = torch.optim.AdamW(ps, lr=args.learning_rate2)
        if args.epochs2 > 0:
            sched = OneCycleLR(opt, max_lr=args.learning_rate2,
                               epochs=args.epochs2, steps_per_epoch=args.max_epoch_size)
        else:
            sched = None

        train(model,
              opt,
              sched,
              obj,
              args.epochs2,
              libchips,
              device,
              copy.deepcopy(args),
              arg_hash)

        print('\t TRAINING ALL LAYERS (1/2)')

        for p in model.parameters():
            p.requires_grad = True
        if hasattr(model, 'immutable_layers'):
            for layer in model.immutable_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        ps = []
        for n, p in model.named_parameters():
            if p.requires_grad == True:
                ps.append(p)
            else:
                p.grad = None
        if args.optimizer == 'sgd':
            opt = torch.optim.SGD(ps, lr=args.learning_rate3, momentum=0.9)
        elif args.optimizer == 'adam':
            opt = torch.optim.Adam(ps, lr=args.learning_rate3)
        elif args.optimizer == 'adamw':
            opt = torch.optim.AdamW(ps, lr=args.learning_rate3)

        train(model,
              opt,
              None,
              obj,
              args.epochs3,
              libchips,
              device,
              copy.deepcopy(args),
              arg_hash)

    # Phase 4
    print('\t TRAINING ALL LAYERS (2/2)')

    if current_epoch != 0:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'weights.pth')
        model.load_state_dict(torch.load('weights.pth'))
        del s3
        print('\t\t SUCCESSFULLY RESTARTED {}'.format(current_pth))

    for p in model.parameters():
        p.requires_grad = True
    if hasattr(model, 'immutable_layers'):
        for layer in model.immutable_layers:
            for p in layer.parameters():
                p.requires_grad = False

    ps = []
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            ps.append(p)
        else:
            p.grad = None
    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(ps, lr=args.learning_rate4, momentum=0.9)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(ps, lr=args.learning_rate4)
    elif args.optimizer == 'adamw':
        opt = torch.optim.AdamW(ps, lr=args.learning_rate4)
    if args.epochs4 > 0:
        sched = OneCycleLR(
            opt,
            max_lr=args.learning_rate4,
            epochs=args.epochs4-current_epoch,
            steps_per_epoch=args.max_epoch_size
        )
    else:
        sched = None

    train(model,
          opt,
          sched,
          obj,
          args.epochs4,
          libchips,
          device,
          copy.deepcopy(args),
          arg_hash,
          no_checkpoints=False,
          starting_epoch=current_epoch)

    if not args.no_upload:
        print('\t UPLOADING')
        torch.save(model.state_dict(), 'weights.pth')
        s3 = boto3.client('s3')
        s3.upload_file('weights.pth', args.s3_bucket,
                       '{}/{}/weights.pth'.format(args.s3_prefix, arg_hash))
        if args.output is not None and args.output.startswith('s3://'):
            parts = args.output[5:].split('/')
            s3_bucket = parts[0]
            s3_prefix = '/'.join(parts[1:])
            s3.upload_file('weights.pth', s3_bucket, s3_prefix)
        del s3

    libchips.stop()

    if not args.no_eval:
        print('\t EVALUATING')
        libchips.start(
            args.read_threads,  # Number of threads
            args.read_threads * 2,  # The number of read slots
            len(args.pairs),  # The number of pairs
            b'/tmp/mul%d.tif',  # Image data
            b'/tmp/mask%d.tif',  # Label data
            6,  # Make all rasters float32
            5,  # Make all labels int32
            None,  # means
            None,  # standard deviations
            args.radius,  # typical radius of a component
            2,  # Evaluation mode
            args.window_size_imagery,
            args.window_size_labels,
            len(args.bands),
            np.array(args.bands, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        evaluate(model,
                 libchips,
                 device,
                 copy.deepcopy(args),
                 arg_hash)
        libchips.stop()

    libchips.deinit()
    exit(0)
