#!/usr/bin/env python3

import argparse
import copy
import ctypes
import hashlib
import os
import random
import re
import sys
import threading
import time
from typing import *
from urllib.parse import urlparse

import numpy as np  # type: ignore

import boto3  # type: ignore
import torch
import torchvision  # type: ignore

WATCHDOG_MUTEX: threading.Lock = threading.Lock()
WATCHDOG_TIME: float = time.time()
EVALUATIONS_BATCHES_DONE = 0

OPT = Union[torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW]
SCALER = Union[int, float]
INT2INT = Dict[int, int]
PRED = Union[Dict[str, torch.Tensor], torch.Tensor]

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

# Watchdog
if True:
    def watchdog_thread(seconds: int):
        """Code for the watchdog thread

        Arguments:
            seconds {int} -- The number of seconds of inactivity to allow before terminating
        """
        while True:
            time.sleep(60)
            if EVALUATIONS_BATCHES_DONE > 0:
                print('EVALUATIONS_DONE={}'.format(EVALUATIONS_BATCHES_DONE))
            with WATCHDOG_MUTEX:
                gap = time.time() - WATCHDOG_TIME
            if gap > seconds:
                print('TERMINATING DUE TO INACTIVITY {} > {}\n'.format(
                    gap, seconds), file=sys.stderr)
                os._exit(-1)

# Training
if True:
    def numpy_replace(np_arr: np.ndarray,
                      replacement_dict: INT2INT,
                      label_nd: SCALER) -> np.ndarray:
        """Replace the contents of np_arr according to the mapping given in replacement_dict

        Arguments:
            np_arr {np.ndarray} -- The numpy array to alter
            replacement_dict {INT2INT} -- The replacement mapping
            label_nd {SCALER} -- The label nodata

        Returns:
            np.ndarray -- The array with replacement performed
        """
        b = np.copy(np_arr)
        b[~np.isin(np_arr, list(replacement_dict.keys()))] = label_nd
        for k, v in replacement_dict.items():
            b[np_arr == k] = v
        return b

    def get_batch(libchips: ctypes.CDLL,
                  args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read the data specified in the given plan

        Arguments:
            libchips {ctypes.CDLL} -- A shared library handle used for reading data
            args {argparse.Namespace} -- The arguments dictionary

        Returns:
            Tuple[torch.Tensor, torch.Tensor] -- The raster data and label data as PyTorch tensors in a tuple
        """
        assert(args.label_nd is not None)

        shape = (len(args.bands), args.window_size, args.window_size)
        temp1 = np.zeros(shape, dtype=np.float32)
        temp1_ptr = temp1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        temp2 = np.zeros((args.window_size, args.window_size), dtype=np.int32)
        temp2_ptr = temp2.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        rasters = []
        labels = []
        for _ in range(args.batch_size):
            libchips.get_next(temp1_ptr, temp2_ptr)
            rasters.append(temp1.copy())
            labels.append(temp2.copy())

        raster_batch = []
        label_batch = []
        for raster, label in zip(rasters, labels):

            # NODATA from labels
            label = np.array(label, dtype=np.long)
            label = numpy_replace(label, args.label_map, args.label_nd)
            label_nds = (label == args.label_nd)

            # NODATA from rasters
            image_nds = np.zeros(raster[0].shape)
            if args.image_nd is not None:
                image_nds += (raster == args.image_nd).sum(axis=0)

            if args.by_the_power_of_greyskull:
                epsilon = 1e-7
                b2 = raster[2-1]
                b3 = raster[3-1]
                b4 = raster[4-1]
                b5 = raster[5-1]
                b8 = raster[8-1]
                b11 = raster[11-1]
                b12 = raster[12-1]
                ndwi = (b3 - b8)/(b3 + b8 + epsilon)
                mndwi = (b3 - b11)/(b3 + b11 + epsilon)
                wri = (b3 + b4)/(b8 + b12 + epsilon)
                ndci = (b5 - b4)/(b5 + b4 + epsilon)
                ndbi = (b11 - b8)/(b11 + b8 + epsilon)
                ndvi = (b8 - b4)/(b8 + b4 + epsilon)
                # blue = (b2 - args.mus[2-1]) / args.sigmas[2-1]
                # green = (b3 - args.mus[3-1]) / args.sigmas[3-1]
                # red = (b4 - args.mus[4-1]) / args.sigmas[4-1]
                # nir = (b8 - args.mus[8-1]) / args.sigmas[8-1]
                inds = [ndwi, mndwi, wri, ndci, ndbi, ndvi]
                raster = np.stack(inds, axis=0)
            else:
                for i in range(len(raster)):
                    raster[i] = (raster[i] - args.mus[i]) / args.sigmas[i]

            # NODATA from rasters
            image_nds += np.isnan(raster).sum(axis=0)

            # Set label NODATA, remove NaNs from rasters
            nodata = ((image_nds + label_nds) > 0)
            label[nodata == True] = args.label_nd
            for i in range(len(raster)):
                raster[i][nodata == True] = 0.0

            raster_batch.append(raster)
            label_batch.append(label)

        raster_batch_tensor = torch.from_numpy(np.stack(raster_batch, axis=0))
        label_batch_tensor = torch.from_numpy(np.stack(label_batch, axis=0))

        return (raster_batch_tensor, label_batch_tensor)

    def train(model: torch.nn.Module,
              opt: OPT,
              obj: torch.nn.CrossEntropyLoss,
              epochs: int,
              libchips: ctypes.CDLL,
              device: torch.device,
              args: argparse.Namespace,
              arg_hash: str,
              no_checkpoints: bool = True,
              starting_epoch: int = 0):
        """Train the model according the supplied data and (implicit and explicit) hyperparameters

        Arguments:
            model {torch.nn.Module} -- The model to train
            opt {torch.optim.SGD} -- The optimizer to use
            obj {torch.nn.CrossEntropyLoss} -- The objective function to use
            epochs {int} -- The number of "epochs"
            libchips {ctypes.CDLL} -- A shared library handle through which data can be read
            device {torch.device} -- The device to use
            args {argparse.Namespace} -- The arguments dictionary
            arg_hash {str} -- The arguments hash

        Keyword Arguments:
            no_checkpoints {bool} -- Whether to not write checkpoint files (default: {True})
            starting_epoch {int} -- The starting epoch (default: {0})
        """
        current_time = time.time()
        model.train()
        if (args.sigmoid > 0.0) and (args.sigmoid <= 1.0):
            obj2: Optional[torch.nn.MSELoss] = torch.nn.MSELoss().to(device)
            sigmoid: Optional[torch.nn.Sigmoid] = torch.nn.Sigmoid().to(device)
        else:
            obj2 = None
            sigmoid = None
        for i in range(starting_epoch, epochs):
            avg_loss = 0.0
            for _ in range(args.max_epoch_size):
                batch = get_batch(libchips, args)
                while not (batch[1] == 1).any() and args.reroll > random.random():
                    batch = get_batch(libchips, args)
                opt.zero_grad()
                pred: PRED = model(batch[0].to(device))
                label_long = batch[1].to(device)
                label_float = (batch[1] == 1).to(device, dtype=torch.float)
                with torch.autograd.detect_anomaly():
                    if isinstance(pred, dict):
                        pred_out: torch.Tensor = \
                            pred.get('out')  # type: ignore
                        out_loss = obj(pred_out, label_long)
                        pred_aux: torch.Tensor = \
                            pred.get('aux')  # type: ignore
                        aux_loss = obj(pred_aux, label_long)
                        loss = 1.0*out_loss + 0.4*aux_loss
                    else:
                        loss = obj(pred, label_long)
                    if obj2 is not None and sigmoid is not None:
                        if isinstance(pred, dict):
                            pred_out2 = pred.get('out')
                            if pred_out is not None:
                                non_background = pred_out[:, 1, :, :]
                            else:
                                raise Exception('Malformed dictionary')
                        else:
                            non_background = pred[:, 1, :, :]
                        non_background = sigmoid(non_background)
                        sigmoid_loss = obj2(non_background, label_float)
                        loss = args.sigmoid*sigmoid_loss + \
                            (1.0-args.sigmoid)*loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    opt.step()
                    avg_loss = avg_loss + loss.item()

            avg_loss = avg_loss / args.max_epoch_size
            if no_checkpoints or avg_loss < args.loss_cutoff:
                libchips.recenter(1)

            last_time = current_time
            current_time = time.time()
            print('\t\t epoch={} time={} avg_loss={}'.format(
                i, current_time - last_time, avg_loss))

            with WATCHDOG_MUTEX:
                global WATCHDOG_TIME
                WATCHDOG_TIME = time.time()

            if ((i == epochs - 1) or ((i > 0) and (i % 5 == 0) and args.s3_bucket and args.s3_prefix)) and not no_checkpoints:
                torch.save(model.state_dict(), 'deeplab.pth')
                s3 = boto3.client('s3')
                s3.upload_file('deeplab.pth', args.s3_bucket,
                               '{}/{}/deeplab_checkpoint_{}.pth'.format(args.s3_prefix, arg_hash, i))
                del s3

# Evaluation
if True:
    def evaluate(model: torch.nn.Module,
                 libchips: ctypes.CDLL,
                 device: torch.device,
                 args: argparse.Namespace,
                 arg_hash: str):
        """Evaluate the performance of the model given the various data.  Results are stored in S3.

        Arguments:
            model {torch.nn.Module} -- The model to evaluate
            libchips {ctypes.CDLL} -- A shared library handle through which data can be read
            device {torch.device} -- The device to use for evaluation
            args {argparse.Namespace} -- The arguments dictionary
            arg_hash {str} -- The hashed arguments
        """
        model.train()  # sic
        with torch.no_grad():
            num_classes = len(args.weights)
            tps = [0.0 for x in range(num_classes)]
            fps = [0.0 for x in range(num_classes)]
            fns = [0.0 for x in range(num_classes)]
            tns = [0.0 for x in range(num_classes)]

            for _ in range(args.max_eval_windows // args.batch_size):
                batch = get_batch(libchips, args)
                out = model(batch[0].to(device))
                if isinstance(out, dict):
                    out = out['out']
                out = out.data.cpu().numpy()
                out = np.apply_along_axis(np.argmax, 1, out)

                labels = batch[1].cpu().numpy()
                del batch

                # Make sure output reflects don't care values
                if args.label_nd is not None:
                    dont_care = (labels == args.label_nd)
                else:
                    dont_care = np.zeros(labels.shape)
                out = out + len(args.weights)*dont_care

                for j in range(len(args.weights)):
                    tps[j] = tps[j] + ((out == j)*(labels == j)).sum()
                    fps[j] = fps[j] + ((out == j)*(labels != j)).sum()
                    fns[j] = fns[j] + ((out != j)*(labels == j)).sum()
                    tns[j] = tns[j] + ((out != j)*(labels != j)).sum()

                if random.randint(0, args.batch_size * 4) == 0:
                    libchips.recenter(1)

                global EVALUATIONS_BATCHES_DONE
                EVALUATIONS_BATCHES_DONE += 1
                with WATCHDOG_MUTEX:
                    global WATCHDOG_TIME
                    WATCHDOG_TIME = time.time()

        print('True Positives  {}'.format(tps))
        print('False Positives {}'.format(fps))
        print('False Negatives {}'.format(fns))
        print('True Negatives  {}'.format(tns))

        recalls = []
        precisions = []
        for j in range(len(args.weights)):
            recall = tps[j] / (tps[j] + fns[j])
            recalls.append(recall)
            precision = tps[j] / (tps[j] + fps[j])
            precisions.append(precision)

        print('Recalls    {}'.format(recalls))
        print('Precisions {}'.format(precisions))

        f1s = []
        for j in range(len(args.weights)):
            f1 = 2 * (precisions[j] * recalls[j]) / \
                (precisions[j] + recalls[j])
            f1s.append(f1)
        print('f1 {}'.format(f1s))

        with open('/tmp/evaluations.txt', 'w') as evaluations:
            evaluations.write('True positives: {}\n'.format(tps))
            evaluations.write('False positives: {}\n'.format(fps))
            evaluations.write('False negatives: {}\n'.format(fns))
            evaluations.write('True negatives: {}\n'.format(tns))
            evaluations.write('Recalls: {}\n'.format(recalls))
            evaluations.write('Precisions: {}\n'.format(precisions))
            evaluations.write('f1 scores: {}\n'.format(f1s))

        s3 = boto3.client('s3')
        s3.upload_file('/tmp/evaluations.txt', args.s3_bucket,
                       '{}/{}/evaluations.txt'.format(args.s3_prefix, arg_hash))
        del s3

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
        parser.add_argument('--architecture',
                            required=True,
                            help='The desired model architecture',
                            choices=['resnet18', 'resnet34', 'resnet101', 'stock'])
        parser.add_argument('--backend',
                            help="Don't use this flag unless you know what you're doing: CPU is far slower than CUDA.",
                            choices=['cpu', 'cuda'], default='cuda')
        parser.add_argument('--bands',
                            required=True,
                            help='list of bands to train on (1 indexed)', nargs='+', type=int)
        parser.add_argument('--batch-size', default=16, type=int)
        parser.add_argument('--by-the-power-of-greyskull',
                            help='Pass this flag to enable special behavior',
                            action='store_true')
        parser.add_argument('--disable-eval',
                            help='Disable evaluation after training',
                            action='store_true')
        parser.add_argument('--epochs1', default=5, type=int)
        parser.add_argument('--epochs2', default=0, type=int)
        parser.add_argument('--epochs3', default=5, type=int)
        parser.add_argument('--epochs4', default=15, type=int)
        parser.add_argument('--image-nd',
                            default=None, type=float,
                            help='image value to ignore - must be on the first band')
        parser.add_argument('--input-stride',
                            default=2, type=int,
                            help='consult this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md')
        parser.add_argument('--label-img',
                            required=True,
                            help='labels to train')
        parser.add_argument('--label-map',
                            help='comma separated list of mappings to apply to training labels',
                            action=StoreDictKeyPair, default=None)
        parser.add_argument('--label-nd',
                            default=None, type=int,
                            help='label to ignore')
        parser.add_argument('--learning-rate1',
                            default=0.005, type=float,
                            help='float (probably between 10^-6 and 1) to tune SGD (see https://arxiv.org/abs/1206.5533)')
        parser.add_argument('--learning-rate2',
                            default=0.001, type=float,
                            help='float (probably between 10^-6 and 1) to tune SGD (see https://arxiv.org/abs/1206.5533)')
        parser.add_argument('--learning-rate3',
                            help='float (probably between 10^-6 and 1) to tune SGD (see https://arxiv.org/abs/1206.5533)', default=0.005, type=float)
        parser.add_argument('--learning-rate4',
                            default=0.001, type=float,
                            help='float (probably between 10^-6 and 1) to tune SGD (see https://arxiv.org/abs/1206.5533)')
        parser.add_argument('--libchips', required=True)
        parser.add_argument('--loss-cutoff', default=0.10, type=float)
        parser.add_argument('--max-epoch-size', default=sys.maxsize, type=int)
        parser.add_argument('--max-eval-windows',
                            default=sys.maxsize, type=int,
                            help='The maximum number of windows that will be used for evaluation')
        parser.add_argument('--optimizer', default='sgd',
                            choices=['sgd', 'adam', 'adamw'])
        parser.add_argument('--radius', default=10000)
        parser.add_argument('--read-threads', default=16, type=int)
        parser.add_argument('--reroll', default=0.90, type=float)
        parser.add_argument('--resolution-divisor', default=1, type=int)
        parser.add_argument('--s3-bucket',
                            required=True,
                            help='prefix to apply when saving models to s3')
        parser.add_argument('--s3-prefix',
                            required=True,
                            help='prefix to apply when saving models to s3')
        parser.add_argument('--sigmoid', default=0.0, type=float)
        parser.add_argument('--start-from',
                            help='The saved model to start the fourth phase from')
        parser.add_argument('--training-img',
                            required=True,
                            help='the input that you are training to produce labels for')
        parser.add_argument('--watchdog-seconds',
                            default=0, type=int,
                            help='The number of seconds that can pass without activity before the program is terminated (0 to disable)')
        parser.add_argument('--weights', nargs='+', required=True, type=float)
        parser.add_argument('--window-size', default=32, type=int)
        return parser

# Architectures
if True:
    class DeepLabResnet18(torch.nn.Module):
        def __init__(self, band_count, input_stride, class_count, divisor):
            super(DeepLabResnet18, self).__init__()
            resnet18 = torchvision.models.resnet.resnet18(pretrained=True)
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
                self.factor = 16 // divisor
            else:
                self.factor = 32 // divisor

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

    def make_model_resnet18(band_count, input_stride=1, class_count=2, divisor=1):
        deeplab = DeepLabResnet18(band_count, input_stride, class_count, divisor)
        return deeplab

    class DeepLabResnet34(torch.nn.Module):
        def __init__(self, band_count, input_stride, class_count, divisor):
            super(DeepLabResnet34, self).__init__()
            resnet34 = torchvision.models.resnet.resnet34(pretrained=True)
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
                self.factor = 16 // divisor
            else:
                self.factor = 32 // divisor

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

    def make_model_resnet34(band_count, input_stride=1, class_count=2, divisor=1):
        deeplab = DeepLabResnet34(band_count, input_stride, class_count, divisor)
        return deeplab

    class DeepLabResnet101(torch.nn.Module):
        def __init__(self, band_count, input_stride, class_count, divisor):
            super(DeepLabResnet101, self).__init__()
            resnet18 = torchvision.models.resnet.resnet101(
                pretrained=True, replace_stride_with_dilation=[False, True, True])
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
                self.factor = 4 // divisor
            else:
                self.factor = 8 // divisor

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

    def make_model_resnet101(band_count, input_stride=1, class_count=2, divisor=1):
        deeplab = DeepLabResnet101(band_count, input_stride, class_count, divisor)
        return deeplab

    def make_model_stock(band_count, input_stride=1, class_count=2, divisor=1):
        deeplab = torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=True)
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
    hashed_args = copy.copy(args)
    hashed_args.script = sys.argv[0]
    del hashed_args.backend
    del hashed_args.disable_eval
    del hashed_args.max_eval_windows
    del hashed_args.read_threads
    del hashed_args.watchdog_seconds
    arg_hash = hash_string(str(hashed_args))
    print('provided args: {}'.format(hashed_args))
    print('hash: {}'.format(arg_hash))

    args.mus = np.ndarray(len(args.bands), dtype=np.float64)
    args.sigmas = np.ndarray(len(args.bands), dtype=np.float64)
    mus_ptr = args.mus.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sigmas_ptr = args.sigmas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if args.by_the_power_of_greyskull:
        args.band_count = 6
    else:
        args.band_count = len(args.bands)

    # ---------------------------------
    print('DATA')

    if not os.path.exists('/tmp/mul.tif'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.training_img)
        print('training image bucket and prefix: {}, {}'.format(bucket, prefix))
        s3.download_file(bucket, prefix, '/tmp/mul.tif')
        try:
            s3.download_file(bucket, '{}.aux.xml'.format(
                prefix), '/tmp/mul.tif.aux.xml')
        except:
            pass
        del s3
    if not os.path.exists('/tmp/mul.tif.aux.xml'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.training_img)
        try:
            s3.download_file(bucket, '{}.aux.xml'.format(
                prefix), '/tmp/mul.tif.aux.xml')
        except:
            pass
        del s3
    if not os.path.exists('/tmp/mask.tif'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.label_img)
        print('training labels bucket and prefix: {}, {}'.format(bucket, prefix))
        s3.download_file(bucket, prefix, '/tmp/mask.tif')
        del s3

    # ---------------------------------
    print('NATIVE CODE')

    if not os.path.exists('/tmp/libchips.so'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.libchips)
        print('shared library bucket and prefix: {}, {}'.format(bucket, prefix))
        s3.download_file(bucket, prefix, '/tmp/libchips.so')
        del s3

    libchips = ctypes.CDLL('/tmp/libchips.so')
    libchips.init()
    libchips.start(
        args.read_threads,  # Number of threads
        max(args.read_threads, args.batch_size * 8),  # Number of slots
        b'/tmp/mul.tif',  # Image data
        b'/tmp/mask.tif',  # Label data
        6,  # Make all rasters float32
        5,  # Make all labels int32
        mus_ptr,  # means
        sigmas_ptr,  # standard deviations
        args.radius,  # typical radius of a component
        1,  # Training mode
        args.window_size,
        len(args.bands),
        np.array(args.bands, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))

    # ---------------------------------
    print('STATISTICS')

    print('\t MEANS={}'.format(args.mus))
    print('\t SIGMAS={}'.format(args.sigmas))

    # ---------------------------------
    print('RECORDING RUN')

    with open('/tmp/args.txt', 'w') as f:
        f.write(str(args) + '\n')
        f.write(str(sys.argv) + '\n')
    s3 = boto3.client('s3')
    s3.upload_file('/tmp/args.txt', args.s3_bucket,
                   '{}/{}/deeplab_training_args.txt'.format(args.s3_prefix, arg_hash))
    del s3

    # ---------------------------------
    print('INITIALIZING')

    complete_thru = -1
    current_epoch = 0
    current_pth = None
    if args.start_from is None:
        for pth in get_matching_s3_keys(
                bucket=args.s3_bucket,
                prefix='{}/{}/'.format(args.s3_prefix, arg_hash),
                suffix='pth'):
            m1 = re.match('.*deeplab_(\d+).pth$', pth)
            m2 = re.match('.*deeplab_checkpoint_(\d+).pth', pth)
            if m1:
                phase = int(m1.group(1))
                if phase > complete_thru:
                    complete_thru = phase
                    current_pth = pth
            if m2:
                checkpoint_epoch = int(m2.group(1))
                if checkpoint_epoch > current_epoch:
                    complete_thru = 4
                    current_epoch = checkpoint_epoch+1
                    current_pth = pth
    elif args.start_from is not None:
        complete_thru = 4
        current_epoch = 0
        current_pth = args.start_from

    device = torch.device(args.backend)

    if args.label_nd is None:
        args.label_nd = len(args.weights)
        print('\t WARNING: LABEL NODATA NOT SET, SETTING TO {}'.format(args.label_nd))

    if args.image_nd is None:
        print('\t WARNING: IMAGE NODATA NOT SET')

    if args.batch_size < 2:
        args.batch_size = 2
        print('\t WARNING: BATCH SIZE MUST BE AT LEAST 2, SETTING TO 2')

    if args.architecture == 'resnet18':
        make_model = make_model_resnet18
    elif args.architecture == 'resnet34':
        make_model = make_model_resnet34
    elif args.architecture == 'resnet101':
        make_model = make_model_resnet101
    elif args.architecture == 'stock':
        make_model = make_model_stock
        if args.window_size < 224:
            print('\t WARNING: WINDOWS SIZE {} IS PROBABLY TOO SMALL'.format(
                args.window_size))
    else:
        raise Exception

    args.max_epoch_size = min(args.max_epoch_size, int((libchips.get_width() * libchips.get_height() * 6.0) /
                                                       (args.window_size * args.window_size * 7.0 * args.batch_size)))

    print('\t STEPS PER EPOCH={}'.format(args.max_epoch_size))
    obj = torch.nn.CrossEntropyLoss(
        ignore_index=args.label_nd,
        weight=torch.FloatTensor(args.weights).to(device)
    ).to(device)

    # ---------------------------------
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

    if complete_thru == -1:
        deeplab = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=len(args.weights),
            divisor=args.resolution_divisor
        ).to(device)

    if complete_thru == 0:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'deeplab.pth')
        deeplab = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=len(args.weights),
            divisor=args.resolution_divisor
        ).to(device)
        deeplab.load_state_dict(torch.load('deeplab.pth'))
        del s3
        print('\t\t SUCCESSFULLY RESTARTED {}'.format(pth))
    elif complete_thru < 0:
        print('\t TRAINING FIRST AND LAST LAYERS')

        last_class = deeplab.classifier[4]
        last_class_aux = deeplab.aux_classifier[4]
        input_filters = deeplab.backbone.conv1
        for p in deeplab.parameters():
            p.requires_grad = False
        for p in last_class.parameters():
            p.requires_grad = True
        for p in last_class_aux.parameters():
            p.requires_grad = True
        for p in input_filters.parameters():
            p.requires_grad = True

        ps = []
        for n, p in deeplab.named_parameters():
            if p.requires_grad == True:
                ps.append(p)
            else:
                p.grad = None
        if args.optimizer == 'sgd':
            opt: OPT = torch.optim.SGD(
                ps, lr=args.learning_rate1, momentum=0.9)
        elif args.optimizer == 'adam':
            opt = torch.optim.Adam(ps, lr=args.learning_rate1)
        elif args.optimizer == 'adamw':
            opt = torch.optim.AdamW(ps, lr=args.learning_rate1)

        train(deeplab,
              opt,
              obj,
              args.epochs1,
              libchips,
              device,
              copy.copy(args),
              arg_hash)

        print('\t UPLOADING')

        torch.save(deeplab.state_dict(), 'deeplab.pth')
        s3 = boto3.client('s3')
        s3.upload_file('deeplab.pth', args.s3_bucket,
                       '{}/{}/deeplab_0.pth'.format(args.s3_prefix, arg_hash))
        del s3

    if complete_thru == 1:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'deeplab.pth')
        deeplab = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=len(args.weights),
            divisor=args.resolution_divisor
        ).to(device)
        deeplab.load_state_dict(torch.load('deeplab.pth'))
        del s3
        print('\t\t SUCCESSFULLY RESTARTED {}'.format(pth))
    elif complete_thru < 1:
        print('\t TRAINING FIRST AND LAST LAYERS AGAIN')

        last_class = deeplab.classifier[4]
        last_class_aux = deeplab.aux_classifier[4]
        input_filters = deeplab.backbone.conv1
        for p in deeplab.parameters():
            p.requires_grad = False
        for p in last_class.parameters():
            p.requires_grad = True
        for p in last_class_aux.parameters():
            p.requires_grad = True
        for p in input_filters.parameters():
            p.requires_grad = True

        ps = []
        for n, p in deeplab.named_parameters():
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

        train(deeplab,
              opt,
              obj,
              args.epochs2,
              libchips,
              device,
              copy.copy(args),
              arg_hash)

        print('\t UPLOADING')

        torch.save(deeplab.state_dict(), 'deeplab.pth')
        s3 = boto3.client('s3')
        s3.upload_file('deeplab.pth', args.s3_bucket,
                       '{}/{}/deeplab_1.pth'.format(args.s3_prefix, arg_hash))
        del s3

    if complete_thru == 2:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'deeplab.pth')
        deeplab = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=len(args.weights),
            divisor=args.resolution_divisor
        ).to(device)
        deeplab.load_state_dict(torch.load('deeplab.pth'))
        del s3
        print('\t\t SUCCESSFULLY RESTARTED {}'.format(pth))
    elif complete_thru < 2:
        print('\t TRAINING ALL LAYERS')

        for p in deeplab.parameters():
            p.requires_grad = True

        ps = []
        for n, p in deeplab.named_parameters():
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

        train(deeplab,
              opt,
              obj,
              args.epochs3,
              libchips,
              device,
              copy.copy(args),
              arg_hash)

        print('\t UPLOADING')

        torch.save(deeplab.state_dict(), 'deeplab.pth')
        s3 = boto3.client('s3')
        s3.upload_file('deeplab.pth', args.s3_bucket,
                       '{}/{}/deeplab_2.pth'.format(args.s3_prefix, arg_hash))
        del s3

    if complete_thru == 3:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'deeplab.pth')
        deeplab = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=len(args.weights),
            divisor=args.resolution_divisor
        ).to(device)
        deeplab.load_state_dict(torch.load('deeplab.pth'))
        del s3
        print('\t\t SUCCESSFULLY RESTARTED {}'.format(pth))
    elif complete_thru < 3:
        print('\t TRAINING ALL LAYERS AGAIN')

        for p in deeplab.parameters():
            p.requires_grad = True

        ps = []
        for n, p in deeplab.named_parameters():
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

        train(deeplab,
              opt,
              obj,
              args.epochs4,
              libchips,
              device,
              copy.copy(args),
              arg_hash,
              no_checkpoints=False)

        print('\t UPLOADING')

        torch.save(deeplab.state_dict(), 'deeplab.pth')
        s3 = boto3.client('s3')
        s3.upload_file('deeplab.pth', args.s3_bucket,
                       '{}/{}/deeplab.pth'.format(args.s3_prefix, arg_hash))
        del s3

    if complete_thru == 4:
        print('\t TRAINING ALL LAYERS FROM CHECKPOINT')

        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'deeplab.pth')
        deeplab = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=len(args.weights),
            divisor=args.resolution_divisor
        ).to(device)
        deeplab.load_state_dict(torch.load('deeplab.pth'))
        del s3

        for p in deeplab.parameters():
            p.requires_grad = True

        ps = []
        for n, p in deeplab.named_parameters():
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

        train(deeplab,
              opt,
              obj,
              args.epochs4,
              libchips,
              device,
              copy.copy(args),
              arg_hash,
              no_checkpoints=False,
              starting_epoch=current_epoch)

        print('\t UPLOADING')

        torch.save(deeplab.state_dict(), 'deeplab.pth')
        s3 = boto3.client('s3')
        s3.upload_file('deeplab.pth', args.s3_bucket,
                       '{}/{}/deeplab.pth'.format(args.s3_prefix, arg_hash))
        del s3

    libchips.stop()

    if not args.disable_eval:
        print('\t EVALUATING')
        libchips.start(
            args.read_threads,  # Number of threads
            args.read_threads,  # The number of read slots
            b'/tmp/mul.tif',  # Image data
            b'/tmp/mask.tif',  # Label data
            6,  # Make all rasters float32
            5,  # Make all labels int32
            ctypes.c_void_p(0),  # means
            ctypes.c_void_p(0),  # standard deviations
            args.radius,  # typical radius of a component
            2,  # Evaluation mode
            args.window_size,
            len(args.bands),
            np.array(args.bands, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        evaluate(deeplab,
                 libchips,
                 device,
                 copy.copy(args),
                 arg_hash)
        libchips.stop()

    libchips.deinit()
    exit(0)
