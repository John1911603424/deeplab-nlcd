#!/usr/bin/env python3

import argparse
import copy
import hashlib
import os
import re
import sys
import threading
import time
from typing import *
from urllib.parse import urlparse

import boto3  # type: ignore
from PIL import Image  # type: ignore

import numpy as np  # type: ignore
import rasterio as rio  # type: ignore
import torch
import torchvision  # type: ignore

if os.environ.get('CURL_CA_BUNDLE') is None:
    os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

MEANS: List[float] = []
STDS: List[float] = []
WATCHDOG_MUTEX: threading.Lock = threading.Lock()
WATCHDOG_TIME: float = time.time()
TRAINING_MUTEX: threading.Lock = threading.Lock()
TRAINING_BATCHES: List[Tuple[torch.Tensor, torch.Tensor]] = []
EVALUATION_MUTEX: threading.Lock = threading.Lock()
EVALUATION_BATCHES: List[Tuple[torch.Tensor, torch.Tensor]] = []


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

    def get_matching_s3_keys(bucket: str, prefix: str = '', suffix: str = '') -> Generator[str, None, None]:
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

# Misc.
if True:
    def retry_read(ds, band: int, window=None, retries: int = 3):
        """Make some number of attempts to read the requested data

        Arguments:
            ds {rasterio dataset} -- The raster dataset to read from
            band {int} -- The band to read from

        Keyword Arguments:
            window {rasterio window} -- The window to read from or None for the whole image (default: {None})
            retries {int} -- The maximum number of attempts (default: {3})

        Returns:
            np.ndarray -- The requested data
        """
        # retry failed reads (they appear to be transient)
        for i in range(retries):
            try:
                return ds.read(band, window=window)
            except rio.errors.RasterioIOError:
                print('Read error for band {} at window {} on try {} of {}'.format(
                    band, window, i+1, retries))
                continue

    def chunks(l: List[Any], n: int) -> Generator[List[Any], None, None]:
        """Break a list of items into equal-sized chunks

        Arguments:
            l {List[Any]} -- The list of items to break-up
            n {int} -- The desired size of each chunk

        Returns:
            Generator[List[Any], None, None] -- A generator of lists
        """
        # https://chrisalbon.com/python/data_wrangling/break_list_into_chunks_of_equal_size/
        for i in range(0, len(l), n):
            yield l[i:i+n]

    def numpy_replace(np_arr: np.ndarray, replacement_dict: Dict[int, int]) -> np.ndarray:
        """Replace the contents of np_arr according to the mapping given in replacement_dict

        Arguments:
            np_arr {np.ndarray} -- The numpy array to alter
            replacement_dict {Dict[int, int]} -- The replacement mapping

        Returns:
            np.ndarray -- The array with replacement performed
        """
        b = np.copy(np_arr)
        for k, v in replacement_dict.items():
            b[np_arr == k] = v
        return b

    def watchdog_thread(seconds: int):
        """Code for the watchdog thread

        Arguments:
            seconds {int} -- The number of seconds of inactivity to allow before terminating
        """
        while True:
            time.sleep(max(seconds//2, 1))
            with WATCHDOG_MUTEX:
                gap = time.time() - WATCHDOG_TIME
            if gap > seconds:
                print('TERMINATING DUE TO INACTIVITY {} > {}\n'.format(
                    gap, seconds), file=sys.stderr)
                os._exit(-1)

# Sampling
if True:
    def get_random_sample(ds: rio.io.DatasetReader, window_size: int, image_nd: Union[None, Union[float, int]]) -> np.ndarray:
        """Get a random sample from the given raster dataset

        Arguments:
            ds {rio.io.DatasetReader} -- The dataset from which to sample
            window_size {int} -- The size of the sample window
            image_nd {Union[None, Union[float, int]]} -- The image nodata

        Returns:
            np.ndarray -- A stack of samples from image bands
        """
        x = np.random.randint(0, ds.width//window_size - 1)
        y = np.random.randint(0, ds.height//window_size - 1)
        window = rio.windows.Window(
            x * window_size, y * window_size,
            window_size, window_size)

        data = []
        for band in ds.indexes:
            a = retry_read(ds, band, window=window)
            if image_nd is not None:
                a = np.extract(a != image_nd, a)
            a = a[~np.isnan(a)]
            data.append(a)
        data = np.stack(data, axis=1)

        return data

# Training
if True:
    def get_window(arguments: Tuple) -> Tuple[np.ndarray, int]:
        """Get a window as specified by the `arguments` tuple

        Arguments:
            arguments {Tuple} -- A tuple of window, band, raster dataset, and label dataset

        Returns:
            Tuple[np.ndarray, int] -- The requested window and its band as a tuple
        """
        window, band, raster_ds, label_ds = arguments

        if band == 0:
            a = np.array(retry_read(label_ds, 1, window=window), dtype=np.long)
        else:
            a = np.float32(retry_read(raster_ds, band, window=window))
            a = (a - MEANS[band-1]) / STDS[band-1]

        return a, band

    def execute_plan(raster_ds: rio.io.DatasetReader, label_ds: rio.io.DatasetReader, band_count: int, plan: List, label_mappings: Dict[int, int], label_nd: Union[int, float], image_nd: Union[None, Union[int, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read the data specified in the given plan

        Arguments:
            raster_ds {rio.io.DatasetReader} -- The raster dataset to read from
            label_ds {rio.io.DatasetReader} -- The label dataset to read from
            band_count {int} -- The number of bands involved, not counting the labels (used for chunking)
            plan {List} -- The execution plan
            label_mappings {Dict[int, int]} -- The mapping between native and internal classes in the lable data
            label_nd {Union[int, float]} -- The label nodata
            image_nd {Union[None, Union[int, float]]} -- The image nodata

        Returns:
            Tuple[torch.Tensor, torch.Tensor] -- The raster data and label data as PyTorch tensors in a tuple
        """
        assert(label_nd is not None)

        # Read the plan
        data = []
        labels = []
        for d, i in map(get_window, plan):
            if i == 0:
                labels.append(d)
            else:
                data.append(d)

        # NODATA processing
        raster_batch = []
        label_batch = []
        if image_nd is not None:
            image_nd = (image_nd - MEANS[0])/STDS[0]

        for raster, label in zip(chunks(data, band_count), labels):

            # NODATA from labels
            label = numpy_replace(label, label_mappings)
            label_nds = (label == label_nd)

            # NODATA from rasters
            nan = np.isnan(raster[0])
            if image_nd is not None:
                image_nds = (raster[0] == image_nd) + nan
            else:
                image_nds = nan

            # Set label NODATA, remove NaNs from rasters
            nodata = ((image_nds + label_nds) > 0)
            label[nodata == True] = label_nd
            for i in range(len(raster)):
                raster[i][nodata == True] = 0.0

            raster_batch.append(np.stack(raster, axis=0))
            label_batch.append(label)

        raster_batch1 = np.stack(raster_batch, axis=0)
        raster_batch2 = torch.from_numpy(raster_batch1)
        label_batch1 = np.stack(label_batch, axis=0)
        label_batch2 = torch.from_numpy(label_batch1)

        return (raster_batch2, label_batch2)

    def training_readahead_thread(raster_filename: str, label_filename: str, window_size: int, batch_size: int, bands: List[int], epoch_size: int, label_mappings: Dict[int, int], label_nd: Union[int, float], image_nd: Union[None, Union[int, float]]):
        """Code for the training read-ahead thread

        Arguments:
            raster_filename {str} -- The name of raster to read from 
            label_filename {str} -- The name of the label raster to read from
            window_size {int} -- The window size
            batch_size {int} -- The batch size
            bands {List[int]} -- The list of bands to read from
            epoch_size {int} -- The epoch size
            label_mappings {Dict[int, int]} -- The mapping between native and internal classes
            label_nd {Union[int, float]} -- The label nodata
            image_nd {Union[None, Union[int, float]]} -- The image nodata
        """
        global TRAINING_BATCHES
        global TRAINING_MUTEX

        with rio.open(raster_filename) as raster_ds, rio.open(label_filename) as mask_ds:
            while True:
                time.sleep(0.01)
                with TRAINING_MUTEX:
                    if len(TRAINING_BATCHES) < epoch_size:
                        bands_plus = bands + [0]
                        plan = []
                        for batch_index in range(batch_size):
                            x = 0
                            y = 0
                            while ((x + y) % 7) == 0:
                                x = np.random.randint(0, width/window_size - 1)
                                y = np.random.randint(
                                    0, height/window_size - 1)
                            window = rio.windows.Window(
                                x * window_size, y * window_size,
                                window_size, window_size)
                            for band in bands_plus:
                                args = (window, band, raster_ds, mask_ds)
                                plan.append(args)
                        batch = execute_plan(raster_ds, mask_ds, len(bands), plan,
                                             label_mappings, label_nd, image_nd)
                        TRAINING_BATCHES = [batch] + TRAINING_BATCHES

    def train(model: torch.nn.Module, opt: torch.optim.SGD, obj: torch.nn.CrossEntropyLoss, batches_per_epoch: int, epochs: int, batch_size: int, raster_ds: rio.io.DatasetReader, label_ds: rio.io.DatasetReader, width: int, height: int, window_size: int, device: torch.device, bands: List[int], label_mapping: Dict[int, int], label_nd: Union[int, float], img_nd: Union[None, Union[int, float]], bucket_name: str, s3_prefix: str, arg_hash: str, no_checkpoints: bool = True, starting_epoch: int = 0):
        """Train the model according the supplied data and (implicit and explicit) hyperparameters

        Arguments:
            model {torch.nn.Module} -- The model to train
            opt {torch.optim.SGD} -- The optimizer to use
            obj {torch.nn.CrossEntropyLoss} -- The objective function to use
            batches_per_epoch {int} -- The number of batches per "epoch"
            epochs {int} -- The number of "epochs"
            batch_size {int} -- The batch size
            raster_ds {rio.io.DatasetReader} -- The imagery dataset
            label_ds {rio.io.DatasetReader} -- The label dataset
            width {int} -- The width of the dataset
            height {int} -- The height of the
            window_size {int} -- The window size
            device {torch.device} -- The device to use
            bands {List[int]} -- The imagery bands to use
            label_mapping {Dict[int, int]} -- The mapping between native and internal classes
            label_nd {Union[int, float]} -- The label nodata
            img_nd {Union[None, Union[int, float]]} -- The imagery nodata
            bucket_name {str} -- The bucket name
            s3_prefix {str} -- The bucket prefix
            arg_hash {str} -- The arguments hash

        Keyword Arguments:
            no_checkpoints {bool} -- Whether to not write checkpoint files (default: {True})
            starting_epoch {int} -- The starting epoch (default: {0})
        """
        current_time = time.time()

        model.train()
        for i in range(starting_epoch, epochs):
            avg_loss = 0.0

            for j in range(batches_per_epoch):

                while True:
                    with TRAINING_MUTEX:
                        if len(TRAINING_BATCHES) > 0:
                            batch = TRAINING_BATCHES.pop()
                            break
                        else:
                            time.sleep(1)

                opt.zero_grad()
                pred = model(batch[0].to(device))
                label = batch[1].to(device)
                if isinstance(pred, dict):
                    pred_out: torch.Tensor = pred.get('out')  # type: ignore
                    pred_aux: torch.Tensor = pred.get('aux')  # type: ignore
                    out_loss = obj(pred_out, label)
                    aux_loss = obj(pred_aux, label)
                    loss = 1.0*out_loss + 0.4*aux_loss
                else:
                    loss = 1.0*obj(pred, label)
                loss.backward()
                opt.step()
                avg_loss = avg_loss + loss.item()

            avg_loss = avg_loss / batches_per_epoch

            last_time = current_time
            current_time = time.time()
            print('\t\t epoch={} time={} avg_loss={}'.format(
                i, current_time - last_time, avg_loss))

            with WATCHDOG_MUTEX:
                global WATCHDOG_TIME
                WATCHDOG_TIME = time.time()

            if ((i == epochs - 1) or ((i > 0) and (i % 5 == 0) and bucket_name and s3_prefix)) and not no_checkpoints:
                torch.save(model.state_dict(), 'deeplab.pth')
                s3 = boto3.client('s3')
                s3.upload_file('deeplab.pth', bucket_name,
                               '{}/{}/deeplab_checkpoint_{}.pth'.format(s3_prefix, arg_hash, i))
                del s3

# Evaluation
if True:
    def evaluation_readahead_thread(raster_filename: str, mask_filename: str, max_eval_windows: int, window_size: int, batch_size: int, bands: List[int], label_mappings: Dict[int, int], label_nd: Union[int, float], image_nd: Union[None, Union[int, float]]):
        """The code for the evaluation data read-ahead thread

        Arguments:
            raster_filename {str} -- The name of the imagery file
            mask_filename {str} -- The name of the label file
            max_eval_windows {int} -- The maximum number of evaluation windows
            window_size {int} -- The window size
            batch_size {int} -- The batch size
            bands {List[int]} -- The imagery bands to use
            label_mappings {Dict[int, int]} -- The mapping between native and internal classes
            label_nd {Union[int, float]} -- The label nodata
            image_nd {Union[None, Union[int, float]]} -- The image nodata
        """
        global EVALUATION_BATCHES
        global EVALUATION_MUTEX

        with rio.open(raster_filename) as raster_ds, rio.open(mask_filename) as mask_ds:
            xys = []
            for x in range(0, raster_ds.width//window_size):
                for y in range(0, raster_ds.height//window_size):
                    if ((x + y) % 7 == 0):
                        xy = (x, y)
                        xys.append(xy)
            xys = xys[0:max_eval_windows]

            for xy in xys:
                x, y = xy
                time.sleep(0.01)
                with EVALUATION_MUTEX:
                    if len(EVALUATION_BATCHES) < min(max_eval_windows, 100):
                        bands_plus = bands + [0]
                        plan = []
                        window = rio.windows.Window(
                            x * window_size, y * window_size,
                            window_size, window_size)
                        for band in bands_plus:
                            args = (window, band, raster_ds, mask_ds)
                            plan.append(args)
                        batch = execute_plan(raster_ds, mask_ds, len(bands), plan,
                                             label_mappings, label_nd, image_nd)
                        EVALUATION_BATCHES = [batch] + EVALUATION_BATCHES
            with EVALUATION_MUTEX:
                EVALUATION_BATCHES = [(None, None)] + EVALUATION_BATCHES

    def evaluate(model: torch.nn.Module, raster_ds: rio.io.DatasetReader, label_ds: rio.io.DatasetReader, bands: List[int], label_count: int, window_size: int, device: torch.device, label_nd: Union[int, float], img_nd: Union[None, Union[int, float]], label_map: Dict[int, int], bucket_name: str, s3_prefix: str, arg_hash: str, max_eval_windows: int, batch_size: str):
        """Evaluate the performance of the model given the various data.  Results are stored in S3.

        Arguments:
            model {torch.nn.Module} -- The model to evaluate
            raster_ds {rio.io.DatasetReader} -- The imagery dataset
            label_ds {rio.io.DatasetReader} -- The label dataset
            bands {List[int]} -- The imagery bands to use
            label_count {int} -- The number of classes
            window_size {int} -- The window size
            device {torch.device} -- The device to use for evaluation
            label_nd {Union[int, float]} -- The label nodata
            img_nd {Union[None, Union[int, float]]} -- The imagery nodata
            label_map {Dict[int, int]} -- The mapping between native and internal classes
            bucket_name {str} -- The bucket name
            s3_prefix {str} -- The S3 prefix
            arg_hash {str} -- The hashed arguments
            max_eval_windows {int} -- The maximum number of evaluation windows to consider
            batch_size {str} -- The batch size
        """
        model.eval()
        with torch.no_grad():
            tps = [0.0 for x in range(label_count)]
            fps = [0.0 for x in range(label_count)]
            fns = [0.0 for x in range(label_count)]
            tns = [0.0 for x in range(label_count)]

            for i in range(max_eval_windows):

                while True:
                    with TRAINING_MUTEX:
                        if len(TRAINING_BATCHES) > 0:
                            batch = TRAINING_BATCHES.pop()
                            break
                        else:
                            time.sleep(1)

                if batch[0] is None or batch[1] is None:
                    break

                out = model(batch[0].to(device))
                if isinstance(out, dict):
                    out = out['out']
                out = out.data.cpu().numpy()
                out = np.apply_along_axis(np.argmax, 1, out)

                labels = batch[1].cpu().numpy()
                del batch

                # Make sure output reflects don't care values
                if label_nd is not None:
                    dont_care = (labels == label_nd)
                else:
                    dont_care = np.zeros(labels.shape)
                out = out + label_count*dont_care

                for j in range(label_count):
                    tps[j] = tps[j] + ((out == j)*(labels == j)).sum()
                    fps[j] = fps[j] + ((out == j)*(labels != j)).sum()
                    fns[j] = fns[j] + ((out != j)*(labels == j)).sum()
                    tns[j] = tns[j] + ((out != j)*(labels != j)).sum()

                with WATCHDOG_MUTEX:
                    global WATCHDOG_TIME
                    WATCHDOG_TIME = time.time()

        print('True Positives  {}'.format(tps))
        print('False Positives {}'.format(fps))
        print('False Negatives {}'.format(fns))
        print('True Negatives  {}'.format(tns))

        recalls = []
        precisions = []
        for j in range(label_count):
            recall = tps[j] / (tps[j] + fns[j])
            recalls.append(recall)
            precision = tps[j] / (tps[j] + fps[j])
            precisions.append(precision)

        print('Recalls    {}'.format(recalls))
        print('Precisions {}'.format(precisions))

        f1s = []
        for j in range(label_count):
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
            evaluations.write('Means:               {}\n'.format(MEANS))
            evaluations.write('Standard Deviations: {}\n'.format(STDS))

        s3 = boto3.client('s3')
        s3.upload_file('/tmp/evaluations.txt', bucket_name,
                       '{}/{}/evaluations.txt'.format(s3_prefix, arg_hash))
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
        parser.add_argument('--architecture', required=True, help='The desired model architecture',
                            choices=['resnet18', 'resnet34', 'resnet101', 'stock'])
        parser.add_argument('--backend', help="Don't use this flag unless you know what you're doing: CPU is far slower than CUDA.",
                            choices=['cpu', 'cuda'], default='cuda')
        parser.add_argument('--bands', required=True,
                            help='list of bands to train on (1 indexed)', nargs='+', type=int)
        parser.add_argument('--batch-size', default=16, type=int)
        parser.add_argument(
            '--disable-eval', help='Disable evaluation after training', action='store_true')
        parser.add_argument('--epochs1', default=5, type=int)
        parser.add_argument('--epochs2', default=5, type=int)
        parser.add_argument('--epochs3', default=5, type=int)
        parser.add_argument('--epochs4', default=15, type=int)
        parser.add_argument(
            '--image-nd', help='image value to ignore - must be on the first band', default=None, type=float)
        parser.add_argument(
            '--input-stride', help='consult this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md', default=2, type=int)
        parser.add_argument('--label-img', required=True,
                            help='labels to train')
        parser.add_argument('--label-map', help='comma separated list of mappings to apply to training labels',
                            action=StoreDictKeyPair, default=None)
        parser.add_argument(
            '--label-nd', help='label to ignore', default=None, type=int)
        parser.add_argument(
            '--learning-rate1', help='float (probably between 10^-6 and 1) to tune SGD (see https://arxiv.org/abs/1206.5533)', default=0.01, type=float)
        parser.add_argument(
            '--learning-rate2', help='float (probably between 10^-6 and 1) to tune SGD (see https://arxiv.org/abs/1206.5533)', default=0.001, type=float)
        parser.add_argument(
            '--learning-rate3', help='float (probably between 10^-6 and 1) to tune SGD (see https://arxiv.org/abs/1206.5533)', default=0.01, type=float)
        parser.add_argument(
            '--learning-rate4', help='float (probably between 10^-6 and 1) to tune SGD (see https://arxiv.org/abs/1206.5533)', default=0.001, type=float)
        parser.add_argument('--max-epoch-size', default=sys.maxsize, type=int)
        parser.add_argument(
            '--max-eval-windows', help='The maximum number of windows that will be used for evaluation', default=sys.maxsize, type=int)
        parser.add_argument('--max-sample-windows', default=0, type=int)
        parser.add_argument('--random-seed', default=33, type=int)
        parser.add_argument('--s3-bucket', required=True,
                            help='prefix to apply when saving models to s3')
        parser.add_argument('--s3-prefix', required=True,
                            help='prefix to apply when saving models to s3')
        parser.add_argument(
            '--start-from', help='The saved model to start the fourth phase from')
        parser.add_argument('--training-img', required=True,
                            help="the input you're training to produce labels for")
        parser.add_argument(
            '--watchdog-seconds', help='The number of seconds that can pass without activity before the program is terminated (0 to disable)', default=0, type=int)
        parser.add_argument('--weights', help='label to ignore',
                            nargs='+', required=True, type=float)
        parser.add_argument('--whole-image-statistics', action='store_true')
        parser.add_argument('--window-size', default=32, type=int)
        return parser

# Architectures
if True:
    class DeepLabResnet18(torch.nn.Module):
        def __init__(self, band_count, input_stride, class_count):
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
    del hashed_args.watchdog_seconds
    arg_hash = hash_string(str(hashed_args))
    print('provided args: {}'.format(hashed_args))
    print('hash: {}'.format(arg_hash))

    np.random.seed(seed=args.random_seed)

    # ---------------------------------
    print('DOWNLOADING DATA')

    if not os.path.exists('/tmp/mul.tif'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.training_img)
        print('training image bucket and prefix: {}, {}'.format(bucket, prefix))
        s3.download_file(bucket, prefix, '/tmp/mul.tif')
        del s3
    if not os.path.exists('/tmp/mask.tif'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.label_img)
        print('training labels bucket and prefix: {}, {}'.format(bucket, prefix))
        s3.download_file(bucket, prefix, '/tmp/mask.tif')
        del s3

    # ---------------------------------
    print('PRE-COMPUTING')

    if args.max_sample_windows == -1:
        with rio.open('/tmp/mul.tif') as raster_ds:
            for i in raster_ds.indexes:
                MEANS.append(0)
                STDS.append(1)
    elif args.max_sample_windows == 0:
        with rio.open('/tmp/mul.tif') as raster_ds:
            for dtype in raster_ds.dtypes:
                if dtype == 'uint8':
                    MEANS.append(0)
                    STDS.append(2**8 - 1)
                elif dtype == 'uint16':
                    MEANS.append(0)
                    STDS.append(2**16 - 1)
                elif dtype == 'float32':
                    MEANS.append(0)
                    STDS.append(1)
                elif dtype == 'float64':
                    MEANS.append(0)
                    STDS.append(1)
                else:
                    raise Exception
    elif args.whole_image_statistics:
        with rio.open('/tmp/mul.tif') as raster_ds:
            for i in range(0, len(raster_ds.indexes)):
                a = raster_ds.read(i+1).flatten()
                a = np.extract(a != args.image_nd, a)
                MEANS.append(a.mean())
                STDS.append(a.std())
            del a
    else:
        with rio.open('/tmp/mul.tif') as raster_ds:
            def sample():
                return get_random_sample(raster_ds, 224, args.image_nd)
            ws = [sample() for i in range(0, args.max_sample_windows)]
        for i in range(0, len(raster_ds.indexes)):
            a = np.concatenate([w[:, i] for w in ws])
            MEANS.append(a.mean())
            STDS.append(a.std())
        del a
        del sample
        del ws

    print('Means:               {}'.format(MEANS))
    print('Standard Deviations: {}'.format(STDS))

    # ---------------------------------
    print('RECORDING RUN IN BUCKET')

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
                    current_epoch = checkpoint_epoch
                    current_pth = pth
    elif args.start_from is not None:
        complete_thru = 4
        current_epoch = 0
        current_pth = args.start_from

    device = torch.device(args.backend)

    with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds:
        width = raster_ds.width
        height = raster_ds.height
        if (height != mask_ds.height) or (width != mask_ds.width):
            print('width', width, mask_ds.width)
            print('height', height, mask_ds.height)
            print('PROBLEM WITH DIMENSIONS')
            sys.exit()

    if args.label_nd is None:
        args.label_nd = len(args.weights)
        print('\t WARNING: LABEL NODATA NOT SET, SETTING TO {}'.format(args.label_nd))

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

    batches_per_epoch = min(args.max_epoch_size, int((width * height * 6.0) /
                                                     (args.window_size * args.window_size * 7.0 * args.batch_size)))

    print('\t STEPS PER EPOCH={}'.format(batches_per_epoch))
    obj = torch.nn.CrossEntropyLoss(
        ignore_index=args.label_nd,
        weight=torch.FloatTensor(args.weights).to(device)
    ).to(device)

    # ---------------------------------
    if args.watchdog_seconds > 0:
        print('STARTING WATCHDOG')
        t = threading.Thread(target=watchdog_thread,
                             args=(args.watchdog_seconds,))
        t.daemon = True
        t.start()
    else:
        print('NOT STARTING WATCHDOG')

    # ---------------------------------
    print('STARTING TRAINING READ-AHEAD THREAD')
    t = threading.Thread(target=training_readahead_thread, args=('/tmp/mul.tif', '/tmp/mask.tif',
                                                                 args.window_size, args.batch_size, args.bands,
                                                                 batches_per_epoch,
                                                                 args.label_map, args.label_nd, args.image_nd))
    t.daemon = True
    t.start()

    # ---------------------------------
    print('STARTING EVALUATION READ-AHEAD THREAD')
    t = threading.Thread(target=evaluation_readahead_thread, args=('/tmp/mul.tif', '/tmp/mask.tif',
                                                                   args.max_eval_windows,
                                                                   args.window_size, args.batch_size, args.bands,
                                                                   args.label_map, args.label_nd, args.image_nd))
    t.daemon = True
    t.start()

    # ---------------------------------
    print('COMPUTING')

    if complete_thru == -1:
        deeplab = make_model(
            len(args.bands),
            input_stride=args.input_stride,
            class_count=len(args.weights)
        ).to(device)

    np.random.seed(seed=(args.random_seed + 1))
    if complete_thru == 0:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'deeplab.pth')
        deeplab = make_model(
            len(args.bands),
            input_stride=args.input_stride,
            class_count=len(args.weights)
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
        opt = torch.optim.SGD(ps, lr=args.learning_rate1, momentum=0.9)

        with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds:
            train(deeplab, opt, obj, batches_per_epoch, args.epochs1, args.batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.image_nd, args.s3_bucket, args.s3_prefix, arg_hash)

        print('\t UPLOADING')

        torch.save(deeplab.state_dict(), 'deeplab.pth')
        s3 = boto3.client('s3')
        s3.upload_file('deeplab.pth', args.s3_bucket,
                       '{}/{}/deeplab_0.pth'.format(args.s3_prefix, arg_hash))
        del s3

    np.random.seed(seed=(args.random_seed + 2))
    if complete_thru == 1:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'deeplab.pth')
        deeplab = make_model(
            len(args.bands),
            input_stride=args.input_stride,
            class_count=len(args.weights)
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
        opt = torch.optim.SGD(ps, lr=args.learning_rate2, momentum=0.9)

        with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds:
            train(deeplab, opt, obj, batches_per_epoch, args.epochs2, args.batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.image_nd, args.s3_bucket, args.s3_prefix, arg_hash)

        print('\t UPLOADING')

        torch.save(deeplab.state_dict(), 'deeplab.pth')
        s3 = boto3.client('s3')
        s3.upload_file('deeplab.pth', args.s3_bucket,
                       '{}/{}/deeplab_1.pth'.format(args.s3_prefix, arg_hash))
        del s3

    np.random.seed(seed=(args.random_seed + 3))
    if complete_thru == 2:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'deeplab.pth')
        deeplab = make_model(
            len(args.bands),
            input_stride=args.input_stride,
            class_count=len(args.weights)
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
        opt = torch.optim.SGD(ps, lr=args.learning_rate3, momentum=0.9)

        with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds:
            train(deeplab, opt, obj, batches_per_epoch, args.epochs3, args.batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.image_nd, args.s3_bucket, args.s3_prefix, arg_hash)

        print('\t UPLOADING')

        torch.save(deeplab.state_dict(), 'deeplab.pth')
        s3 = boto3.client('s3')
        s3.upload_file('deeplab.pth', args.s3_bucket,
                       '{}/{}/deeplab_2.pth'.format(args.s3_prefix, arg_hash))
        del s3

    np.random.seed(seed=(args.random_seed + 4))
    if complete_thru == 3:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'deeplab.pth')
        deeplab = make_model(
            len(args.bands),
            input_stride=args.input_stride,
            class_count=len(args.weights)
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
        opt = torch.optim.SGD(ps, lr=args.learning_rate4, momentum=0.9)

        with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds:
            train(deeplab, opt, obj, batches_per_epoch, args.epochs4, args.batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.image_nd, args.s3_bucket, args.s3_prefix, arg_hash,
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
            len(args.bands),
            input_stride=args.input_stride,
            class_count=len(args.weights)
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
        opt = torch.optim.SGD(ps, lr=args.learning_rate4, momentum=0.9)

        with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds:
            train(deeplab, opt, obj, batches_per_epoch, args.epochs4, args.batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.image_nd, args.s3_bucket, args.s3_prefix, arg_hash,
                  no_checkpoints=False, starting_epoch=current_epoch)

    np.random.seed(seed=(args.random_seed + 6))
    if not args.disable_eval:
        print('\t EVALUATING')
        with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds:
            evaluate(deeplab, raster_ds, mask_ds, args.bands, len(args.weights), args.window_size,
                     device, args.label_nd, args.image_nd, args.label_map, args.s3_bucket,
                     args.s3_prefix, arg_hash, args.max_eval_windows, args.batch_size)

    exit(0)
