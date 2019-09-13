#!/usr/bin/env python3

import argparse
import copy
import hashlib
import os
import re
import sys
import threading
import time
from multiprocessing import Pool
from urllib.parse import urlparse

import numpy as np
from PIL import Image

import boto3
import rasterio as rio
import torch
import torchvision

if os.environ.get('CURL_CA_BUNDLE') is None:
    os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

MEANS = []
STDS = []
RASTER_DS = None
LABEL_DS = None
MUTEX = threading.Lock()
WATCHDOG_TIMER = time.time()


def retry_read(rio_ds, band, window=None, retries=3):
    # retry failed reads (they appear to be transient)
    for i in range(retries):
        try:
            return rio_ds.read(band, window=window)
        except rio.errors.RasterioIOError:
            print('Read error for band {} at window {} on try {} of {}'.format(
                band, window, i+1, retries))
            continue


def parse_s3_url(url):
    # Break s3uris into bucket and prefix
    parsed = urlparse(url, allow_fragments=False)
    return (parsed.netloc, parsed.path.lstrip('/'))


def chunks(l, n):
    """
    Break list in chunks of equal size

    https://chrisalbon.com/python/data_wrangling/break_list_into_chunks_of_equal_size/
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]


def hash_string(string):
    """
    Return a SHA-256 hash of the given string

    Useful for generating an ID based on a set of parameters
    https://gist.github.com/nmalkin/e287f71788c57fd71bd0a7eec9345add
    """
    return hashlib.sha256(string.encode('utf-8')).hexdigest()


def numpy_replace(np_arr, replacement_dict):
    """
    Quickly replace contents of a np_arr with mappings provided by
    replacement_dict used primarily to map mask labels to the
    (assuming N training labels) 0 to N-1 categories expected.
    """
    b = np.copy(np_arr)
    for k, v in replacement_dict.items():
        b[np_arr == k] = v
    return b


def get_random_sample(raster_ds, width, height, window_size, bands, img_nd):
    x = np.random.randint(0, width/window_size - 1)
    y = np.random.randint(0, height/window_size - 1)
    window = rio.windows.Window(
        x * window_size, y * window_size,
        window_size, window_size)

    data = []
    for band in bands:
        a = retry_read(raster_ds, band, window=window)
        if img_nd is not None:
            a = np.extract(a != img_nd, a)
        a = a[~np.isnan(a)]
        data.append(a)
    data = np.stack(data, axis=1)

    return data


def get_matching_s3_keys(bucket, prefix='', suffix=''):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    """
    s3 = boto3.client('s3')
    kwargs = {'Bucket': bucket}

    # https://alexwlchan.net/2017/07/listing-s3-keys/

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


def get_window(arguments):
    window, band = arguments

    # Labels
    if band == 0:
        a = np.array(retry_read(LABEL_DS, 1, window=window), dtype=np.long)
    else:
        a = np.float32(retry_read(RASTER_DS, band, window=window))
        a = (a - MEANS[band-1]) / STDS[band-1]

    return a, band


def get_evaluation_batch(raster_ds, label_ds, bands, xys, window_size, label_nd, image_nd, label_mappings, device):

    global RASTER_DS
    global LABEL_DS

    RASTER_DS = raster_ds
    LABEL_DS = label_ds

    bands_plus = bands + [0]
    plan = []
    for xy in xys:
        x, y = xy
        window = rio.windows.Window(
            x * window_size, y * window_size,
            window_size, window_size)
        for band in bands_plus:
            args = (window, band)
            plan.append(args)

    data = []
    labels = []
    if disable_pmap:
        for d, i in map(get_window, plan):
            if i == 0:
                labels.append(d)
            else:
                data.append(d)
    else:
        with Pool(min(32, len(bands_plus))) as p:
            for d, i in p.map(get_window, plan):
                if i == 0:
                    labels.append(d)
                else:
                    data.append(d)

    RASTER_DS = None
    LABEL_DS = None

    raster_batch = []
    label_batch = []
    if image_nd is not None:
        image_nd = (image_nd - MEANS[0])/STDS[0]

    for raster, label in zip(chunks(data, len(bands)), labels):

        label = numpy_replace(label, label_mappings)
        if label_nd is not None:
            label_nds = (label == label_nd)
        else:
            label_nds = np.zeros(labels.shape)

        nan = np.isnan(raster[0])
        if image_nd is not None:
            image_nds = (raster[0] == image_nd) + nan
        else:
            image_nds = nan

        nodata = ((image_nds + label_nds) > 0)
        label[nodata == True] = label_nd
        for i in range(len(raster)):
            raster[i][nodata == True] = 0.0

        raster_batch.append(np.stack(raster, axis=0))
        label_batch.append(label)

    raster_batch = np.stack(raster_batch, axis=0)
    raster_batch = torch.from_numpy(raster_batch).to(device)
    label_batch = np.stack(label_batch, axis=0)
    label_batch = torch.from_numpy(label_batch).to(device)

    return (raster_batch, label_batch)


def evaluate(raster_ds, label_ds,
             bands, label_count, window_size, device,
             label_nd, img_nd, label_map,
             bucket_name, s3_prefix, arg_hash,
             max_eval_windows, batch_size):

    deeplab.eval()

    with torch.no_grad():
        tps = [0.0 for x in range(label_count)]
        fps = [0.0 for x in range(label_count)]
        fns = [0.0 for x in range(label_count)]
        tns = [0.0 for x in range(label_count)]
        preds = []
        ground_truth = []

        width = raster_ds.width
        height = raster_ds.height

        xys = []
        for x in range(0, width//window_size):
            for y in range(0, height//window_size):
                if ((x + y) % 7 == 0):
                    xy = (x, y)
                    xys.append(xy)
        xys = xys[0:max_eval_windows]

        for xy in chunks(xys, batch_size):
            batch, labels = get_evaluation_batch(
                raster_ds, label_ds, bands, xy, window_size, label_nd, img_nd, label_map, device)
            labels = labels.data.cpu().numpy()
            out = deeplab(batch)
            if isinstance(out, dict):
                out = out['out']
            out = out.data.cpu().numpy()
            out = np.apply_along_axis(np.argmax, 1, out)

            if label_nd is not None:
                dont_care = labels == label_nd
            else:
                dont_care = np.zeros(labels.shape)

            out = out + label_count*dont_care

            for i in range(label_count):
                tps[i] = tps[i] + ((out == i)*(labels == i)).sum()
                fps[i] = fps[i] + ((out == i)*(labels != i)).sum()
                fns[i] = fns[i] + ((out != i)*(labels == i)).sum()
                tns[i] = tns[i] + ((out != i)*(labels != i)).sum()

            preds.append(out.flatten())
            ground_truth.append(labels.flatten())

            with MUTEX:
                global WATCHDOG_TIMER
                WATCHDOG_TIMER = time.time()

    print('True Positives  {}'.format(tps))
    print('False Positives {}'.format(fps))
    print('False Negatives {}'.format(fns))
    print('True Negatives  {}'.format(tns))

    recalls = []
    precisions = []
    for i in range(label_count):
        recall = tps[i] / (tps[i] + fns[i])
        recalls.append(recall)
        precision = tps[i] / (tps[i] + fps[i])
        precisions.append(precision)

    print('Recalls    {}'.format(recalls))
    print('Precisions {}'.format(precisions))

    f1s = []
    for i in range(label_count):
        f1 = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
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

    preds = np.concatenate(preds).flatten()
    ground_truth = np.concatenate(ground_truth).flatten()
    preds = np.extract(ground_truth < 2, preds)
    ground_truth = np.extract(ground_truth < 2, ground_truth)
    np.save('/tmp/predictions.npy', preds, False)
    np.save('/tmp/ground_truth.npy', ground_truth, False)
    s3 = boto3.client('s3')
    s3.upload_file('/tmp/evaluations.txt', bucket_name,
                   '{}/{}/evaluations.txt'.format(s3_prefix, arg_hash))
    s3.upload_file('/tmp/predictions.npy', bucket_name,
                   '{}/{}/predictions.npy'.format(s3_prefix, arg_hash))
    s3.upload_file('/tmp/ground_truth.npy', bucket_name,
                   '{}/{}/ground_truth.npy'.format(s3_prefix, arg_hash))
    del s3


def get_random_training_batch(raster_ds, label_ds, width, height, window_size, batch_size, device, bands, label_mappings, label_nd, image_nd):

    global RASTER_DS
    global LABEL_DS

    RASTER_DS = raster_ds
    LABEL_DS = label_ds

    # Create a list of window, band pairs to read
    bands_plus = bands + [0]
    plan = []
    for batch_index in range(batch_size):
        x = 0
        y = 0
        while ((x + y) % 7) == 0:
            x = np.random.randint(0, width/window_size - 1)
            y = np.random.randint(0, height/window_size - 1)
        window = rio.windows.Window(
            x * window_size, y * window_size,
            window_size, window_size)
        for band in bands_plus:
            args = (window, band)
            plan.append(args)

    # Do all of the reads
    data = []
    labels = []
    if disable_pmap:
        for d, i in map(get_window, plan):
            if i == 0:
                labels.append(d)
            else:
                data.append(d)
    else:
        with Pool(min(32, len(bands_plus))) as p:
            for d, i in p.map(get_window, plan):
                if i == 0:
                    labels.append(d)
                else:
                    data.append(d)

    RASTER_DS = None
    LABEL_DS = None

    # NODATA processing
    raster_batch = []
    label_batch = []
    if image_nd is not None:
        image_nd = (image_nd - MEANS[0])/STDS[0]

    for raster, label in zip(chunks(data, len(bands)), labels):

        # NODATA from labels
        label = numpy_replace(label, label_mappings)
        if label_nd is not None:
            label_nds = (label == label_nd)
        else:
            label_nds = np.zeros(labels.shape)

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

    raster_batch = np.stack(raster_batch, axis=0)
    raster_batch = torch.from_numpy(raster_batch).to(device)
    label_batch = np.stack(label_batch, axis=0)
    label_batch = torch.from_numpy(label_batch).to(device)

    return (raster_batch, label_batch)


def train(model, opt, obj,
          steps_per_epoch, epochs, batch_size,
          raster_ds, label_ds,
          width, height, window_size, device,
          bands, label_mapping, label_nd, img_nd,
          bucket_name, s3_prefix, arg_hash,
          no_checkpoints=True, starting_epoch=0):

    model.train()

    current_time = time.time()

    for i in range(starting_epoch, epochs):
        avg_loss = 0.0

        for j in range(steps_per_epoch):
            batch_tensor = get_random_training_batch(
                raster_ds, label_ds, width, height, window_size, batch_size, device,
                bands, label_mapping, label_nd, img_nd)
            opt.zero_grad()
            pred = model(batch_tensor[0])
            if isinstance(pred, dict):
                loss = 1.0 * \
                    obj(pred.get('out'), batch_tensor[1]) + \
                    0.4*obj(pred.get('aux'), batch_tensor[1])
            else:
                loss = 1.0 * obj(pred, batch_tensor[1])
            loss.backward()
            opt.step()
            avg_loss = avg_loss + loss.item()

        avg_loss = avg_loss / steps_per_epoch

        last_time = current_time
        current_time = time.time()
        print('\t\t epoch={} time={} avg_loss={}'.format(
            i, current_time - last_time, avg_loss))

        with MUTEX:
            global WATCHDOG_TIMER
            WATCHDOG_TIMER = time.time()

        if (i > 0) and (i % 5 == 0) and bucket_name and s3_prefix and not no_checkpoints:
            torch.save(model.state_dict(), 'deeplab.pth')
            s3 = boto3.client('s3')
            s3.upload_file('deeplab.pth', bucket_name,
                           '{}/{}/deeplab_checkpoint_{}.pth'.format(s3_prefix, arg_hash, i))
            del s3


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(','):
            k, v = kv.split(':')
            my_dict[int(k)] = int(v)
        setattr(namespace, self.dest, my_dict)


def training_cli_parser():
    """
    https://stackoverflow.com/questions/29986185/python-argparse-dict-arg
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--bands',
                        help='list of bands to train (1 indexed)',
                        nargs='+',
                        default=os.environ.get('TRAINING_BANDS', [1, 2, 3]),
                        type=int)
    parser.add_argument('--epochs1',
                        help='',
                        default=os.environ.get('TRAINING_EPOCHS_1', 5),
                        type=int)
    parser.add_argument('--learning-rate1',
                        # https://arxiv.org/abs/1206.5533
                        help='float (probably between 10^-6 and 1) to tune SGD',
                        default=os.environ.get('LEARNING_RATE_1', 0.01),
                        type=float)
    parser.add_argument('--epochs2',
                        help='',
                        default=os.environ.get('TRAINING_EPOCHS_2', 5),
                        type=int)
    parser.add_argument('--learning-rate2',
                        # https://arxiv.org/abs/1206.5533
                        help='float (probably between 10^-6 and 1) to tune SGD',
                        default=os.environ.get('LEARNING_RATE_2', 0.001),
                        type=float)
    parser.add_argument('--epochs3',
                        help='',
                        default=os.environ.get('TRAINING_EPOCHS_3', 5),
                        type=int)
    parser.add_argument('--learning-rate3',
                        # https://arxiv.org/abs/1206.5533
                        help='float (probably between 10^-6 and 1) to tune SGD',
                        default=os.environ.get('LEARNING_RATE_3', 0.01),
                        type=float)
    parser.add_argument('--epochs4',
                        help='',
                        default=os.environ.get('TRAINING_EPOCHS_4', 15),
                        type=int)
    parser.add_argument('--learning-rate4',
                        # https://arxiv.org/abs/1206.5533
                        help='float (probably between 10^-6 and 1) to tune SGD',
                        default=os.environ.get('LEARNING_RATE_4', 0.001),
                        type=float)
    parser.add_argument('--training-img',
                        required=True,
                        help="the input you're training to produce labels for")
    parser.add_argument('--label-img',
                        required=True,
                        help='labels to train')
    parser.add_argument('--label-map',
                        help='comma separated list of mappings to apply to training labels',
                        action=StoreDictKeyPair,
                        default=os.environ.get('LABEL_MAPPING', None))
    parser.add_argument('--label-nd',
                        help='label to ignore',
                        default=os.environ.get('TRAINING_LABEL_ND', None),
                        type=int)
    parser.add_argument('--img-nd',
                        help='image value to ignore - must be on the first band',
                        default=os.environ.get('TRAINING_IMAGE_ND', None),
                        type=float)
    parser.add_argument('--weights',
                        help='label to ignore',
                        nargs='+',
                        required=True,
                        type=float)
    parser.add_argument('--input-stride',
                        help='consult this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md',
                        default=os.environ.get('INPUT_STRIDE', 2),
                        type=int)
    parser.add_argument('--random-seed',
                        default=33,
                        type=int)
    parser.add_argument('--batch-size',
                        default=16,
                        type=int)
    parser.add_argument('--backend',
                        help="Don't use this flag unless you know what you're doing: CPU is far slower than CUDA.",
                        choices=['cpu', 'cuda'],
                        default='cuda')
    parser.add_argument('--s3-bucket',
                        required=True,
                        help='prefix to apply when saving models and diagnostic images to s3')
    parser.add_argument('--s3-prefix',
                        required=True,
                        help='prefix to apply when saving models and diagnostic images to s3')
    parser.add_argument('--window-size',
                        default=32,
                        type=int)
    parser.add_argument('--max-epoch-size',
                        default=sys.maxsize,
                        type=int)
    parser.add_argument('--disable-eval',
                        help='Disable evaluation after training',
                        action='store_true')
    parser.add_argument('--disable-pmap',
                        action='store_true')
    parser.add_argument('--max-eval-windows',
                        help='The maximum number of windows that will be used for evaluation',
                        default=sys.maxsize,
                        type=int)
    parser.add_argument('--start-from',
                        help='The saved model to start the fourth phase from')
    parser.add_argument('--watchdog-seconds',
                        help='The number of seconds that can pass without activity before the program is terminated (0 to disable)',
                        default=0,
                        type=int)
    return parser


def watchdog_thread(seconds):
    while True:
        time.sleep(max(seconds//2, 1))
        with MUTEX:
            gap = time.time() - WATCHDOG_TIMER
        if gap > seconds:
            print('TERMINATING DUE TO INACTIVITY {} > {}\n'.format(
                gap, seconds), file=sys.stderr)
            os._exit(-1)


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


def make_model(band_count, input_stride=1, class_count=2):
    deeplab = DeepLabResnet34(band_count, input_stride, class_count)
    return deeplab


if __name__ == '__main__':

    parser = training_cli_parser()
    args = training_cli_parser().parse_args()
    hashed_args = copy.copy(args)
    hashed_args.script = sys.argv[0]
    del hashed_args.backend
    del hashed_args.disable_eval
    del hashed_args.disable_pmap
    del hashed_args.max_eval_windows
    del hashed_args.watchdog_seconds
    arg_hash = hash_string(str(hashed_args))
    print('provided args: {}'.format(hashed_args))
    print('hash: {}'.format(arg_hash))

    np.random.seed(seed=args.random_seed)

    disable_pmap = args.disable_pmap

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

    with rio.open('/tmp/mul.tif') as raster_ds:
        def sample():
            return get_random_sample(raster_ds, raster_ds.width, raster_ds.height,
                                     224, raster_ds.indexes,
                                     args.img_nd)
        ws = [sample() for i in range(0, 133)]
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
    print('RECORDING RUN')

    # recording parameters in bucket
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

    batch_size = args.batch_size
    if batch_size < 2:
        batch_size = 2
        print('\t WARNING: BATCH SIZE MUST BE AT LEAST 2, SETTING TO 2')

    steps_per_epoch = min(args.max_epoch_size, int((width * height * 6.0) /
                                                   (args.window_size * args.window_size * 7.0 * batch_size)))

    print('\t STEPS PER EPOCH={}'.format(steps_per_epoch))
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
            train(deeplab, opt, obj, steps_per_epoch, args.epochs1, batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.img_nd, args.s3_bucket, args.s3_prefix, arg_hash)

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
            train(deeplab, opt, obj, steps_per_epoch, args.epochs2, batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.img_nd, args.s3_bucket, args.s3_prefix, arg_hash)

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
            train(deeplab, opt, obj, steps_per_epoch, args.epochs3, batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.img_nd, args.s3_bucket, args.s3_prefix, arg_hash)

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
            train(deeplab, opt, obj, steps_per_epoch, args.epochs4, batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.img_nd, args.s3_bucket, args.s3_prefix, arg_hash,
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
            train(deeplab, opt, obj, steps_per_epoch, args.epochs4, batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.img_nd, args.s3_bucket, args.s3_prefix, arg_hash,
                  no_checkpoints=False, starting_epoch=current_epoch)

        print('\t UPLOADING')

        torch.save(deeplab.state_dict(), 'deeplab.pth')
        s3 = boto3.client('s3')
        s3.upload_file('deeplab.pth', args.s3_bucket,
                       '{}/{}/deeplab.pth'.format(args.s3_prefix, arg_hash))
        del s3

    np.random.seed(seed=(args.random_seed + 6))
    if not args.disable_eval:
        print('\t EVALUATING')
        with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds:
            evaluate(raster_ds, mask_ds, args.bands, len(args.weights), args.window_size,
                     device, args.label_nd, args.img_nd, args.label_map, args.s3_bucket,
                     args.s3_prefix, arg_hash, args.max_eval_windows, args.batch_size)

    exit(0)
