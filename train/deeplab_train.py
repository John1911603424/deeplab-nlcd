#!/usr/bin/env python3

import copy
import hashlib
import argparse
import os
import time
from urllib.parse import urlparse
from PIL import Image

import boto3
import numpy as np
import rasterio as rio
import torch
import torchvision

# Break s3uris into bucket and prefix
def parse_s3_url(url):
    parsed = urlparse(url, allow_fragments=False)
    return (parsed.netloc, parsed.path.lstrip('/'))

def generate_preview(s3_img_url, bands, img_nd, device, label_count, s3_bucket, s3_prefix, arg_hash):
    s3 = boto3.client('s3')
    bucket, prefix = parse_s3_url(s3_img_url)
    preview_filename = prefix.split('/')[-1]
    local_img = '/tmp/{}'.format(preview_filename)
    print("Downloading preview from {}".format(s3_img_url))
    s3.download_file(bucket, prefix, local_img)
    del s3

    with rio.open(local_img) as preview_ds:
        # Nodata
        nodata = preview_ds.read(1) == img_nd
        not_nodata = nodata == 0

        # Normalized float32 imagery bands
        data = []
        for band in bands:
            a = preview_ds.read(band, window=window)
            MEAN = a.flatten().mean()
            STD = a.flatten().std()

            a = np.array((a - MEAN) / STD, dtype=np.float32)
            a = a * not_nodata
            data.append(a)
        data = np.stack(data, axis=0)

    deeplab = torch.load('deeplab.pth').to(device)

    batch = torch.unsqueeze(torch.from_numpy(data), dim=0).to(device)
    deeplab.eval()

    with torch.no_grad():
      out = deeplab(batch)['out'].data.cpu().numpy()[0]

    prediction = np.apply_along_axis(np.argmax, 0, out)

    # the expansion coefficient is the amount of spread needed to approach a max of 255
    expansion_coefficient = (255 / label_count)
    img = Image.fromarray(np.uint8((prediction * expansion_coefficient).floor))

    minv = out[1].min()
    maxv = out[1].max()
    spread = maxv - minv
    grey = np.uint8(0xff * ((out[1] + minv) / spread))
    img = Image.fromarray(grey)
    local_preview = '/tmp/preview_{}'.format(preview_filename)
    img.save(local_preview)

    s3 = boto3.client('s3')
    s3.upload_file(local_preview, s3_bucket,
                   '{}/{}/preview_{}'.format(s3_prefix, arg_hash, preview_filename))
    del s3


def get_eval_window(raster_ds, mask_ds, bands, x, y, window_size, label_nd, img_nd, replacement_dict):
    window = rio.windows.Window(
        x * window_size, y * window_size,
        window_size, window_size)

    # Labels
    labels = mask_ds.read(1, window=window)
    labels = numpy_replace(labels, replacement_dict)

    if label_nd is not None:
        label_nds = labels == label_nd
    else:
        label_nds = np.zeros(labels.shape)

    # We assume here that the ND value will be on the first band
    if img_nd is not None:
        img_nds = raster_ds.read(1, window=window) == img_nd
    else:
        img_nds = np.zeros(labels.shape)

    # nodata mask for regions without labels
    nodata = (img_nds + label_nds) > 0
    not_nodata = (nodata == 0)

    # Toss out nodata labels
    labels = labels * not_nodata

    # Normalized float32 imagery bands
    data = []
    for band in bands:
        a = raster_ds.read(band, window=window)
        a = np.array((a - MEANS[band-1]) / STDS[band-1], dtype=np.float32)
        a = a * not_nodata
        data.append(a)
    data = np.stack(data, axis=0)

    return (data, labels)

def get_eval_batch(raster_ds, label_ds, bands, xys, window_size, label_nd, img_nd, replacement_dict, device):
    data = []
    labels = []
    for x, y in xys:
        d, l = get_eval_window(raster_ds, label_ds, bands, x, y, window_size, label_nd, img_nd, replacement_dict)
        data.append(d)
        labels.append(l)

    data = np.stack(data, axis=0)
    data = torch.from_numpy(data).to(device)
    labels = np.array(np.stack(labels, axis=0), dtype=np.long)
    labels = torch.from_numpy(labels).to(device)
    return (data, labels)


def evaluate(raster_ds, label_ds, bands, label_count, window_size, label_nd, img_nd, replacement_dict, device, bucket_name, s3_prefix, arg_hash):

    deeplab.eval()
    batch_size = 64
    tps = [0.0 for x in range(label_count)]
    fps = [0.0 for x in range(label_count)]
    fns = [0.0 for x in range(label_count)]
    preds = []
    ground_truth = []

    with torch.no_grad():
        width = raster_ds.width
        height = raster_ds.height

        xys = []
        for x in range(0, width//window_size):
            for y in range(0, height//window_size):
                if ((x + y) % 7 == 0):
                    xy = (x, y)
                    xys.append(xy)

        for xy in chunks(xys, batch_size):
            batch, labels = get_eval_batch(raster_ds, label_ds, bands, xy, window_size, label_nd, img_nd, replacement_dict, device)
            labels = labels.data.cpu().numpy()
            out = deeplab(batch)['out'].data.cpu().numpy()
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

            preds.append(out.flatten())
            ground_truth.append(labels.flatten())

    print('True Positives  {}'.format(tps))
    print('False Positives {}'.format(fps))
    print('False Negatives {}'.format(fns))

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

    preds = np.concatenate(preds).flatten()
    ground_truth = np.concatenate(ground_truth).flatten()
    preds = np.extract(ground_truth < 2, preds)
    ground_truth = np.extract(ground_truth < 2, ground_truth)
    np.save('/tmp/predictions.npy', preds, False)
    np.save('/tmp/ground_truth.npy', ground_truth, False)
    s3 = boto3.client('s3')
    s3.upload_file('/tmp/predictions.npy', bucket_name, '{}/{}/predictions.npy'.format(s3_prefix, arg_hash))
    s3.upload_file('/tmp/ground_truth.npy', bucket_name, '{}/{}/ground_truth.npy'.format(s3_prefix, arg_hash))
    del s3


# break list in chunks of equal size
def chunks(l, n):
    # https://chrisalbon.com/python/data_wrangling/break_list_into_chunks_of_equal_size/
    for i in range(0, len(l), n):
        yield l[i:i+n]


# Useful for generating an ID based on a set of parameters
# https://gist.github.com/nmalkin/e287f71788c57fd71bd0a7eec9345add
def hash_string(string):
    """
    Return a SHA-256 hash of the given string
    """
    return hashlib.sha256(string.encode('utf-8')).hexdigest()


# quickly replace contents of a np_arr with mappings provided by replacement_dict
# used primarily to map mask labels to the (assuming N training labels) 0 to N-1 categories expected
def numpy_replace(np_arr, replacement_dict):
    b = np.copy(np_arr)
    for k, v in replacement_dict.items():
        b[np_arr==k] = v
    return b


def get_random_training_window(raster_ds, label_ds, width, height, window_size, bands, label_mappings, label_nd, img_nd):
    x = 0
    y = 0
    while ((x + y) % 7) == 0:
        x = np.random.randint(0, width/window_size - 1)
        y = np.random.randint(0, height/window_size - 1)
    window = rio.windows.Window(
        x * window_size, y * window_size,
        window_size, window_size)

    # Labels
    labels = label_ds.read(1, window=window)
    labels = numpy_replace(labels, label_mappings)
    #print('unique: {}'.format(np.unique(labels)))

    if label_nd is not None:
        label_nds = labels == label_nd
    else:
        label_nds = np.zeros(labels.shape)

    # We assume here that the ND value will be on the first band
    if img_nd is not None:
        img_nds = raster_ds.read(1, window=window) == img_nd
    else:
        img_nds = np.zeros(labels.shape)

    # nodata mask for regions without labels
    nodata = (img_nds + label_nds) > 0
    not_nodata = (nodata == 0)

    # Normalized float32 imagery bands
    data = []
    for band in bands:
        a = raster_ds.read(band, window=window)
        a = np.array((a - MEANS[band-1]) / STDS[band-1], dtype=np.float32)
        a = a * not_nodata
        data.append(a)
    data = np.stack(data, axis=0)

    return (data, labels)


def get_random_training_batch(raster_ds, label_ds, width, height, window_size, batch_size, device, bands, label_mappings, label_nd, img_nd):
    data = []
    labels = []
    for i in range(0, batch_size):
        d, l = get_random_training_window(raster_ds, label_ds, width, height, window_size, bands, label_mappings, label_nd, img_nd)
        data.append(d)
        labels.append(l)

    data = np.stack(data, axis=0)
    data = torch.from_numpy(data).to(device)
    labels = np.array(np.stack(labels, axis=0), dtype=np.long)
    labels = torch.from_numpy(labels).to(device)
    return (data, labels)


def train(model, opt, obj,
          steps_per_epoch, epochs, batch_size,
          raster_ds, label_ds,
          width, height, window_size, device,
          bands, label_mapping, label_nd, img_nd,
          bucket_name, s3_prefix, arg_hash):
    model.train()
    current_time = time.time()
    for i in range(epochs):
        avg_loss = 0.0
        for j in range(steps_per_epoch):
            batch_tensor = get_random_training_batch(
                raster_ds, label_ds, width, height, window_size, batch_size, device,
                bands, label_mapping, label_nd, img_nd)
            opt.zero_grad()
            pred = model(batch_tensor[0])
            loss = 1.0*obj(pred.get('out'), batch_tensor[1]) \
                + 0.4*obj(pred.get('aux'), batch_tensor[1])
            loss.backward()
            opt.step()
            avg_loss = avg_loss + loss.item()
        avg_loss = avg_loss / steps_per_epoch
        last_time = current_time
        current_time = time.time()
        print('\t\t epoch={} time={} avg_loss={}'.format(
            i, current_time - last_time, avg_loss))
        if (epochs > 5) and (i > 0) and (i % 5 == 0) and bucket_name and s3_prefix:
            torch.save(model, 'deeplab.pth')
            s3 = boto3.client('s3')
            s3.upload_file('deeplab.pth', bucket_name,
                           '{}/{}/deeplab_checkpoint_{}.pth'.format(s3_output_prefix, arg_hash, i))
            del s3

# https://stackoverflow.com/questions/29986185/python-argparse-dict-arg
class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split(":")
            my_dict[int(k)] = int(v)
        setattr(namespace, self.dest, my_dict)

def training_cli_parser():
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
                        help='float (probably between 10^-6 and 1) to tune SGD', # https://arxiv.org/abs/1206.5533
                        default=os.environ.get('LEARNING_RATE_1', 0.01),
                        type=float)
    parser.add_argument('--epochs2',
                        help='',
                        default=os.environ.get('TRAINING_EPOCHS_2', 5),
                        type=int)
    parser.add_argument('--learning-rate2',
                        help='float (probably between 10^-6 and 1) to tune SGD', # https://arxiv.org/abs/1206.5533
                        default=os.environ.get('LEARNING_RATE_2', 0.001),
                        type=float)
    parser.add_argument('--epochs3',
                        help='',
                        default=os.environ.get('TRAINING_EPOCHS_2', 5),
                        type=int)
    parser.add_argument('--learning-rate3',
                        help='float (probably between 10^-6 and 1) to tune SGD', # https://arxiv.org/abs/1206.5533
                        default=os.environ.get('LEARNING_RATE_3', 0.01),
                        type=float)
    parser.add_argument('--epochs4',
                        help='',
                        default=os.environ.get('TRAINING_EPOCHS_2', 5),
                        type=int)
    parser.add_argument('--learning-rate4',
                        help='float (probably between 10^-6 and 1) to tune SGD', # https://arxiv.org/abs/1206.5533
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
    parser.add_argument('--output-dilation',
                        help='consult this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md',
                        default=os.environ.get('OUTPUT_DILATION', 1),
                        type=int)
    parser.add_argument('--output-kernel',
                        help='consult this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md',
                        default=os.environ.get('OUTPUT_KERNEL', 1),
                        type=int)
    parser.add_argument('--output-stride',
                        help='consult this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md',
                        default=os.environ.get('OUTPUT_STRIDE', 1),
                        type=int)
    parser.add_argument('--input-dilation',
                        help='consult this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md',
                        default=os.environ.get('INPUT_DILATION', 1),
                        type=int)
    parser.add_argument('--input-kernel',
                        help='consult this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md',
                        default=os.environ.get('INPUT_KERNEL', 7),
                        type=int)
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
    parser.add_argument('--inference-previews',
                        help='tif images against which inferences should be run to generate previews',
                        nargs='+')
    parser.add_argument('--window-size',
                        default=224,
                        type=int)
    parser.add_argument('--start-from')
    return parser


if __name__ == "__main__":

    parser = training_cli_parser()
    args = training_cli_parser().parse_args()
    hashed_args = copy.copy(args)
    del hashed_args.backend
    del hashed_args.inference_previews
    arg_hash = hash_string(str(hashed_args))
    print("provided args: {}".format(hashed_args))
    print("hash: {}".format(arg_hash))

    MEANS = []
    STDS = []

    if not os.path.exists('/tmp/mul.tif'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.training_img)
        print("training image bucket and prefix: {}, {}".format(bucket, prefix))
        s3.download_file(bucket, prefix, '/tmp/mul.tif')
        del s3
    if not os.path.exists('/tmp/mask.tif'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.label_img)
        print("training labels bucket and prefix: {}, {}".format(bucket, prefix))
        s3.download_file(bucket, prefix, '/tmp/mask.tif')
        del s3

    print('PRE-COMPUTING')

    with rio.open('/tmp/mul.tif') as raster_ds:
        for i in range(0, len(raster_ds.indexes)):
            a = raster_ds.read(i+1).flatten()
            MEANS.append(a.mean())
            STDS.append(a.std())
        del a

    print("Means: {}".format(MEANS))
    print("Standard Deviations: {}".format(STDS))

    print('INITIALIZING')

    np.random.seed(seed=args.random_seed)
    device = torch.device(args.backend)
    deeplab = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=True).to(device)
    print("label count: {}".format(len(args.weights)))
    last_class = deeplab.classifier[4] = torch.nn.Conv2d(
        256, len(args.weights), kernel_size=args.output_kernel, stride=args.output_stride, dilation=args.output_dilation).to(device)
    last_class_aux = deeplab.aux_classifier[4] = torch.nn.Conv2d(
        256, len(args.weights), kernel_size=args.output_kernel, stride=args.output_stride, dilation=args.output_dilation).to(device)
    input_filters = deeplab.backbone.conv1 = torch.nn.Conv2d(
        len(args.bands), 64, kernel_size=args.input_kernel, stride=args.input_stride, dilation=args.input_dilation, padding=(3, 3), bias=False).to(device)

    print('COMPUTING')

    # recording parameters in bcuket
    with open('/tmp/args.txt', 'w') as f:
        f.write(str(args))
    s3 = boto3.client('s3')
    s3.upload_file('/tmp/args.txt', args.s3_bucket,
                   '{}/{}/deeplab_training_args.txt'.format(args.s3_prefix, arg_hash))
    del s3


    with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds:

        width = raster_ds.width
        height = raster_ds.height

        if (height != mask_ds.height) or (width != mask_ds.width):
            print("width", width, mask_ds.width)
            print("height", height, mask_ds.height)
            print('PROBLEM WITH DIMENSIONS')
            sys.exit()

        batch_size = args.batch_size
        steps_per_epoch = int((width * height * 6.0) /
                              (args.window_size * args.window_size * 7.0 * batch_size))

        print('\t STEPS PER EPOCH={}'.format(steps_per_epoch))

        if args.label_nd is not None:
            obj = torch.nn.CrossEntropyLoss(
                ignore_index=args.label_nd,
                weight=torch.FloatTensor(args.weights).to(device)
            ).to(device)
        else:
            obj = torch.nn.CrossEntropyLoss(
                weight=torch.FloatTensor(args.weights).to(device)
            ).to(device)


        print('\t TRAINING FIRST AND LAST LAYERS')

        try:
            s3 = boto3.client('s3')
            s3.download_file(
                args.s3_bucket, '{}/{}/deeplab_0.pth'.format(args.s3_prefix, arg_hash), 'deeplab.pth')
            deeplab = torch.load('deeplab.pth').to(device)
            print('\t\t SUCCESSFULLY RESTARTED')
        except:
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

            train(deeplab, opt, obj, steps_per_epoch, args.epochs1, args.batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.img_nd, args.s3_bucket, args.s3_prefix, arg_hash)

            print('\t UPLOADING')

            torch.save(deeplab, 'deeplab.pth')
            s3 = boto3.client('s3')
            s3.upload_file('deeplab.pth', args.s3_bucket,
                           '{}/{}/deeplab_0.pth'.format(args.s3_prefix, arg_hash))
            del s3

        print('\t TRAINING FIRST AND LAST LAYERS AGAIN')

        try:
            s3 = boto3.client('s3')
            s3.download_file(
                args.s3_bucket, '{}/{}/deeplab_1.pth'.format(args.s3_prefix, arg_hash), 'deeplab.pth')
            deeplab = torch.load('deeplab.pth').to(device)
            del s3
            print('\t\t SUCCESSFULLY RESTARTED')
        except:
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
            train(deeplab, opt, obj, steps_per_epoch, args.epochs2, args.batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.img_nd, args.s3_bucket, args.s3_prefix, arg_hash)

            print('\t UPLOADING')

            torch.save(deeplab, 'deeplab.pth')
            s3 = boto3.client('s3')
            s3.upload_file('deeplab.pth', args.s3_bucket,
                           '{}/{}/deeplab_1.pth'.format(args.s3_prefix, arg_hash))
            del s3

        print('\t TRAINING ALL LAYERS')

        try:
            s3 = boto3.client('s3')
            s3.download_file(
                args.s3_bucket, '{}/{}/deeplab_2.pth'.format(args.s3_prefix, arg_hash), 'deeplab.pth')
            deeplab = torch.load('deeplab.pth').to(device)
            del s3
            print('\t\t SUCCESSFULLY RESTARTED')
        except:
            for p in deeplab.parameters():
                p.requires_grad = True

            ps = []
            for n, p in deeplab.named_parameters():
                if p.requires_grad == True:
                    ps.append(p)
                else:
                    p.grad = None
            opt = torch.optim.SGD(ps, lr=args.learning_rate3, momentum=0.9)

            train(deeplab, opt, obj, steps_per_epoch, args.epochs3, batch_size,
                  raster_ds, mask_ds, width, height, args.window_size, device,
                  args.bands, args.label_map, args.label_nd, args.img_nd, args.s3_bucket, args.s3_prefix, arg_hash)

            print('\t UPLOADING')

            torch.save(deeplab, 'deeplab.pth')
            s3 = boto3.client('s3')
            s3.upload_file('deeplab.pth', args.s3_bucket,
                           '{}/{}/deeplab_2.pth'.format(args.s3_prefix, arg_hash))
            del s3

        print('\t TRAINING ALL LAYERS AGAIN')

        if args.start_from:
            s3 = boto3.client('s3')
            s3.download_file(
                args.bucket, args.start_from, 'deeplab.pth')
            deeplab = torch.load('deeplab.pth').to(device)

        for p in deeplab.parameters():
            p.requires_grad = True

        ps = []
        for n, p in deeplab.named_parameters():
            if p.requires_grad == True:
                ps.append(p)
            else:
                p.grad = None

        opt = torch.optim.SGD(ps, lr=args.learning_rate4, momentum=0.9)

        train(deeplab, opt, obj, steps_per_epoch, args.epochs4, batch_size,
              raster_ds, mask_ds, width, height, args.window_size, device,
              args.bands, args.label_map, args.label_nd, args.img_nd, args.s3_bucket, args.s3_prefix, arg_hash)

        print('\t UPLOADING')

        torch.save(deeplab, 'deeplab.pth')
        s3 = boto3.client('s3')
        s3.upload_file('deeplab.pth', args.s3_bucket,
                       '{}/{}/deeplab.pth'.format(args.s3_prefix, arg_hash))
        del s3

        evaluate(raster_ds, mask_ds, args.bands, len(args.weights), args.window_size,
                 args.label_nd, args.img_nd, args.label_map, device, args.s3_bucket,
                 args.s3_prefix, arg_hash)

        for preview in args.inference_previews:
            print("generating preview: {}".format(preview))
            generate_preview(preview, args.bands, args.img_nd, device, len(args.weights), args.s3_bucket, args.s3_prefix, arg_hash)

        exit(0)

