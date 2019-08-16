#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import sys

import boto3
import numpy as np
import rasterio as rio
import torch
import torchvision

os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

WINDOW_SIZE = 224
UUID = 'xxx'
MEANS = []
STDS = []
CHANNELS = 3


def numpy_replace(np_arr, replacement_dict):
    b = np.copy(np_arr)
    for k, v in replacement_dict.items():
        b[np_arr==k] = v
    return b

def get_random_training_window(raster_ds, label_ds, width, height):
    x = 0
    y = 0
    while ((x + y) % 7) == 0:
        x = np.random.randint(0, width/WINDOW_SIZE - 1)
        y = np.random.randint(0, height/WINDOW_SIZE - 1)
    window = rio.windows.Window(
        x * WINDOW_SIZE, y * WINDOW_SIZE,
        WINDOW_SIZE, WINDOW_SIZE)

    if CHANNELS == 3:
        bands = [2, 1, 0]
    else:
        bands = raster_ds.indexes[0:CHANNELS]

    replacement_dict = { 0:0, 11:1, 12:2, 21:3, 22:3, 23:3, 24:3, 31:4, 41:5, 42:5, 43:5, 51:6, 52:6, 71:7, 72:7, 73:7, 74:7, 81:8, 82:8, 90:9, 95:9 }
    # Labels
    labels = label_ds.read(1, window=window)
    labels = numpy_replace(labels, replacement_dict)

    # nodata mask for regions without labels
    nodata = labels == (2**8)-1
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


def get_random_training_batch(raster_ds, label_ds, width, height, batch_size, device):
    data = []
    labels = []
    for i in range(0, batch_size):
        d, l = get_random_training_window(raster_ds, label_ds, width, height)
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
          width, height, device,
          bucket_name=None, dataset_name=None):
    model.train()
    current_time = time.time()
    for i in range(epochs):
        avg_loss = 0.0
        for j in range(steps_per_epoch):
            batch_tensor = get_random_training_batch(
                raster_ds, label_ds, width, height, batch_size, device)
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
        if (epochs > 5) and (i > 0) and (i % 5 == 0) and bucket_name and dataset_name:
            torch.save(model, 'deeplab.pth')
            s3 = boto3.client('s3')
            s3.upload_file('deeplab.pth', bucket_name,
                           '{}/deeplab_{}_checkpoint_{}.pth'.format(dataset_name, UUID, i))
            del s3


if __name__ == "__main__":

    print('ARGUMENTS={}'.format(sys.argv))

    CHANNELS = int(sys.argv[1])
    UUID = sys.argv[2]
    epochs = []
    for i in range(0, 4):
        epochs.append(int(sys.argv[3 + i]))
    bucket_name = sys.argv[7]
    img_tif = sys.argv[8]
    label_tif = sys.argv[9]
    dataset_name = sys.argv[10]
    if len(sys.argv) > 11:
        weight = float(sys.argv[11])
    else:
        weight = 1.0
    if len(sys.argv) > 12:
        start_from = sys.argv[12]
    else:
        start_from = None

    print('DATA')

    if not os.path.exists('/tmp/mul.tif'):
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, img_tif, '/tmp/mul.tif')
        del s3
    if not os.path.exists('/tmp/mask.tif'):
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, label_tif, '/tmp/mask.tif')
        del s3

    print('PRE-COMPUTING')

    with rio.open('/tmp/mul.tif') as raster_ds:
        for i in range(0, len(raster_ds.indexes)):
            a = raster_ds.read(i+1).flatten()
            MEANS.append(a.mean())
            STDS.append(a.std())
        del a
    print(MEANS)
    print(STDS)

    print('INITIALIZING')

    np.random.seed(seed=33)
    device = torch.device('cuda')
    deeplab = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=True).to(device)
    last_class = deeplab.classifier[4] = torch.nn.Conv2d(
        256, 10, kernel_size=(1, 1), stride=(1, 1)).to(device)
    last_class_aux = deeplab.aux_classifier[4] = torch.nn.Conv2d(
        256, 10, kernel_size=(1, 1), stride=(1, 1)).to(device)
    input_filters = deeplab.backbone.conv1 = torch.nn.Conv2d(CHANNELS, 64, kernel_size=(
        7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)

    print('COMPUTING')

    with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds:

        width = raster_ds.width
        height = raster_ds.height

        if (height != mask_ds.height) or (width != mask_ds.width):
            print("width", width, mask_ds.width)
            print("height", height, mask_ds.height)
            print('PROBLEM WITH DIMENSIONS')
            sys.exit()

        batch_size = 16
        steps_per_epoch = int((width * height * 6.0) /
                              (WINDOW_SIZE * WINDOW_SIZE * 7.0 * batch_size))
        print('\t STEPS PER EPOCH={}'.format(steps_per_epoch))

        obj = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

        print('\t TRAINING FIRST AND LAST LAYERS')

        try:
            s3 = boto3.client('s3')
            s3.download_file(
                bucket_name, '{}/deeplab_{}_0.pth'.format(dataset_name, UUID), 'deeplab.pth')
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
            opt = torch.optim.SGD(ps, lr=0.01, momentum=0.9)

            train(deeplab, opt, obj, steps_per_epoch, epochs[0], batch_size,
                  raster_ds, mask_ds, width, height, device)

            print('\t UPLOADING')

            torch.save(deeplab, 'deeplab.pth')
            s3 = boto3.client('s3')
            s3.upload_file('deeplab.pth', bucket_name,
                           '{}/deeplab_{}_0.pth'.format(dataset_name, UUID))
            del s3

        print('\t TRAINING FIRST AND LAST LAYERS AGAIN')

        try:
            s3 = boto3.client('s3')
            s3.download_file(
                bucket_name, '{}/deeplab_{}_1.pth'.format(dataset_name, UUID), 'deeplab.pth')
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
            opt = torch.optim.SGD(ps, lr=0.001, momentum=0.9)
            train(deeplab, opt, obj, steps_per_epoch, epochs[1], batch_size,
                  raster_ds, mask_ds, width, height, device)

            print('\t UPLOADING')

            torch.save(deeplab, 'deeplab.pth')
            s3 = boto3.client('s3')
            s3.upload_file('deeplab.pth', bucket_name,
                           '{}/deeplab_{}_1.pth'.format(dataset_name, UUID))
            del s3

        print('\t TRAINING ALL LAYERS')

        try:
            s3 = boto3.client('s3')
            s3.download_file(
                bucket_name, '{}/deeplab_{}_2.pth'.format(dataset_name, UUID), 'deeplab.pth')
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
            opt = torch.optim.SGD(ps, lr=0.01, momentum=0.9)

            train(deeplab, opt, obj, steps_per_epoch, epochs[2], batch_size,
                  raster_ds, mask_ds, width, height, device)

            print('\t UPLOADING')

            torch.save(deeplab, 'deeplab.pth')
            s3 = boto3.client('s3')
            s3.upload_file('deeplab.pth', bucket_name,
                           '{}/deeplab_{}_2.pth'.format(dataset_name, UUID))
            del s3

        print('\t TRAINING ALL LAYERS AGAIN')

        if start_from:
            s3 = boto3.client('s3')
            s3.download_file(
                bucket_name, start_from, 'deeplab.pth')
            deeplab = torch.load('deeplab.pth').to(device)

        for p in deeplab.parameters():
            p.requires_grad = True

        ps = []
        for n, p in deeplab.named_parameters():
            if p.requires_grad == True:
                ps.append(p)
            else:
                p.grad = None

        opt = torch.optim.SGD(ps, lr=0.001, momentum=0.9)

        train(deeplab, opt, obj, steps_per_epoch, epochs[3], batch_size,
              raster_ds, mask_ds, width, height, device, bucket_name, dataset_name)

        print('\t UPLOADING')

        torch.save(deeplab, 'deeplab.pth')
        s3 = boto3.client('s3')
        s3.upload_file('deeplab.pth', bucket_name,
                       '{}/deeplab_{}.pth'.format(dataset_name, UUID))
        del s3

# ./download_run.sh s3://geotrellis-test/courage-services/train_simplified_nlcd.py 8 8channels1x 5 5 5 15 geotrellis-test courage-services/landsat-cloudless-2016.tif courage-services/nlcd-resized-2016.tif courage-services/central-valley-simplified
