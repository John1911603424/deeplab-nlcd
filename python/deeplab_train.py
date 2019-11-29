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
#
# The code in this file is under the MIT license except where
# indicted.

import argparse
import codecs
import copy
import ctypes
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

import boto3  # type: ignore
import numpy as np  # type: ignore
import requests

import torch
import torchvision  # type: ignore

WATCHDOG_MUTEX: threading.Lock = threading.Lock()
WATCHDOG_TIME: float = time.time()
EVALUATIONS_BATCHES_DONE = 0

INT2INT = Dict[int, int]
OBJ = Union[torch.nn.CrossEntropyLoss, torch.nn.BCEWithLogitsLoss]
OPT = Union[torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW]
PRED = Union[Dict[str, torch.Tensor], torch.Tensor]
SCALER = Union[int, float]

# OneCycleLR
if not hasattr(torch.optim.lr_scheduler, 'OneCycleLR'):
    # This class is from PyTorch.  Please see https://github.com/pytorch/pytorch/blob/master/LICENSE for license and copyright information.
    class OneCycleLR(torch.optim.lr_scheduler._LRScheduler):
        r"""Sets the learning rate of each parameter group according to the
        1cycle learning rate policy. The 1cycle policy anneals the learning
        rate from an initial learning rate to some maximum learning rate and then
        from that maximum learning rate to some minimum learning rate much lower
        than the initial learning rate.
        This policy was initially described in the paper `Super-Convergence:
        Very Fast Training of Neural Networks Using Large Learning Rates`_.

        The 1cycle learning rate policy changes the learning rate after every batch.
        `step` should be called after a batch has been used for training.

        This scheduler is not chainable.

        This class has two built-in annealing strategies:
        "cos":
            Cosine annealing
        "linear":
            Linear annealing

        Note also that the total number of steps in the cycle can be determined in one
        of two ways (listed in order of precedence):
        1) A value for total_steps is explicitly provided.
        2) A number of epochs (epochs) and a number of steps per epoch
        (steps_per_epoch) are provided.
        In this case, the number of total steps is inferred by
        total_steps = epochs * steps_per_epoch
        You must either provide a value for total_steps or provide a value for both
        epochs and steps_per_epoch.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            max_lr (float or list): Upper learning rate boundaries in the cycle
                for each parameter group.
            total_steps (int): The total number of steps in the cycle. Note that
                if a value is provided here, then it must be inferred by providing
                a value for epochs and steps_per_epoch.
                Default: None
            epochs (int): The number of epochs to train for. This is used along
                with steps_per_epoch in order to infer the total number of steps in the cycle
                if a value for total_steps is not provided.
                Default: None
            steps_per_epoch (int): The number of steps per epoch to train for. This is
                used along with epochs in order to infer the total number of steps in the
                cycle if a value for total_steps is not provided.
                Default: None
            pct_start (float): The percentage of the cycle (in number of steps) spent
                increasing the learning rate.
                Default: 0.3
            anneal_strategy (str): {'cos', 'linear'}
                Specifies the annealing strategy.
                Default: 'cos'
            cycle_momentum (bool): If ``True``, momentum is cycled inversely
                to learning rate between 'base_momentum' and 'max_momentum'.
                Default: True
            base_momentum (float or list): Lower momentum boundaries in the cycle
                for each parameter group. Note that momentum is cycled inversely
                to learning rate; at the peak of a cycle, momentum is
                'base_momentum' and learning rate is 'max_lr'.
                Default: 0.85
            max_momentum (float or list): Upper momentum boundaries in the cycle
                for each parameter group. Functionally,
                it defines the cycle amplitude (max_momentum - base_momentum).
                Note that momentum is cycled inversely
                to learning rate; at the start of a cycle, momentum is 'max_momentum'
                and learning rate is 'base_lr'
                Default: 0.95
            div_factor (float): Determines the initial learning rate via
                initial_lr = max_lr/div_factor
                Default: 25
            final_div_factor (float): Determines the minimum learning rate via
                min_lr = initial_lr/final_div_factor
                Default: 1e4
            last_epoch (int): The index of the last batch. This parameter is used when
                resuming a training job. Since `step()` should be invoked after each
                batch instead of after each epoch, this number represents the total
                number of *batches* computed, not the total number of epochs computed.
                When last_epoch=-1, the schedule is started from the beginning.
                Default: -1

        Example:
            >>> data_loader = torch.utils.data.DataLoader(...)
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)
            >>> for epoch in range(10):
            >>>     for batch in data_loader:
            >>>         train_batch(...)
            >>>         scheduler.step()


        .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
            https://arxiv.org/abs/1708.07120
        """

        def __init__(self,
                     optimizer,
                     max_lr,
                     total_steps=None,
                     epochs=None,
                     steps_per_epoch=None,
                     pct_start=0.3,
                     anneal_strategy='cos',
                     cycle_momentum=True,
                     base_momentum=0.85,
                     max_momentum=0.95,
                     div_factor=25.,
                     final_div_factor=1e4,
                     last_epoch=-1):

            # Validate optimizer
            self.optimizer = optimizer

            # Validate total_steps
            if total_steps is None and epochs is None and steps_per_epoch is None:
                raise ValueError(
                    "You must define either total_steps OR (epochs AND steps_per_epoch)")
            elif total_steps is not None:
                if total_steps <= 0 or not isinstance(total_steps, int):
                    raise ValueError(
                        "Expected non-negative integer total_steps, but got {}".format(total_steps))
                self.total_steps = total_steps
            else:
                if epochs <= 0 or not isinstance(epochs, int):
                    raise ValueError(
                        "Expected non-negative integer epochs, but got {}".format(epochs))
                if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                    raise ValueError(
                        "Expected non-negative integer steps_per_epoch, but got {}".format(steps_per_epoch))
                self.total_steps = epochs * steps_per_epoch
            self.step_size_up = float(pct_start * self.total_steps) - 1
            self.step_size_down = float(
                self.total_steps - self.step_size_up) - 1

            # Validate pct_start
            if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
                raise ValueError(
                    "Expected float between 0 and 1 pct_start, but got {}".format(pct_start))

            # Validate anneal_strategy
            if anneal_strategy not in ['cos', 'linear']:
                raise ValueError(
                    "anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
            elif anneal_strategy == 'cos':
                self.anneal_func = self._annealing_cos
            elif anneal_strategy == 'linear':
                self.anneal_func = self._annealing_linear

            # Initialize learning rate variables
            max_lrs = self._format_param('max_lr', self.optimizer, max_lr)
            if last_epoch == -1:
                for idx, group in enumerate(self.optimizer.param_groups):
                    group['lr'] = max_lrs[idx] / div_factor
                    group['max_lr'] = max_lrs[idx]
                    group['min_lr'] = group['lr'] / final_div_factor

            # Initialize momentum variables
            self.cycle_momentum = cycle_momentum
            if self.cycle_momentum:
                if 'momentum' not in self.optimizer.defaults and 'betas' not in self.optimizer.defaults:
                    raise ValueError(
                        'optimizer must support momentum with `cycle_momentum` option enabled')
                self.use_beta1 = 'betas' in self.optimizer.defaults
                max_momentums = self._format_param(
                    'max_momentum', optimizer, max_momentum)
                base_momentums = self._format_param(
                    'base_momentum', optimizer, base_momentum)
                if last_epoch == -1:
                    for m_momentum, b_momentum, group in zip(max_momentums, base_momentums, optimizer.param_groups):
                        if self.use_beta1:
                            _, beta2 = group['betas']
                            group['betas'] = (m_momentum, beta2)
                        else:
                            group['momentum'] = m_momentum
                        group['max_momentum'] = m_momentum
                        group['base_momentum'] = b_momentum

            super(OneCycleLR, self).__init__(optimizer, last_epoch)

        def _format_param(self, name, optimizer, param):
            """Return correctly formatted lr/momentum for each param group."""
            if isinstance(param, (list, tuple)):
                if len(param) != len(optimizer.param_groups):
                    raise ValueError("expected {} values for {}, got {}".format(
                        len(optimizer.param_groups), name, len(param)))
                return param
            else:
                return [param] * len(optimizer.param_groups)

        def _annealing_cos(self, start, end, pct):
            "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
            cos_out = math.cos(math.pi * pct) + 1
            return end + (start - end) / 2.0 * cos_out

        def _annealing_linear(self, start, end, pct):
            "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
            return (end - start) * pct + start

        def get_lr(self):
            lrs = []
            step_num = self.last_epoch

            if step_num > self.total_steps:
                raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                                 .format(step_num + 1, self.total_steps))

            for group in self.optimizer.param_groups:
                if step_num <= self.step_size_up:
                    computed_lr = self.anneal_func(
                        group['initial_lr'], group['max_lr'], step_num / self.step_size_up)
                    if self.cycle_momentum:
                        computed_momentum = self.anneal_func(group['max_momentum'], group['base_momentum'],
                                                             step_num / self.step_size_up)
                else:
                    down_step_num = step_num - self.step_size_up
                    computed_lr = self.anneal_func(
                        group['max_lr'], group['min_lr'], down_step_num / self.step_size_down)
                    if self.cycle_momentum:
                        computed_momentum = self.anneal_func(group['base_momentum'], group['max_momentum'],
                                                             down_step_num / self.step_size_down)

                lrs.append(computed_lr)
                if self.cycle_momentum:
                    if self.use_beta1:
                        _, beta2 = group['betas']
                        group['betas'] = (computed_momentum, beta2)
                    else:
                        group['momentum'] = computed_momentum

            return lrs
else:
    from torch.optim.lr_scheduler import OneCycleLR

SCHED = Optional[OneCycleLR]

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
                  args: argparse.Namespace,
                  batch_multiplier: int = 1,
                  should_jitter: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read the data specified in the given plan

        Arguments:
            libchips {ctypes.CDLL} -- A shared library handle used for reading data
            args {argparse.Namespace} -- The arguments dictionary

        Keyword Arguments:
            batch_multiplier {int} -- How many base batches to fetch at once

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
        for _ in range(args.batch_size * batch_multiplier):
            libchips.get_next(temp1_ptr, temp2_ptr)
            if should_jitter:
                jitter = np.random.rand(
                    len(args.bands), 1, 1).astype(np.float32)/5.0 + 0.90
                rasters.append(temp1.copy() * jitter)
            else:
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

            if not args.architecture.endswith('-binary.py'):
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
              sched: SCHED,
              obj: OBJ,
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
            opt {OPT} -- The optimizer to use
            obj {OBJ} -- The objective function to use
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
        for i in range(starting_epoch, epochs):
            avg_loss = 0.0
            for _ in range(args.max_epoch_size):
                batch = get_batch(
                    libchips, args, should_jitter=args.color_jitter)
                while (not (batch[1] == 1).any()) and (args.reroll > random.random()):
                    batch = get_batch(
                        libchips, args, should_jitter=args.color_jitter)
                opt.zero_grad()
                pred: PRED = model(batch[0].to(device))
                with torch.autograd.detect_anomaly():
                    if args.architecture.endswith('-binary.py'):
                        if '-regression' in args.architecture:
                            pred_out: torch.Tensor = pred.get('out')
                            pred_pcts: torch.Tensor = pred.get('pct')

                            # segmentation
                            label_float = (batch[1] == 1).to(
                                device, dtype=torch.float)
                            pred_sliced = pred_out[:, 0, :, :]
                            seg_loss = obj.get('obj1')(
                                pred_sliced, label_float)

                            # regression
                            pcts = []
                            # XXX assumes that background and target are 0 and 1, respectively
                            for label in batch[1].cpu().numpy():
                                ones = float((label == 1).sum())
                                zeros = float((label == 0).sum())
                                pcts.append([(ones/(ones + zeros + 1e-6))])
                            pcts = torch.FloatTensor(pcts).to(device)
                            reg_loss = obj.get('obj2')(pred_pcts, pcts)
                            reg_loss = reg_loss

                            loss = seg_loss + reg_loss
                        else:
                            label_float = (batch[1] == 1).to(
                                device, dtype=torch.float)
                            pred_sliced = pred[:, 0, :, :]
                            loss = obj(pred_sliced, label_float)
                    else:
                        label_long = batch[1].to(device)
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
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
                    opt.step()
                    if sched is not None:
                        sched.step()
                    avg_loss = avg_loss + loss.item()

            avg_loss = avg_loss / args.max_epoch_size
            libchips.recenter(1)

            last_time = current_time
            current_time = time.time()
            print('\t\t epoch={} time={} avg_loss={}'.format(
                i, current_time - last_time, avg_loss))

            with WATCHDOG_MUTEX:
                global WATCHDOG_TIME
                WATCHDOG_TIME = time.time()

            if ((i == epochs - 1) or ((i > 0) and (i % 5 == 0) and args.s3_bucket and args.s3_prefix)) and not no_checkpoints:
                if not args.no_upload:
                    torch.save(model.state_dict(), 'weights.pth')
                    s3 = boto3.client('s3')
                    s3.upload_file('weights.pth', args.s3_bucket,
                                   '{}/{}/weights_checkpoint_{}.pth'.format(args.s3_prefix, arg_hash, i))
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
        model.eval()
        with torch.no_grad():
            class_count = len(args.weights)
            tps = [0.0 for x in range(class_count)]
            fps = [0.0 for x in range(class_count)]
            fns = [0.0 for x in range(class_count)]
            tns = [0.0 for x in range(class_count)]
            percents = []

            batch_mult = 2
            for _ in range(args.max_eval_windows // (batch_mult * args.batch_size)):
                batch = get_batch(libchips, args, batch_multiplier=batch_mult)
                out = model(batch[0].to(device))
                if isinstance(out, dict):
                    if 'pct' in out:
                        percents.append(out.get('pct').cpu().numpy())
                    out = out['out']
                out = out.data.cpu().numpy()
                if args.architecture.endswith('-binary.py'):
                    out = np.array(out > 0.5, dtype=np.long)
                    out = out[:, 0, :, :]
                else:
                    out = np.apply_along_axis(np.argmax, 1, out)

                labels = batch[1].cpu().numpy()
                del batch

                # Make sure output reflects don't care values
                if args.label_nd is not None:
                    dont_care = (labels == args.label_nd)
                else:
                    dont_care = np.zeros(labels.shape)

                for j in range(class_count):
                    tps[j] = tps[j] + ((out == j)*(labels == j)
                                       * (dont_care != 1)).sum()
                    fps[j] = fps[j] + ((out == j)*(labels != j)
                                       * (dont_care != 1)).sum()
                    fns[j] = fns[j] + ((out != j)*(labels == j)
                                       * (dont_care != 1)).sum()
                    tns[j] = tns[j] + ((out != j)*(labels != j)
                                       * (dont_care != 1)).sum()

                if random.randint(0, args.batch_size * 4) == 0:
                    libchips.recenter(1)

                global EVALUATIONS_BATCHES_DONE
                EVALUATIONS_BATCHES_DONE += 1
                with WATCHDOG_MUTEX:
                    global WATCHDOG_TIME
                    WATCHDOG_TIME = time.time()

        if '-regression' in args.architecture and len(percents) > 0:
            pred_percentage = np.stack(percents).mean()
        else:
            pred_percentage = None

        print('True Positives  {}'.format(tps))
        print('False Positives {}'.format(fps))
        print('False Negatives {}'.format(fns))
        print('True Negatives  {}'.format(tns))

        recalls = []
        precisions = []
        for j in range(class_count):
            recall = tps[j] / (tps[j] + fns[j])
            recalls.append(recall)
            precision = tps[j] / (tps[j] + fps[j])
            precisions.append(precision)

        print('Recalls    {}'.format(recalls))
        print('Precisions {}'.format(precisions))

        f1s = []
        for j in range(class_count):
            f1 = 2 * (precisions[j] * recalls[j]) / \
                (precisions[j] + recalls[j])
            f1s.append(f1)
        print('f1 {}'.format(f1s))
        if '-regression' in args.architecture:
            # XXX assumes that background and target are 0 and 1, respectively
            print('predicted percentage = {}, actual percentage = {}'.format(
                pred_percentage, (tps[1] + fns[1])/(tps[1] + fns[1] + tns[1] + fps[1])))

        with open('/tmp/evaluations.txt', 'w') as evaluations:
            evaluations.write('True positives: {}\n'.format(tps))
            evaluations.write('False positives: {}\n'.format(fps))
            evaluations.write('False negatives: {}\n'.format(fns))
            evaluations.write('True negatives: {}\n'.format(tns))
            evaluations.write('Recalls: {}\n'.format(recalls))
            evaluations.write('Precisions: {}\n'.format(precisions))
            evaluations.write('f1 scores: {}\n'.format(f1s))
            if '-regression' in args.architecture:
                # XXX assumes that background and target are 0 and 1, respectively
                evaluations.write('predicted percentage = {}, actual percentage = {}'.format(
                    pred_percentage, (tps[1] + fns[1])/(tps[1] + fns[1] + tns[1] + fps[1])))

        if not args.no_upload:
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
        parser.add_argument('--architecture', required=True,
                            help='The desired model architecture')
        parser.add_argument('--backend',
                            help="Don't use this flag unless you know what you're doing: CPU is far slower than CUDA.",
                            choices=['cpu', 'cuda'], default='cuda')
        parser.add_argument('--bands',
                            required=True, help='list of bands to train on (1 indexed)', nargs='+', type=int)
        parser.add_argument('--batch-size', default=16, type=int)
        parser.add_argument('--color-jitter', action='store_true')
        parser.add_argument('--epochs1', default=0, type=int)
        parser.add_argument('--epochs2', default=13, type=int)
        parser.add_argument('--epochs3', default=0, type=int)
        parser.add_argument('--epochs4', default=33, type=int)
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
        parser.add_argument('--optimizer', default='adam',
                            choices=['sgd', 'adam', 'adamw'])
        parser.add_argument('--radius', default=10000)
        parser.add_argument('--read-threads', type=int)
        parser.add_argument('--reroll', default=0.90, type=float)
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
                            help='the input that you are training to produce labels for')
        parser.add_argument('--watchdog-seconds',
                            default=0, type=int,
                            help='The number of seconds that can pass without activity before the program is terminated (0 to disable)')
        parser.add_argument('--weights', nargs='+', type=float)
        parser.add_argument('--window-size', default=32, type=int)
        return parser

# Architectures
if True:
    def make_model(band_count, input_stride=1, class_count=1, divisor=1, pretrained=False):
        raise Exception()

    def load_architectures(uri: str) -> None:
        arch_str = read_text(uri)
        arch_code = compile(arch_str, uri, 'exec')
        exec(arch_code, globals())


if __name__ == '__main__':

    parser = training_cli_parser()
    args = training_cli_parser().parse_args()
    hashed_args = copy.copy(args)
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

    args.mus = np.ndarray(len(args.bands), dtype=np.float64)
    args.sigmas = np.ndarray(len(args.bands), dtype=np.float64)
    mus_ptr = args.mus.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sigmas_ptr = args.sigmas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    args.band_count = len(args.bands)

    load_architectures(args.architecture)

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

    args.pairs = list(zip(args.training_img, args.label_img))
    indexed_pairs = zip(range(len(args.pairs)), args.pairs)
    for (i, (training_img, label_img)) in indexed_pairs:
        mul = '/tmp/mul{}.tif'.format(i)
        mask = '/tmp/mask{}.tif'.format(i)
        if not os.path.exists(mul):
            s3 = boto3.client('s3')
            bucket, prefix = parse_s3_url(training_img)
            print('training image bucket and prefix: {}, {}'.format(bucket, prefix))
            s3.download_file(bucket, prefix, mul)
            del s3
        if not os.path.exists(mask):
            s3 = boto3.client('s3')
            bucket, prefix = parse_s3_url(label_img)
            print('training labels bucket and prefix: {}, {}'.format(bucket, prefix))
            s3.download_file(bucket, prefix, mask)
            del s3

    if not args.read_threads:
        args.read_threads = len(args.pairs)

    # ---------------------------------
    print('NATIVE CODE')

    if not os.path.exists('/tmp/libchips.so'):
        s3 = boto3.client('s3')
        bucket, prefix = parse_s3_url(args.libchips)
        print('shared library bucket and prefix: {}, {}'.format(bucket, prefix))
        s3.download_file(bucket, prefix, '/tmp/libchips.so')
        del s3

    libchips = ctypes.CDLL('/tmp/libchips.so')
    libchips.recenter.argtypes = [ctypes.c_int]
    libchips.get_next.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int)
    ]
    libchips.start_multi.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

    libchips.init()
    libchips.start_multi(
        args.read_threads,  # Number of threads
        max(args.read_threads, args.batch_size * 8),  # Number of slots
        len(args.pairs),  # The number of pairs
        b'/tmp/mul%d.tif',  # Image data
        b'/tmp/mask%d.tif',  # Label data
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
    if not args.no_upload:
        s3 = boto3.client('s3')
        s3.upload_file('/tmp/args.txt', args.s3_bucket,
                       '{}/{}/training_args.txt'.format(args.s3_prefix, arg_hash))
        del s3

    # ---------------------------------
    print('INITIALIZING')

    complete_thru = -1
    current_epoch = 0
    current_pth = None

    if args.start_from is None:
        if not args.no_upload:
            for pth in get_matching_s3_keys(
                    bucket=args.s3_bucket,
                    prefix='{}/{}/'.format(args.s3_prefix, arg_hash),
                    suffix='pth'):
                m1 = re.match('.*weights_(\d+).pth$', pth)
                m2 = re.match('.*weights_checkpoint_(\d+).pth', pth)
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

    if not args.weights:
        args.weights = [1.0, 1.0]
    class_count = len(args.weights)

    if args.label_nd is None:
        args.label_nd = class_count
        print('\t WARNING: LABEL NODATA NOT SET, SETTING TO {}'.format(args.label_nd))

    if args.image_nd is None:
        print('\t WARNING: IMAGE NODATA NOT SET')

    if args.batch_size < 2:
        args.batch_size = 2
        print('\t WARNING: BATCH SIZE MUST BE AT LEAST 2, SETTING TO 2')

    natural_epoch_size = 0.0
    for i in range(len(args.pairs)):
        natural_epoch_size = natural_epoch_size + \
            (libchips.get_width(i) * libchips.get_height(i))
    natural_epoch_size = (6.0 * natural_epoch_size) / \
        (7.0 * args.window_size * args.window_size)
    natural_epoch_size = int(natural_epoch_size)
    print('\t NATURAL EPOCH SIZE={}'.format(natural_epoch_size))
    args.max_epoch_size = min(args.max_epoch_size, natural_epoch_size)
    print('\t STEPS PER EPOCH={}'.format(args.max_epoch_size))

    if args.architecture.endswith('-binary.py'):
        obj = torch.nn.BCEWithLogitsLoss().to(device)
    else:
        obj = torch.nn.CrossEntropyLoss(
            ignore_index=args.label_nd,
            weight=torch.FloatTensor(args.weights).to(device)
        ).to(device)

    if '-regression' in args.architecture:
        obj = {'obj1': obj, 'obj2': torch.nn.MSELoss()}

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
        model = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=class_count,
            divisor=args.resolution_divisor,
            pretrained=True
        ).to(device)

    # Phase 1
    if complete_thru == 0:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'weights.pth')
        model = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=class_count,
            divisor=args.resolution_divisor,
            pretrained=True
        ).to(device)
        model.load_state_dict(torch.load('weights.pth'))
        del s3
        print('\t\t SUCCESSFULLY RESTARTED {}'.format(pth))
    elif complete_thru < 0:
        print('\t TRAINING FIRST AND LAST LAYERS (1/2)')

        for p in model.parameters():
            p.requires_grad = False
        for layer in model.input_layers + model.output_layers:
            for p in layer.parameters():
                p.requires_grad = True

        ps = []
        for n, p in model.named_parameters():
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

        train(model,
              opt,
              None,
              obj,
              args.epochs1,
              libchips,
              device,
              copy.copy(args),
              arg_hash)

    # Phase 2
    if complete_thru == 1:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'weights.pth')
        model = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=class_count,
            divisor=args.resolution_divisor,
            pretrained=True
        ).to(device)
        model.load_state_dict(torch.load('weights.pth'))
        del s3
        print('\t\t SUCCESSFULLY RESTARTED {}'.format(pth))
    elif complete_thru < 1:
        print('\t TRAINING FIRST AND LAST LAYERS (2/2)')

        for p in model.parameters():
            p.requires_grad = False
        for layer in model.input_layers + model.output_layers:
            for p in layer.parameters():
                p.requires_grad = True

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
        sched: SCHED = OneCycleLR(opt, max_lr=args.learning_rate2,
                                  epochs=args.epochs2, steps_per_epoch=args.max_epoch_size)

        train(model,
              opt,
              sched,
              obj,
              args.epochs2,
              libchips,
              device,
              copy.copy(args),
              arg_hash)

        if not args.no_upload:
            print('\t UPLOADING')
            torch.save(model.state_dict(), 'weights.pth')
            s3 = boto3.client('s3')
            s3.upload_file('weights.pth', args.s3_bucket,
                           '{}/{}/weights_1.pth'.format(args.s3_prefix, arg_hash))
            del s3

    # Phase 3
    if complete_thru == 2:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'weights.pth')
        model = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=class_count,
            divisor=args.resolution_divisor
        ).to(device)
        model.load_state_dict(torch.load('weights.pth'))
        del s3
        print('\t\t SUCCESSFULLY RESTARTED {}'.format(pth))
    elif complete_thru < 2:
        print('\t TRAINING ALL LAYERS (1/2)')

        for p in model.parameters():
            p.requires_grad = True

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
              copy.copy(args),
              arg_hash)

    # Phase 4
    if complete_thru == 3:
        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'weights.pth')
        model = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=class_count,
            divisor=args.resolution_divisor
        ).to(device)
        model.load_state_dict(torch.load('weights.pth'))
        del s3
        print('\t\t SUCCESSFULLY RESTARTED {}'.format(pth))
    elif complete_thru < 3:
        print('\t TRAINING ALL LAYERS (2/2)')

        for p in model.parameters():
            p.requires_grad = True

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
        sched = OneCycleLR(opt, max_lr=args.learning_rate4,
                           epochs=args.epochs4, steps_per_epoch=args.max_epoch_size)

        train(model,
              opt,
              sched,
              obj,
              args.epochs4,
              libchips,
              device,
              copy.copy(args),
              arg_hash,
              no_checkpoints=False)

        if not args.no_upload:
            print('\t UPLOADING')
            torch.save(model.state_dict(), 'weights.pth')
            s3 = boto3.client('s3')
            s3.upload_file('weights.pth', args.s3_bucket,
                           '{}/{}/weights.pth'.format(args.s3_prefix, arg_hash))
            del s3

    # Restart in Phase 4
    if complete_thru == 4:
        print('\t TRAINING ALL LAYERS FROM CHECKPOINT')

        s3 = boto3.client('s3')
        s3.download_file(args.s3_bucket, current_pth, 'weights.pth')
        model = make_model(
            args.band_count,
            input_stride=args.input_stride,
            class_count=class_count,
            divisor=args.resolution_divisor
        ).to(device)
        model.load_state_dict(torch.load('weights.pth'))
        del s3

        for p in model.parameters():
            p.requires_grad = True

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
        sched = OneCycleLR(opt, max_lr=args.learning_rate4,
                           epochs=args.epochs4, steps_per_epoch=args.max_epoch_size)
        for _ in range(current_epoch):
            sched.step()

        train(model,
              opt,
              sched,
              obj,
              args.epochs4,
              libchips,
              device,
              copy.copy(args),
              arg_hash,
              no_checkpoints=False,
              starting_epoch=current_epoch)

        if not args.no_upload:
            print('\t UPLOADING')
            torch.save(model.state_dict(), 'weights.pth')
            s3 = boto3.client('s3')
            s3.upload_file('weights.pth', args.s3_bucket,
                           '{}/{}/weights.pth'.format(args.s3_prefix, arg_hash))
            del s3

    libchips.stop()

    if not args.no_eval:
        print('\t EVALUATING')
        libchips.start_multi(
            args.read_threads,  # Number of threads
            args.read_threads,  # The number of read slots
            len(args.pairs),  # The number of pairs
            b'/tmp/mul%d.tif',  # Image data
            b'/tmp/mask%d.tif',  # Label data
            6,  # Make all rasters float32
            5,  # Make all labels int32
            None,  # means
            None,  # standard deviations
            args.radius,  # typical radius of a component
            2,  # Evaluation mode
            args.window_size,
            len(args.bands),
            np.array(args.bands, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        evaluate(model,
                 libchips,
                 device,
                 copy.copy(args),
                 arg_hash)
        libchips.stop()

    libchips.deinit()
    exit(0)
