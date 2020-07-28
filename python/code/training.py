# The MIT License (MIT)
# =====================
#
# Copyright © 2020 Azavea
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


def get_batch(libchips,
              args,
              batch_multiplier=1):
    """Read a batch of imagery and labels

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

        while True:
            again = False
            libchips.get_next(temp1_ptr, temp2_ptr)
            if args.forbidden_imagery_value is not None:
                again = again or np.any(
                    temp1 == args.forbidden_imagery_value)
            if args.forbidden_label_value is not None:
                again = again or np.any(
                    temp2 == args.forbidden_label_value)
            if args.desired_label_value is not None:
                if not np.any(temp2 == args.desired_label_value):
                    again = again or (args.reroll > random.random())
            if not again:
                break

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

        # NODATA from NaNs in rasters
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


def train(model,
          opt,
          sched,
          obj,
          epochs,
          libchips,
          device,
          args,
          arg_hash,
          no_checkpoints=True,
          starting_epoch=0):
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
            batch = get_batch(libchips, args)
            opt.zero_grad()
            pred: PRED = model(batch[0].to(device))
            loss = None

            if isinstance(pred, dict):
                pred_seg = pred.get('seg', pred.get('out', None))
                pred_aux = pred.get('aux', None)
                pred_2seg = pred.get('2seg', None)
                pred_reg = pred.get('reg', None)
            else:
                pred_seg = pred
                pred_aux = pred_2seg = pred_reg = None

            if pred_seg is not None and pred_aux is None:
                # segmentation only
                labels = batch[1].to(device)
                loss = obj.get('seg')(pred_seg, labels)
            elif pred_seg is not None and pred_aux is not None:
                # segmentation with auxiliary output
                labels = batch[1].to(device)
                loss = obj.get('seg')(pred_seg, labels) + \
                    0.4 * obj.get('seg')(pred_aux, labels)
            elif pred_2seg is not None and pred_reg is None:
                # binary segmentation only
                labels = (batch[1] == 1).to(device, dtype=torch.float)
                # XXX the above assumes that background and target are 0 and 1, respectively
                pred_2seg = pred_2seg[:, 0, :, :]
                loss = obj.get('2seg')(pred_2seg, labels)
            elif pred_2seg is not None and pred_reg is not None:
                # binary segmentation with percent regression
                labels = (batch[1] == 1).to(device, dtype=torch.float)
                # XXX the above and below assume that background and target are 0 and 1, respectively
                pcts = []
                for label in batch[1].cpu().numpy():
                    ones = float((label == 1).sum())
                    zeros = float((label == 0).sum())
                    pcts.append([(ones/(ones + zeros + 1e-8))])
                pcts = torch.FloatTensor(pcts).to(device)
                pred_2seg = pred_2seg[:, 0, :, :]
                loss = obj.get('2seg')(pred_2seg, labels) + \
                    obj.get('l1')(pred_reg, pcts)
            elif pred_seg is None and pred_aux is None and pred_2seg is None and pred_reg is not None:
                # regression only
                pcts = []
                for label in batch[1].cpu().numpy():
                    # XXX assumes that background and target are 0 and 1, respectively
                    ones = float((label == 1).sum())
                    zeros = float((label == 0).sum())
                    pcts.append([(ones/(ones + zeros + 1e-8))])
                pcts = torch.FloatTensor(pcts).to(device)
                if args.bce:
                    # Binary cross entropy
                    loss = obj.get('2seg')(pred_reg, pcts)
                else:
                    # l1 and l2
                    loss = obj.get('l1')(pred_reg, pcts) + \
                        obj.get('l2')(pred_reg, pcts)

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
        print('\t\t epoch={}/{} time={} avg_loss={}'.format(
            i+1, epochs, current_time - last_time, avg_loss))

        with WATCHDOG_MUTEX:
            global WATCHDOG_TIME
            WATCHDOG_TIME = time.time()

        if ((i == epochs - 1) or ((i > 0) and (i % 13 == 0) and args.s3_bucket and args.s3_prefix)) and not no_checkpoints:
            if not args.no_upload:
                torch.save(model.state_dict(), 'weights.pth')
                s3 = boto3.client('s3')
                checkpoint_name = '{}/{}/weights_checkpoint_{}.pth'.format(
                    args.s3_prefix, arg_hash, i)
                print('\t\t checkpoint_name={}'.format(checkpoint_name))
                s3.upload_file(
                    'weights.pth', args.s3_bucket, checkpoint_name)
                del s3
