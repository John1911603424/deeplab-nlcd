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


def evaluate(model,
             libchips,
             device,
             args,
             arg_hash):
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
        class_count = len(args.class_weights)
        tps = [0.0 for x in range(class_count)]
        fps = [0.0 for x in range(class_count)]
        fns = [0.0 for x in range(class_count)]
        tns = [0.0 for x in range(class_count)]
        pred_pcts = []
        gt_pcts = []
        l1s = []
        l2s = []

        batch_mult = 2
        for _ in range(args.max_eval_windows // (batch_mult * args.batch_size)):
            batch = get_batch(libchips, args, batch_multiplier=batch_mult)
            pred = model(batch[0].to(device))

            if isinstance(pred, dict):
                pred_seg = pred.get('seg', pred.get('out', None))
                pred_2seg = pred.get('2seg', None)
                pred_reg = pred.get('reg', None)
            else:
                pred_seg = pred
                pred_2seg = pred_reg = None

            if args.window_size_labels != args.window_size_imagery:
                if pred_seg is not None:
                    pred_seg = torch.nn.functional.interpolate(
                        pred_seg, args.window_size_labels, mode='bilinear', align_corners=False)
                if pred_2seg is not None:
                    pred_2seg = torch.nn.functional.interpolate(
                        pred_2seg, args.window_size_labels, mode='bilinear', align_corners=False)

            # segmentation predictions
            pred_seg_mask = None
            if pred_seg is not None:
                pred_seg = torch.max(pred_seg, 1)[1].cpu().numpy()
                pred_seg_mask = pred_seg
            if pred_2seg is not None:
                pred_2seg = pred_2seg.cpu().numpy()
                pred_2seg = np.array(pred_2seg > 0.0, dtype=np.long)
                pred_2seg = pred_2seg[:, 0, :, :]
                pred_seg_mask = pred_2seg
            if pred_reg is not None:
                pred_reg = pred_reg.cpu().numpy()
                if args.bce:
                    pred_reg = (pred_reg > 0).astype(np.float32)
                labels_reg = batch[1].cpu().numpy()
                if pred_reg.shape[-1] == 1:
                    for (pred, actual) in zip(pred_reg, labels_reg):
                        pred_pcts.append(float(pred))
                        yes = float((actual == 1).sum())
                        no = float((actual == 0).sum())
                        gt_pct = yes/(yes + no + 1e-8)
                        gt_pcts.append(gt_pct)
                else:
                    for (pred, actual) in zip(pred_reg, labels_reg):
                        diff = pred - actual
                        l1s.append(diff)
                        l2s.append(diff**2)
                    pred_seg_mask = pred_reg.astype(np.long)

            # segmentation labels
            labels_seg = batch[1].cpu().numpy()

            # don't care values
            if args.label_nd is not None:
                dont_care = (labels_seg == args.label_nd)
            else:
                dont_care = np.zeros(labels_seg.shape)

            if pred_seg_mask is not None:
                for j in range(class_count):
                    tps[j] = tps[j] + (
                        (pred_seg_mask == j) *
                        (labels_seg == j) *
                        (dont_care != 1)
                    ).sum()
                    fps[j] = fps[j] + (
                        (pred_seg_mask == j) *
                        (labels_seg != j) *
                        (dont_care != 1)
                    ).sum()
                    fns[j] = fns[j] + (
                        (pred_seg_mask != j) *
                        (labels_seg == j) *
                        (dont_care != 1)
                    ).sum()
                    tns[j] = tns[j] + (
                        (pred_seg_mask != j) *
                        (labels_seg != j) *
                        (dont_care != 1)
                    ).sum()

            if random.randint(0, args.batch_size * 4) == 0:
                libchips.recenter(1)

            global EVALUATIONS_BATCHES_DONE
            EVALUATIONS_BATCHES_DONE += 1
            with WATCHDOG_MUTEX:
                global WATCHDOG_TIME
                WATCHDOG_TIME = time.time()

    with open('/tmp/evaluations.txt', 'w') as evaluations:
        if tps and fps and tns and fns:
            recalls = []
            precisions = []
            f1s = []
            for j in range(class_count):
                recall = tps[j] / (tps[j] + fns[j] + 1e-8)
                recalls.append(recall)
                precision = tps[j] / (tps[j] + fps[j] + 1e-8)
                precisions.append(precision)
            for j in range(class_count):
                f1 = 2 * (precisions[j] * recalls[j]) / \
                    (precisions[j] + recalls[j] + 1e-8)
                f1s.append(f1)
            print('True Positives  {}'.format(tps))
            print('False Positives {}'.format(fps))
            print('False Negatives {}'.format(fns))
            print('True Negatives  {}'.format(tns))
            print('Recalls    {}'.format(recalls))
            print('Precisions {}'.format(precisions))
            print('f1 {}'.format(f1s))
            evaluations.write('True positives: {}\n'.format(tps))
            evaluations.write('False positives: {}\n'.format(fps))
            evaluations.write('False negatives: {}\n'.format(fns))
            evaluations.write('True negatives: {}\n'.format(tns))
            evaluations.write('Recalls: {}\n'.format(recalls))
            evaluations.write('Precisions: {}\n'.format(precisions))
            evaluations.write('f1 scores: {}\n'.format(f1s))
        if pred_pcts and gt_pcts:
            pred_pcts = np.array(pred_pcts)
            gt_pcts = np.array(gt_pcts)
            errors = pred_pcts - gt_pcts
            relative_errors = errors / (gt_pcts + 1e-8)
            print('MAE = {}, MSE = {}, MRE = {}, MARE = {}'.format(
                np.abs(errors).mean(), (errors**2).mean(),
                relative_errors.mean(), np.abs(relative_errors).mean()))
            print('mean prediction = {}, mean actual = {}'.format(
                pred_pcts.mean(), gt_pcts.mean()))
            evaluations.write('MAE = {}, MSE = {}, MRE = {}, MARE = {}'.format(
                np.abs(errors).mean(), (errors**2).mean(),
                relative_errors.mean(), np.abs(relative_errors).mean()))
            evaluations.write('mean prediction = {}, mean actual = {}'.format(
                pred_pcts.mean(), gt_pcts.mean()))
        if l1s and l2s:
            l1s = np.stack(l1s)
            l2s = np.stack(l2s)
            print('MAE = {}, MSE = {}'.format(l1s.mean(), l2s.mean()))
            evaluations.write(
                'MAE = {}, MSE = {}'.format(l1s.mean(), l2s.mean()))

    if not args.no_upload:
        s3 = boto3.client('s3')
        s3.upload_file('/tmp/evaluations.txt', args.s3_bucket,
                       '{}/{}/evaluations.txt'.format(args.s3_prefix, arg_hash))
        del s3
