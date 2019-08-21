
def get_eval_batch(raster_ds, label_ds, xys, device):
    data = []
    labels = []
    for x, y in xys:
        d, l = get_eval_window(raster_ds, 1abel_ds, x, y)
        data.append(d)
        labels.append(l)

    data = np.stack(data, axis=0)
    data = torch.from_numpy(data).to(device)
    labels = np.array(np.stack(labels, axis=0), dtype=np.long)
    labels = torch.from_numpy(labels).to(device)
    return (data, labels)

def evaluate(raster_ds, label_ds, label_nd, label_count, window_size):
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
            batch, labels = get_eval_batch(raster_ds, label_ds, xy, device)
            labels = labels.data.cpu().numpy()
            out = deeplab(batch)['out'].data.cpu().numpy()
            out = np.apply_along_axis(np.argmax, 1, out)

            if label_nd is not None:
                dont_care = labels == label_nd
            else:
                dont_care = np.zeros(labels.shape)

            out = out + 10*dont_care

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

    exit(0)

# ./download_run.sh s3://geotrellis-test/courage-services/eval_full_nlcd.py 8 geotrellis-test landsat-cloudless-2016.tif nlcd-resized-2016.tif central-valley-update/deeplab_8channels5x.pth central-valley-update/8channels5x.npy central-valley-update/gt8.npy
