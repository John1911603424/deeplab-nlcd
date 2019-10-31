FROM jamesmcclain/aws-batch-ml:4

RUN pip install shapely pyproj
COPY libchips/src/libchips.so /tmp/
COPY models/deeplab_sample_water1.pth /tmp/
COPY models/deeplab_sample_water2.pth /tmp/
COPY models/deeplab_sample_buildings.pth /tmp/
COPY python/deeplab_inference.py /scripts/
COPY python/elaborate.py /scripts/
COPY scripts/local.sh /scripts/
COPY scripts/aws.sh /scripts/
