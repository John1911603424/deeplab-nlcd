FROM jamesmcclain/aws-batch-ml:4

RUN pip install shapely pyproj
COPY libchips/src/libchips.so /tmp/
COPY models/deeplab_water.pth /tmp/
COPY models/deeplab_buildings.pth /tmp/
COPY python/deeplab_inference.py /scripts/
COPY python/elaborate.py /scripts/
COPY scripts/local.sh /scripts/
COPY scripts/aws.sh /scripts/
