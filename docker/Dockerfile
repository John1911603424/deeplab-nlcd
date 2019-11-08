FROM pytorch/pytorch:nightly-runtime-cuda10.0-cudnn7

RUN conda install -c pytorch -c fastai fastai && \
    conda clean -a
RUN pip install awscli boto3 torchvision jupyterlab rasterio sklearn tabulate pyitlib --upgrade && \
    rm -rf /root/.cache/pip
RUN apt-get update && apt-get install -y libcurl4-gnutls-dev pkg-config wget libgeos-dev && \
    apt-get autoremove && apt-get autoclean
RUN cd /usr/local/src && \
    wget -k 'https://download.osgeo.org/gdal/2.4.2/gdal-2.4.2.tar.gz' && \
    wget -k 'https://download.osgeo.org/proj/proj-4.9.3.tar.gz' && \
    tar axvf gdal-2.4.2.tar.gz && tar axvf proj-4.9.3.tar.gz && \
    cd proj-4.9.3 && ./configure --prefix=/usr && nice -n 19 make -j33 && make install && \
    cd ../gdal-2.4.2 && ./configure --prefix=/usr --without-xml2 && nice -n 19 make -j33 && make install && \
    cd .. && rm -r proj-4.9.3/ gdal-2.4.2/ proj-4.9.3.tar.gz gdal-2.4.2.tar.gz

COPY ./download_run.sh /workspace/download_run.sh
COPY ./download_run_upload_jupyter.sh /workspace/download_run_upload_jupyter.sh
