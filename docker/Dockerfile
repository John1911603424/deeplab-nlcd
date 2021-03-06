FROM pytorch/pytorch:nightly-runtime-cuda10.0-cudnn7

RUN conda install -c pytorch pytorch && \
    conda clean -a
RUN apt-get update && apt-get install -y libcurl4-gnutls-dev pkg-config wget libgeos-dev cmake && \
    apt-get autoremove && apt-get autoclean
RUN cd /usr/local/src && \
    wget -k 'https://download.osgeo.org/gdal/2.4.3/gdal-2.4.3.tar.gz' && \
    wget -k 'https://download.osgeo.org/proj/proj-4.9.3.tar.gz' && \
    wget -k 'https://github.com/uclouvain/openjpeg/archive/v2.3.1.tar.gz' -O 'openjpeg-2.3.1.tar.gz' && \
    tar axvf gdal-2.4.3.tar.gz && tar axvf proj-4.9.3.tar.gz && tar axvf openjpeg-2.3.1.tar.gz && \
    cd proj-4.9.3 && ./configure --prefix=/usr/local && nice -n 19 make -j33 && make install && \
    cd ../openjpeg-2.3.1 && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && nice -n 19 make -j33 && make install && \
    cd ../../gdal-2.4.3 && ./configure --prefix=/usr/local --with-proj=/usr/local --with-openjpeg=/usr/local --without-xml2 && nice -n 19 make -j33 && make install && cd swig/python && python3 setup.py build && mkdir -p /opt/conda/lib/python3.6/site-packages && python3 setup.py install --prefix=/opt/conda && \
    cd ../../.. && rm -r proj-4.9.3/ gdal-2.4.3/ openjpeg-2.3.1/ proj-4.9.3.tar.gz gdal-2.4.3.tar.gz openjpeg-2.3.1.tar.gz && \
    echo '/usr/local/lib >> /etc/ld.so.conf' && ldconfig
RUN pip install --upgrade awscli boto3 pyproj rasterio pystac==0.3.1 shapely sklearn torchvision --upgrade && \
    rm -rf /root/.cache/pip
RUN pip install s2cloudless

COPY ./download_run.sh /workspace/download_run.sh
