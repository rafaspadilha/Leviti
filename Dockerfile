# docker-tensorflow - Tensorflow in Docker for personal purpose and reproducible deep learning

FROM tensorflow/tensorflow:latest-gpu 

# install packages
RUN apt-get update -qq \
 && apt-get -y upgrade \
 && apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    git \
    htop \
    vim \
    cmake \
    curl \
    # install python 2
    python \
    python-dev \
    python-pip \
    python-setuptools \
    python-virtualenv \
    python-colorama \
    python-wheel \
    pkg-config \
    # requirements for numpy
    libopenblas-base \
    python-numpy \
    python-scipy \
    python-skimage


# opencv
RUN apt-get install -y -q libavformat-dev libavcodec-dev libavfilter-dev libswscale-dev

RUN apt-get install -y -q libjpeg-dev libpng-dev libtiff-dev libjasper-dev zlib1g-dev libopenexr-dev libxine-dev libeigen3-dev libtbb-dev

ADD build_opencv.sh /build_opencv.sh

RUN /bin/sh /build_opencv.sh

RUN rm -rf /build_opencv.sh



## CLEAN
RUN apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# dump package lists
RUN dpkg-query -l > /dpkg-query-l.txt \
 && pip2 freeze > /pip2-freeze.txt




RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
    ipykernel \
    jupyter \
    matplotlib \
    h5py \
    pydot-ng \
    graphviz \
    && \
    python -m ipykernel.kernelspec



# keras
RUN pip install keras


#RUN groupadd --gid 1000 recod && \
#    useradd --create-home --shell /bin/bash --uid 10037 --gid 1000 piresramon && \
#    echo "piresramon:piresramon" | chpasswd && \
#    adduser piresramon sudo && \
#    su -l piresramon

#USER piresramon

WORKDIR /home/rpadilha