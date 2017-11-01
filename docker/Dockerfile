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
    python-skimage \
    #requirement for matplotlib
    python-tk \
    # requirements for opencv
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libxext6


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
    opencv-contrib-python \
    && \
    python -m ipykernel.kernelspec



# keras
RUN pip install keras


RUN groupadd --gid 1000 recod && \
    useradd --create-home --shell /bin/bash --uid 10032 --gid 1000 rpadilha && \
    echo "rpadilha:rpadilha" | chpasswd && \
    adduser rpadilha sudo && \
    su -l rpadilha

USER rpadilha

WORKDIR /home/rpadilha
