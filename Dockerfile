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

RUN apt-get install -y -q libavformat-dev libavcodec-dev libavfilter-dev libswscale-dev

RUN apt-get install -y -q libjpeg-dev libpng-dev libtiff-dev libjasper-dev zlib1g-dev libopenexr-dev libxine-dev libeigen3-dev libtbb-dev

ADD build_opencv.sh /build_opencv.sh

RUN /bin/sh /build_opencv.sh

RUN rm -rf /build_opencv.sh

RUN apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# dump package lists
RUN dpkg-query -l > /dpkg-query-l.txt \
 && pip2 freeze > /pip2-freeze.txt

RUN pip install ghalton click

#RUN groupadd --gid 1000 recod && \
#    useradd --create-home --shell /bin/bash --uid 10037 --gid 1000 piresramon && \
#    echo "piresramon:piresramon" | chpasswd && \
#    adduser piresramon sudo && \
#    su -l piresramon

#USER piresramon

WORKDIR /home/rpadilha