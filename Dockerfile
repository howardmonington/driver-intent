FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y git wget

WORKDIR /workspace/notebooks

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils openmpi-bin openmpi-common libopenmpi-dev && \
    apt-get clean

RUN apt-get -y install python3-pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
