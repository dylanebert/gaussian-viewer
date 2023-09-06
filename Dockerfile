FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDAToolkit_ROOT=/usr/local/cuda

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,video
ENV TORCH_CUDA_ARCH_LIST=7.5+PTX

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt-get install -y \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libavutil-dev \
    libglm-dev \
    wget \
    build-essential \
    ninja-build \
    cmake && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm -rf /var/lib/apt/lists/*

COPY . /

RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
RUN python3.10 -m pip install --no-cache-dir diff-gaussian-rasterization/
RUN python3.10 -m pip install --no-cache-dir VideoProcessingFramework/
RUN python3.10 -m pip install --no-cache-dir VideoProcessingFramework/src/PytorchNvCodec/

EXPOSE 443

CMD ["python3.10", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "443", "--ssl-keyfile", "privkey.pem", "--ssl-certfile", "fullchain.pem"]
