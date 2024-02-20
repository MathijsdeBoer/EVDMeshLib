FROM nvidia/cuda:12.0.1-devel-ubuntu22.04
LABEL authors="Mathijs de Boer"

ARG DEBIAN_FRONTEND=noninteractive
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="${PATH}:/usr/local/cargo/bin:~/.local/bin"\
    TZ=Europe/Amsterdam

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    curl \
    wget \
    build-essential \
    git \
    software-properties-common \
    libx11-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    virtualenv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && cargo --version \
    && python -m pip install -U pip \
    && pip --version \
    && python --version \
    && ldconfig /usr/local/cuda-12.3/compat/ \
    && ldconfig -p | grep libcuda

COPY . /evdplanner
RUN pip install /evdplanner
