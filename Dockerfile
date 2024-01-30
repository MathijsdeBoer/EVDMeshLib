FROM nvcr.io/nvidia/pytorch:23.12-py3
LABEL authors="Mathijs de Boer"

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="${PATH}:/usr/local/cargo/bin:~/.local/bin"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    wget \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && python -m pip install -U pip

COPY . /code
RUN pip install /code
