FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt