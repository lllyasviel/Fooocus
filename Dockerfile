FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip git libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app
RUN pip3 install --no-cache-dir -r requirements_versions.txt \
    torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/cu121
RUN useradd -u 1000 -U -d /app -s /bin/false fooocus && \
    usermod -G users fooocus && \
    chown -R 1000:1000 /app
USER fooocus
CMD ["python3", "entry_with_update.py", "--listen"]
