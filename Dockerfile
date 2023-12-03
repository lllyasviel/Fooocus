FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip git libgl1-mesa-glx libglib2.0-0
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements_versions.txt
RUN pip3 install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0
CMD ["python3", "entry_with_update.py", "--listen"]
