# Base image
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install linux packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    gcc git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup ultralytics
WORKDIR /usr/src/ultralytics
RUN git clone https://github.com/ultralytics/ultralytics -b main /usr/src/ultralytics
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt /usr/src/ultralytics/
RUN sed -i '/opencv-python/d' pyproject.toml
RUN python3 -m pip install --upgrade pip wheel \
    && pip install --no-cache-dir tqdm matplotlib pyyaml psutil pandas onnx "numpy==1.23" \
    && pip install --no-cache-dir -e .

# Set environment variables
ENV OMP_NUM_THREADS=1

