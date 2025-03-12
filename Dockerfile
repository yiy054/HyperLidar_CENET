# Use NVIDIA L4T PyTorch base image compatible with Jetson Orin
# for nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth1.12-py3 not found: manifest unknown: manifest unknown
# FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth1.12-py3 
FROM dustynv/l4t-pytorch:r36.2.0
# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    git wget ffmpeg libsm6 libxext6 vim tmux python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Clone CENet repository
WORKDIR /root
RUN git clone --depth=1 https://github.com/huixiancheng/CENet

# Set working directory
WORKDIR /root/CENet

COPY . .
# Create and activate a Python virtual environment
RUN python3 -m venv cenet_env && \
    . cenet_env/bin/activate && \
    pip install --upgrade pip

# Install Python dependencies
RUN . cenet_env/bin/activate && \
    pip install -r requirements_cenet.txt

RUN . cenet_env/bin/activate && \
    pip install requests PyYaml==3.12 \
    pip install --upgrade pip wheel setuptools requests \
    pip install tensorboard==2.12.0 protobuf==3.20.3 
# git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch common

# Ensure the virtual environment is activated on login
RUN echo "source /root/CENet/cenet_env/bin/activate" >> /root/.bashrc

# Set entrypoint
CMD ["/bin/bash"]