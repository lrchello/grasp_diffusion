FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt update && apt install -y \
    python3 python3-distutils python3-dev python3-pip \
    git wget curl && \
    rm -rf /var/lib/apt/lists/*

# faster pip
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# install pytorch first (not in requirements)
RUN pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

WORKDIR /workspace
COPY . /workspace

# install dependencies
RUN pip install --no-cache-dir -r requirements_clean.txt

CMD ["/bin/bash"]

