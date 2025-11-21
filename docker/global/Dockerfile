# Use the official OpenVINO runtime image with Intel GPU support
FROM openvino/ubuntu22_runtime:latest

USER root

# Install libgl for opencv support & Noto fonts for Chinese characters
RUN apt-get update && \
    apt-get install -y \
        wget \
        gnupg \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 \
        libglib2.0-0 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add Intel oneAPI APT repository and install the Base Toolkit + PTI (for libpti_view)
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        intel-oneapi-base-toolkit \
        intel-oneapi-runtime-libs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Intel PyTorch XPU wheels (torch/vision/audio from the Intel index)
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/xpu \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 && \
    python3 -m pip cache purge

# Ensure transformers is present for the VLM backend
RUN python3 -m pip install --no-cache-dir transformers && \
    python3 -m pip cache purge

# Install mineru with OpenVINO ONNX Runtime EP
RUN python3 -m pip install --no-cache-dir onnxruntime-openvino 'mineru[core]' --break-system-packages && \
    python3 -m pip cache purge

# Download models and update the configuration file
RUN /bin/bash -c "mineru-models-download -s huggingface -m all"

# Set the entry point to preload OpenVINO + oneAPI environments
ENV MINERU_MODEL_SOURCE=local
ENTRYPOINT ["/bin/bash", "-lc", "source /opt/intel/openvino/setupvars.sh && source /opt/intel/oneapi/setvars.sh && exec \"$@\"", "--"]
