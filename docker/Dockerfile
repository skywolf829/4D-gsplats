FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
ENV MAMBA_VERSION=23.11.0-0
ENV PATH=/opt/conda/bin:/root/.local/bin:$PATH
ENV CUDA_ARCHITECTURES=75;80;86;89
ENV TORCH_CUDA_ARCH_LIST="8.9 8.6 6.0 7.5"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
ENV TORCH_HOME=/tmp/torch_cache
# Install system deps and curl/git for downloading Mambaforge/uv
RUN apt-get update && \
    apt-get install -y curl git build-essential ninja-build && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Mambaforge (Miniconda + Mamba)
RUN curl -L -o mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-Linux-x86_64.sh && \
    bash mambaforge.sh -b -p /opt/conda && \
    rm mambaforge.sh

# Install Python 3.12 in base env
RUN mamba install -y python=3.12 -c conda-forge && \
    mamba clean -a -y

# Install uv from Astral
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV UV_LINK_MODE=copy

# Set working directory and copy the project
WORKDIR /dynamic-gsplats
COPY . .

# Use uv to install deps + local package (editable)
RUN uv venv --python=/opt/conda/bin/python && \
    uv sync
RUN uv pip install --no-build-isolation "vggt @ git+https://github.com/facebookresearch/vggt.git@8e3d2809c1c93fa27298c290605841d7e876b285"
RUN uv pip install --no-build-isolation "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim.git@8bdb59feb7b9a41b1fab625907cb21f5417deaac"
RUN uv pip install --no-build-isolation "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"


# Optional: activate base env in bash sessions
RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

CMD ["bash"]
