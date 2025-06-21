FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
ENV MAMBA_VERSION=23.11.0-0
ENV PATH=/opt/conda/bin:/root/.local/bin:$PATH

# Install system deps and curl/git for downloading Mambaforge/uv
RUN apt-get update && \
    apt-get install -y curl git && \
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

# Set working directory and copy the project
WORKDIR /dynamic-gsplats
COPY . .

# Use uv to install deps + local package (editable)
RUN uv sync

# Optional: activate base env in bash sessions
RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

CMD ["bash"]
