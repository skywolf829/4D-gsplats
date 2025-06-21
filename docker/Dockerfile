FROM dynamic-gsplats-base

# Install uv
#ADD https://astral.sh/uv/install.sh /uv-installer.sh
#RUN sh /uv-installer.sh && rm /uv-installer.sh
RUN curl -L https://astral.sh/uv/install.sh | sh
ENV UV_LINK_MODE=copy

# Copy and install project
WORKDIR /dynamic-gsplats
COPY . .

RUN uv venv --python=/opt/conda/bin/python && \
    uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128 && \
    uv sync --no-build-isolation

CMD ["bash"]
