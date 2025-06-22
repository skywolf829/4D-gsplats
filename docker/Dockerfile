FROM dynamic-gsplats-base

# install to venv
WORKDIR /dynamic-gsplats
RUN uv venv --python 3.12 ./venv
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
RUN uv pip uninstall gsplat
RUN uv pip install wheel ninja 
#RUN uv pip install --no-build-isolation "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"
ENV PATH=/dynamic-gsplats/bin:$PATH

# Copy and install project

COPY . .

CMD ["bash"]
