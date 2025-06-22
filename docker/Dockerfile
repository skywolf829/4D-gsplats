FROM dynamic-gsplats-base

# install to venv
RUN --mount=type=bind,source=uv.lock,target=uv.lock,readwrite \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-build-isolation --no-install-project
RUN uv pip uninstall gsplat
RUN uv pip install --no-build-isolation "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"

# Copy and install project

WORKDIR /dynamic-gsplats
COPY . .
RUN uv sync
CMD ["bash"]
