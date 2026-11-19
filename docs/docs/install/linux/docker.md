---
title: "Docker"
description: "Run OpenFlowLM in Docker or Docker Compose on Linux with AMD XDNA2 NPU passthrough."
icon: "deployed_code"
weight: 20
toc: true
---

Run `oflm serve` inside an `x86_64` Linux container with `/dev/accel/accel0` NPU passthrough, persistent `./oflm-data` and `./npu-cache` volumes, and HTTP API access on `127.0.0.1:52625`.

## Host Prerequisites

Install the host `amdxdna` kernel driver and NPU firmware (`1.1.0.0+`) first so `/dev/accel/accel0` exists on the Linux host (see [Packages](/docs/install/linux/packages/) or [Nix & NixOS](/docs/install/linux/nix/)).

- **Official runtime image (`openflowlm:runtime`):** Bundles userspace `libxrt2` and `libxrt-npu2` inside the Ubuntu 26.04 container, so host userspace XRT libraries are not required.
- **Custom minimal containers:** If you build a custom image without in-container XRT packages, bind-mount host XRT (`-v /opt/xilinx/xrt:/opt/xilinx/xrt:ro`) and set `LD_LIBRARY_PATH=/opt/xilinx/xrt/lib`.
- **Memory locking:** Always grant `--cap-add=IPC_LOCK` and `--ulimit memlock=-1:-1` so XRT can pin NPU memory buffers.

## Build and Run with Docker Compose

1. Clone the repository with submodules and build the `.deb` package inside the containerized builder:

    ```bash
    git clone --recursive https://github.com/Atomic-Germ/OpenFlowLM-Next.git
    cd OpenFlowLM-Next
    ./build_in_docker.sh
    ```

    Keep a single `openflowlm*.deb` inside `build/packages/` before building the runtime image.

2. Build and start the `oflm serve` container with the repository `compose.yaml` (or save a standalone `docker-compose.yml` below):

    ```bash
    docker compose up -d --build
    ```

3. Verify that the HTTP server responds and lists registered models:

    ```bash
    curl --fail http://127.0.0.1:52625/api/version
    curl --fail http://127.0.0.1:52625/v1/models
    ```

### Example `docker-compose.yml`

```yaml
services:
  oflm:
    image: openflowlm:runtime
    build:
      context: .
      target: runtime
    init: true
    restart: unless-stopped
    devices:
      - "${OFLM_NPU_DEVICE:-/dev/accel/accel0}:/dev/accel/accel0"
    cap_add:
      - IPC_LOCK
    ulimits:
      memlock:
        soft: -1
        hard: -1
    environment:
      OFLM_MODEL_PATH: /root/.config/oflm
      NPU_CACHE_HOME: /root/.npu/cache
    ports:
      - "${OFLM_BIND_ADDRESS:-127.0.0.1}:${OFLM_PORT:-52625}:52625"
    volumes:
      - ./oflm-data:/root/.config/oflm
      - ./npu-cache:/root/.npu/cache
```

| `.env` Variable | Default | Purpose |
|---|---|---|
| `OFLM_BIND_ADDRESS` | `127.0.0.1` | Host bind address (`0.0.0.0` exposes on LAN) |
| `OFLM_PORT` | `52625` | Published host port |
| `OFLM_NPU_DEVICE` | `/dev/accel/accel0` | Host NPU character device passed to the container |

## Run with `docker run`

Once `openflowlm:runtime` is built (`docker build --target runtime -t openflowlm:runtime .`), you can run `oflm` directly without Docker Compose:

```bash
mkdir -p oflm-data npu-cache

# Validate NPU access inside the container
docker run --rm \
  --device=/dev/accel/accel0 \
  --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 \
  openflowlm:runtime validate

# Download and register a GGUF model into ./oflm-data
docker run --rm \
  --device=/dev/accel/accel0 \
  --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 \
  -v "$PWD/oflm-data:/root/.config/oflm" \
  -v "$PWD/npu-cache:/root/.npu/cache" \
  openflowlm:runtime add unsloth/Qwen3.5-4B-GGUF

# Start oflm serve in the background
docker run -d --name oflm \
  --restart unless-stopped \
  --device=/dev/accel/accel0 \
  --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 \
  -p 127.0.0.1:52625:52625 \
  -v "$PWD/oflm-data:/root/.config/oflm" \
  -v "$PWD/npu-cache:/root/.npu/cache" \
  openflowlm:runtime serve qwen3.5:4b --host 0.0.0.0 --port 52625
```

## Persistent Volumes and Model Management

Downloaded GGUF weights and `settings.json` persist in `./oflm-data` (`/root/.config/oflm`), while runtime NPU cache files persist in `./npu-cache` (`/root/.npu/cache`).

```bash
# Add or list models with Docker Compose
docker compose run --rm oflm add unsloth/Qwen3.5-4B-GGUF
docker compose exec oflm /opt/openflowlm/bin/oflm list

# Follow server logs or stop the container
docker compose logs -f oflm
docker compose down

# Verify the OpenAI-compatible models endpoint
curl http://127.0.0.1:52625/v1/models
```

Continue to [Server Mode](/docs/instructions/server/) or [CLI Usage](/docs/instructions/cli/).
