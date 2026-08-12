# ─────────────────────────────────────────────────────────────────────────────
# ICLScan on RCP — minimal image (option A)
# Base: official PyTorch 2.5.1 + CUDA 12.4 (matches ICLScan's requirements.txt)
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# --- my RCP identity (bakes UID/GID so NFS scratch is writable by me) ----------
ARG USER_NAME=peechara
ARG USER_UID=341964
ARG USER_GID=11259
ARG GROUP_NAME=SACS-StaffU

# --- system packages: git (clone repos), build tools (some pip pkgs compile) ---
# DEBIAN_FRONTEND avoids interactive prompts hanging the build.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ninja-build \
        sudo \
        vim \
        ca-certificates \
        pkg-config \
        cmake \
    && rm -rf /var/lib/apt/lists/*

# --- create a user that matches my EPFL UID/GID ------------------------------
# This is what makes writes to /mnt/sacs/scratch/home/peechara work.
RUN groupadd -g ${USER_GID} ${GROUP_NAME} \
    && useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USER_NAME} \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME} \
    && chmod 0440 /etc/sudoers.d/${USER_NAME}

# --- python deps -------------------------------------------------------------
# torch/torchvision/torchaudio already come from the base image at the right
# CUDA build, so we DON'T reinstall them (avoids clobbering the working CUDA
# torch with a pip wheel). We install ICLScan's other requirements.
#
# We copy a FILTERED requirements file (see build steps) that has torch* and the
# two source-compiled packages removed, then add those two separately with flags
# that keep the build from exploding.
COPY requirements.docker.txt /tmp/requirements.docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements.docker.txt

# flash-attn and auto-gptq compile CUDA extensions. Install them last, isolated,
# so if one fails you can see exactly which — and the rest of the image is fine.
# --no-build-isolation makes flash-attn use the already-installed torch.
RUN pip install --no-cache-dir ninja packaging \
    && pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation || \
       echo "WARNING: flash-attn install failed — continuing without it"

# auto-gptq is only needed if you run GPTQ-quantized models; allow it to fail soft.
RUN pip install --no-cache-dir auto_gptq==0.7.1 || \
       echo "WARNING: auto_gptq install failed — continuing without it"

# --- default to my user ------------------------------------------------------
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

CMD ["/bin/bash"]
