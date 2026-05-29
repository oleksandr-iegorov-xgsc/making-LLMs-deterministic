# vllm_invariant — Setup Guide

This guide is complementary to youtube video https://www.youtube.com/watch?v=iNPlUeWWYDw

End-to-end setup for the batch-invariant vLLM determinism test rig.

- **OS:** Ubuntu 24.04 (noble), x86_64
- **GPU:** Blackwell / sm_120 (the launch scripts pass `TORCH_CUDA_ARCH_LIST="12.0"`)
- **Python:** 3.11 (managed by uv)
- **CUDA toolkit:** 13.0

---

## 1. Install CUDA Toolkit 13.0 (NVIDIA repo, Ubuntu 24.04)

Use NVIDIA's network (apt) repository. The keyring package wires up the repo and
GPG key, then you install the versioned toolkit metapackage.

```bash
# Add the NVIDIA CUDA apt repository for Ubuntu 24.04 / x86_64
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

sudo apt-get update

# Install ONLY the toolkit (compiler, libraries, headers) — not the driver.
sudo apt-get install -y cuda-toolkit-13-0
```

> **Note on the package name:** the toolkit metapackage uses dashes —
> `cuda-toolkit-13-0` — not `cuda-toolkit-13.0`. Installing `cuda` (without the
> `-toolkit` suffix) would also pull the NVIDIA *driver*; install that separately
> only if you don't already have a working driver.

This installs to `/usr/local/cuda-13.0` and creates the `/usr/local/cuda` →
`/usr/local/cuda-13.0` symlink. The launch scripts set `CUDA_HOME` and `PATH`
themselves, so a global profile edit is optional, but if you want `nvcc` on your
`PATH` in every shell:

```bash
echo 'export CUDA_HOME=/usr/local/cuda-13.0'      >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH'           >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
nvcc --version            # should report release 13.0
nvidia-smi                # driver present + GPU visible
```

> **Driver requirement:** CUDA 13.0 needs a recent NVIDIA driver. If `nvidia-smi`
> is missing or reports a too-old version, install/upgrade the driver, e.g.
> `sudo apt-get install -y cuda-drivers` (then reboot).

---

## 2. Install uv

The project is driven entirely by [uv](https://docs.astral.sh/uv/). Install it
if you don't have it (this rig was built with uv 0.7.6):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# restart your shell, then:
uv --version
```

---

## 3. Set up the `vllm_invariant` directory

This directory is **not** a uv *project* — there is no `pyproject.toml` or
`uv.lock`. It is a standalone `.venv` (Python 3.11) into which packages are
installed with `uv pip`. The `*.sh` scripts call `uv run …`, which automatically
uses the `.venv` in the current directory.

```bash
# Get the directory onto the target machine, e.g.:
cd /home/$USER
# git clone <repo-url> vllm_invariant   # (or copy the files over)
cd /home/$USER/vllm_invariant

# Create the virtual environment with Python 3.11 (uv will fetch it if needed)
uv venv --python 3.11

# Install the exact pinned package set (vllm 0.21.0, torch 2.11.0, etc.)
uv pip install -r requirements.txt
```

`requirements.txt` is a full `uv pip freeze` of the working environment (179
packages), so the install is reproducible. Key pins:

| Package        | Version       |
|----------------|---------------|
| vllm           | 0.21.0        |
| torch          | 2.11.0        |
| torchvision    | 0.26.0        |
| torchaudio     | 2.11.0        |
| transformers   | 5.9.0         |
| flashinfer     | 0.6.11.post3  |
| openai         | 2.38.0        |
| Python         | 3.11.12       |

> To regenerate the pin set after changing packages:
> `uv pip freeze > requirements.txt`

### `env` file

The scripts `source` an `env` file for shared settings. It currently holds the
batching concurrency used by both the server (`--max-num-seqs`) and the test
client (`--concurrency`):

```bash
CONCURRENCY=768
```

Create it if it's missing:

```bash
echo 'CONCURRENCY=768' > env
```

---

## 4. Create the `logs` directory

The launch scripts append server output to `logs/…log`. The directory must exist
first or the redirect will fail:

```bash
mkdir -p /home/$USER/vllm_invariant/logs
```

---

## 5. Run it

```bash
cd /home/$USER/vllm_invariant

# Start a server (pick one):
./launch_qwen3-8b-awq-32K-invariant.sh        # batch-invariant kernels ON
./launch_qwen3-8b-awq-32K-no-invariant.sh     # default kernels

# In another shell, once the server is up on :8000, run the determinism test:
./test_determinism.sh
./test_determinism_client_temperature_0.6.sh

# Stop the server:
./stop_vllm.sh
```

Server logs land in `logs/vllm-Qwen3-8B-AWQ-32K-{invariant,no-invariant}.log`.
