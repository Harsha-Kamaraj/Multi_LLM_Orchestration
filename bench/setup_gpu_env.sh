#!/usr/bin/env bash
#
# Provision the serving stack that produced run 2026-08-13-c76a55d-4f4767.
#
# Every pin and flag here is load-bearing: each one is a failure that actually
# happened on this hardware, not a precaution. Run once, then use the commands
# in bench/README.md.
#
# Target: WSL2 (Ubuntu) with an NVIDIA card visible to `nvidia-smi`. Native
# Windows is not a target -- vLLM has no Windows build.
set -euo pipefail

VLLM_VERSION=0.11.0
TRANSFORMERS_VERSION=4.55.2
PY_VERSION=3.12
ENV_DIR="${VLLM_ENV:-$HOME/vllm-env}"

echo "=== 1. C toolchain ==="
# vLLM's server path runs torch.compile, and Inductor shells out to a C
# compiler. Without one the engine dies at startup with
#   InductorError: Failed to find C compiler
# The offline LLM() path with enforce_eager=True skips compile and hides this,
# so a passing offline smoke test does not prove the server will start.
command -v gcc >/dev/null || apt-get install -y build-essential

echo "=== 2. uv + Python $PY_VERSION ==="
# Ubuntu 26.04 ships Python 3.14, which has no vLLM wheels. uv fetches a
# standalone 3.12 rather than disturbing the system interpreter.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python "$PY_VERSION" "$ENV_DIR"

echo "=== 3. vLLM $VLLM_VERSION + transformers $TRANSFORMERS_VERSION ==="
# vLLM is pinned below 0.27: 0.27's GPU runner allocates through UvaBuffer and
# dies on WSL2 with "UVA is not available", which the paravirtualised GPU does
# not expose. There is no env override -- the constructor hard-raises.
#
# transformers is pinned below 5.x: 5.x removed
# Qwen2Tokenizer.all_special_tokens_extended, which vLLM 0.11 calls. Installing
# vLLM alone resolves a transformers that breaks it, so they are pinned together.
VIRTUAL_ENV="$ENV_DIR" uv pip install \
    "vllm==$VLLM_VERSION" "transformers==$TRANSFORMERS_VERSION" \
    numpy pyarrow anthropic

echo "=== 4. verify (allocates VRAM briefly) ==="
VLLM_USE_FLASHINFER_SAMPLER=0 "$ENV_DIR/bin/python" - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-Coder-1.5B-Instruct", max_model_len=2048,
          gpu_memory_utilization=0.35, dtype="bfloat16", enforce_eager=True)
out = llm.generate(["def add(a, b):"], SamplingParams(max_tokens=16, temperature=0))
print("generated:", repr(out[0].outputs[0].text[:60]))
print("SETUP_OK")
PY

cat <<'NOTES'

=== environment required at run time ===

  export VLLM_USE_FLASHINFER_SAMPLER=0

    flashinfer's sampler JIT-compiles with nvcc. With no CUDA toolkit installed
    the engine dies at
      RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
    Note this is the *sampler*, not attention -- VLLM_ATTENTION_BACKEND does not
    avoid it. vLLM's native sampler needs no toolchain.

  export HF_HUB_CACHE=$HOME/hf     # optional

    Staging weights on the Linux filesystem rather than /mnt/c cuts model load
    time noticeably. Any HF cache location works.

=== characterization server flags ===

  vllm serve <model> --port <p> --dtype bfloat16 --max-model-len 8192 \
      --gpu-memory-utilization 0.90 --no-enable-prefix-caching

  --no-enable-prefix-caching is REQUIRED for characterization and is the
  subtlest item on this page. The probe grid reuses one prompt per prefill
  target across every max_tokens and repeat. With caching on, every probe after
  the first hits a cached prefix and pays ~zero prefill, so prefill *length*
  varies across the design while prefill *cost* does not. The fit answers with a
  negative prefill coefficient -- a longer prompt making a request faster. It
  measured -0.185 ms/token here and failed the definition of done at R^2=0.708.
  With caching off and warmup on, the same fit is R^2=0.9994.

  --max-model-len 8192 covers this corpus: the longest prompt is 5158 tokens
  and GenParams.max_tokens is 2048.

  --gpu-memory-utilization 0.90 is the practical maximum on a 24GB card that is
  also driving a desktop; the compositor holds ~1.5GB that cannot be evicted.

=== one card ===

  Characterization measures latency at a declared concurrency. Do not run
  anything else on the GPU during it, and do not serve both arms at once to
  save time: contention makes wall-clock a queue-depth measurement, and the
  coefficients come out fitted to the queue rather than the model. Serve one
  arm, characterize it, tear down, serve the next.
NOTES
