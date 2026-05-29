export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH

source /home/$USER/vllm_invariant/env

uv run vllm serve Qwen/Qwen3-8B-AWQ \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --max-num-seqs ${CONCURRENCY}\
  --enforce-eager \
  --override-generation-config '{"temperature": 0}' \
  --gpu-memory-utilization 0.90 >> /home/$USER/vllm_invariant/logs/vllm-Qwen3-8B-AWQ-32K-no-invariant.log 2>&1
