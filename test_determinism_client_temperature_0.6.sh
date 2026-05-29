source /home/$USER/vllm_invariant/env
uv run python test_determinism.py --n 1000 --concurrency ${CONCURRENCY} --show-answers --temperature 0.6
