export CUDA_VISIBLE_DEVICES=0

vllm serve path/to/llama3-8b-instruct  \
  --dtype auto  \
  --port xxxx \
  --api-key token-abc123 \
  --gpu_memory_utilization 0.6