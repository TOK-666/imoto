vllm serve ./models/Qwen2.5-VL-3B-Instruct \
  --served-model-name qwen \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192

uvicorn server:app --host 127.0.0.1 --port 8001

vllm serve ./models/Qwen2.5-VL-3B-Instruct-AWQ \
    --served-model-name qwen \
    --host 127.0.0.1 \
    --port 8000 \
    --quantization bitsandbytes \
    --gpu-memory-utilization 0.80 \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --enforce-eager

VLLM_USE_V2_MODEL_RUNNER=0 \
vllm serve ./models/Qwen2.5-VL-3B-Instruct-AWQ \
    --served-model-name qwen \
    --host 127.0.0.1 \
    --port 8000 \
    --quantization awq \
    --gpu-memory-utilization 0.80 \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --enforce-eager