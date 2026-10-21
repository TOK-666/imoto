#!/usr/bin/env bash
set -e

PROJECT_DIR="$HOME/llm_project/imouto_v2/imoto"
SESSION="imoto"

cd "$PROJECT_DIR"

# 如果已有同名 session，先杀掉
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

# 创建 session，第一个窗口跑 vLLM
tmux new-session -d -s "$SESSION" -n vllm
tmux send-keys -t "$SESSION:vllm" "cd $PROJECT_DIR" C-m
tmux send-keys -t "$SESSION:vllm" \
  "VLLM_USE_V2_MODEL_RUNNER=0 vllm serve ./models/Qwen2.5-VL-3B-Instruct-AWQ \
   --served-model-name qwen --host 127.0.0.1 --port 8000 \
   --quantization awq --gpu-memory-utilization 0.80 \
   --max-model-len 2048 --max-num-seqs 1 --enforce-eager" C-m

# 等 vLLM 起来（探测 8000 端口）
echo "Waiting for vLLM on 127.0.0.1:8000 ..."
for i in $(seq 1 120); do
  if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "vLLM is up."
    break
  fi
  sleep 2
done

# 新窗口跑 uvicorn
tmux new-window -t "$SESSION" -n api
tmux send-keys -t "$SESSION:api" "cd $PROJECT_DIR" C-m
tmux send-keys -t "$SESSION:api" \
  "uvicorn server:app --host 127.0.0.1 --port 8001" C-m

# 切到 vllm 窗口
tmux select-window -t "$SESSION:vllm"
tmux attach -t "$SESSION"