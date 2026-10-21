#!/usr/bin/env bash
set -e

PROJECT_DIR="$HOME/llm_project/imouto_v2/imoto"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

# 清理旧进程（可选，避免端口占用）
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "uvicorn server:app" 2>/dev/null || true
sleep 2

echo "Starting vLLM ..."
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve ./models/Qwen2.5-VL-3B-Instruct-AWQ \
    --served-model-name qwen \
    --host 127.0.0.1 \
    --port 8000 \
    --quantization awq \
    --gpu-memory-utilization 0.80 \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --enforce-eager \
    > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# 等 vLLM ready
echo "Waiting for vLLM on 127.0.0.1:8000 ..."
for i in $(seq 1 120); do
  if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "vLLM is up."
    break
  fi
  # vLLM 进程挂了就报错退出
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM died. Check $LOG_DIR/vllm.log"
    exit 1
  fi
  sleep 2
done

echo "Starting uvicorn ..."
uvicorn server:app --host 127.0.0.1 --port 8001 \
    > "$LOG_DIR/api.log" 2>&1 &
API_PID=$!
echo "uvicorn PID: $API_PID"

# 记录 PID 到文件，方便 stop 脚本用
echo "$VLLM_PID" > "$LOG_DIR/vllm.pid"
echo "$API_PID"  > "$LOG_DIR/api.pid"

echo ""
echo "Both started. Logs:"
echo "  tail -f $LOG_DIR/vllm.log"
echo "  tail -f $LOG_DIR/api.log"
echo ""
echo "Stop with:  ./stop.sh"

# 前台等待，Ctrl-C 会触发下面的清理
trap 'echo "Stopping ..."; kill $VLLM_PID $API_PID 2>/dev/null; exit 0' INT TERM
wait