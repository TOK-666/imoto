#!/usr/bin/env bash
LOG_DIR="$HOME/llm_project/imouto_v2/imoto/logs"

if [ -f "$LOG_DIR/vllm.pid" ]; then
  kill "$(cat $LOG_DIR/vllm.pid)" 2>/dev/null
  rm -f "$LOG_DIR/vllm.pid"
fi
if [ -f "$LOG_DIR/api.pid" ]; then
  kill "$(cat $LOG_DIR/api.pid)" 2>/dev/null
  rm -f "$LOG_DIR/api.pid"
fi

# 兜底
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "uvicorn server:app" 2>/dev/null || true

echo "Stopped."