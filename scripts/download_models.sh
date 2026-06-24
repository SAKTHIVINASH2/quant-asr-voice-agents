#!/usr/bin/env bash
# Download GGUF models at several quant levels + a Piper voice.
# Uses huggingface-cli. Edit REPOS/quant lists for your chosen models.
#
#   pip install huggingface_hub
#   bash scripts/download_models.sh
set -euo pipefail

MODELS_DIR="${VA_MODELS_DIR:-$(dirname "$0")/../models}"
mkdir -p "$MODELS_DIR"

# --- Target LLM: Qwen2.5-7B-Instruct GGUF (one file per quant) ---------------
# Bartowski publishes per-quant GGUF files. File names follow:
#   Qwen2.5-7B-Instruct-<QUANT>.gguf
TARGET_REPO="bartowski/Qwen2.5-7B-Instruct-GGUF"
for q in Q8_0 Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  echo ">> target $q"
  huggingface-cli download "$TARGET_REPO" "Qwen2.5-7B-Instruct-${q}.gguf" \
    --local-dir "$MODELS_DIR" --local-dir-use-symlinks False || \
    echo "   (skip $q — check exact filename on the repo)"
  # normalize to the name src/config.py expects:
  [ -f "$MODELS_DIR/Qwen2.5-7B-Instruct-${q}.gguf" ] && \
    mv -f "$MODELS_DIR/Qwen2.5-7B-Instruct-${q}.gguf" \
          "$MODELS_DIR/qwen2.5-7b-instruct-${q}.gguf"
done

# --- Draft LLM (RQ3): Qwen2.5-0.5B-Instruct GGUF ----------------------------
DRAFT_REPO="bartowski/Qwen2.5-0.5B-Instruct-GGUF"
for q in Q8_0 Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  echo ">> draft $q"
  huggingface-cli download "$DRAFT_REPO" "Qwen2.5-0.5B-Instruct-${q}.gguf" \
    --local-dir "$MODELS_DIR" --local-dir-use-symlinks False || \
    echo "   (skip $q)"
  [ -f "$MODELS_DIR/Qwen2.5-0.5B-Instruct-${q}.gguf" ] && \
    mv -f "$MODELS_DIR/Qwen2.5-0.5B-Instruct-${q}.gguf" \
          "$MODELS_DIR/qwen2.5-0.5b-instruct-${q}.gguf"
done

# --- Piper TTS voice --------------------------------------------------------
echo ">> piper voice en_US-amy-medium"
huggingface-cli download rhasspy/piper-voices \
  en/en_US/amy/medium/en_US-amy-medium.onnx \
  en/en_US/amy/medium/en_US-amy-medium.onnx.json \
  --local-dir "$MODELS_DIR" --local-dir-use-symlinks False || true
find "$MODELS_DIR" -name 'en_US-amy-medium.onnx*' -exec mv -f {} "$MODELS_DIR/" \; 2>/dev/null || true

# faster-whisper downloads its model automatically on first use (config.ASR_MODEL).
echo "Done. Models in $MODELS_DIR"
