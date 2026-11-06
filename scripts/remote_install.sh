#!/usr/bin/env bash
# Install the generation stack into /workspace/wsp/venv; writes EXIT=<code> as the last line of install.log.
cd /workspace/wsp
{
  VIRTUAL_ENV=/workspace/wsp/venv uv pip install "vllm==0.33.0" "transformers==5.20.2" "math-verify[antlr4_13_2]" \
      huggingface_hub hf_transfer safetensors numpy pandas
  echo "EXIT=$?"
} > install.log 2>&1
