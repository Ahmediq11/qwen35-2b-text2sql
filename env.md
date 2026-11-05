# Environment (recorded 2026-11-04)

## Host
- OS: Windows 10 Pro 10.0.19045
- CPU: AMD Ryzen 5 3600 (6 cores / 12 threads)
- RAM: 15.9 GB (WSL2 VM sees ~7 GB by default)
- Disk: X: 131 GB free; WSL Ubuntu root 946 GB free

## GPU
- NVIDIA GeForce RTX 2060 SUPER, 8 GB VRAM, compute capability **7.5 (Turing)**
- Driver 621.01, CUDA UMD 13.5 (WDDM; the desktop already uses about 1.1 GB VRAM)
- **No native bf16.** Per PLAN §2, fall back to fp16 and check fp16 against bf16 on 20 prompts
  (the bf16 reference has to come from CPU or emulated bf16, which is slow).

## Python / tooling
- Windows: Python 3.14.3 (system), 3.11.16 (uv-managed), uv 0.12.7, git 2.43.0
- WSL2 distros: Ubuntu (Python 3.12.3, GPU visible), UbuntuAI, docker-desktop
- vLLM has no native Windows support, so generation will run inside WSL Ubuntu (to be confirmed).

## Status
- [x] vLLM 0.33.0 (torch 2.14.0+cu130, transformers 5.20.2) in WSL venv `~/wsp-venv` loads Qwen3.5-0.8B and
      generates: fp16, TRITON_ATTN backend (FA2 needs sm80+), FlashInfer sampler falls back. Needed `build-essential`
      in WSL for Triton.
- Measured: 151 tokens/s at 10 concurrent 32k-token sequences (pilot, 2026-11-04).
- fp16 vs bf16 20-prompt check: not done yet (no bf16 on this GPU; would need CPU bf16 reference).
