"""Download every model in src/generate.py MODELS at its pinned revision into /workspace/wsp/models.
Configs are scanned for auto_map (custom code): none of these repos should need trust_remote_code."""
import json
import os
import sys

sys.path.insert(0, "/workspace/wsp/src")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import snapshot_download  # noqa: E402

from generate import MODELS  # noqa: E402

for key, (path, family, repo, rev) in MODELS.items():
    d = snapshot_download(repo, revision=rev, local_dir=os.path.join("/workspace/wsp", path),
                          allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE", "*.model"])
    cfg = json.load(open(os.path.join(d, "config.json")))
    py = [f for f in os.listdir(d) if f.endswith(".py")]
    print(key, repo, rev[:7], cfg.get("architectures"), "auto_map:", cfg.get("auto_map"), "py files:", py)
