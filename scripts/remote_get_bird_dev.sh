#!/usr/bin/env bash
# BIRD dev (bird-bench.github.io "Dev Set") + XiYan-7B at its pinned revision.
#   setsid nohup bash scripts/remote_get_bird_dev.sh < /dev/null > bird_dev_get.log 2>&1 &
set -uo pipefail
cd /workspace/wsp
mkdir -p data/bird_dev
[ -s data/raw/bird_dev.zip ] || wget -q --tries=5 --timeout=60 -O data/raw/bird_dev.zip \
  https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
ls -l data/raw/bird_dev.zip; sha256sum data/raw/bird_dev.zip | tee -a data/raw/SHA256SUMS
unzip -q -o data/raw/bird_dev.zip -d data/bird_dev
find data/bird_dev -name 'dev_databases.zip' -exec unzip -q -o {} -d data/bird_dev \;
find data/bird_dev -name __MACOSX -type d -prune -exec rm -rf {} +
echo "bird_dev sqlite files: $(find data/bird_dev -name '*.sqlite' | wc -l)"
venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
d = snapshot_download("XGenerationLab/XiYanSQL-QwenCoder-7B-2504", revision="afe320fcaf18bee342174b4e35e5a481ae93719f",
                      local_dir="models/XiYanSQL-QwenCoder-7B-2504",
                      allow_patterns=["*.json", "*.txt", "*.safetensors"])
import os; print("xiyan-7b files:", sorted(os.listdir(d)))
EOF
echo GET_DONE
