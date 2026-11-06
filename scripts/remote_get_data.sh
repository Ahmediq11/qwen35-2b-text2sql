#!/usr/bin/env bash
# Download Spider 1.0 and BIRD (mini-dev + train) on the GPU box into /workspace/wsp/data, record sizes and sha256.
# URLs verified 2026-11-06 against the official pages (see data/SOURCES.md).
#   setsid nohup bash scripts/remote_get_data.sh < /dev/null > data_get.log 2>&1 &
set -uo pipefail
cd /workspace/wsp
PY=/workspace/wsp/venv/bin/python
mkdir -p data/raw data/spider data/bird_minidev data/bird_train
VIRTUAL_ENV=/workspace/wsp/venv uv pip install -q gdown

get() {  # url out
  [ -s "$2" ] && { echo "have $2"; return 0; }
  wget -q --tries=5 --timeout=60 --continue -O "$2.part" "$1" && mv "$2.part" "$2" && echo "got $2"
}

# Spider 1.0 (Yale LILY page -> Google Drive)
[ -s data/raw/spider.zip ] || $PY -m gdown -O data/raw/spider.zip 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J   # Drive file id from the Spider page
# BIRD mini-dev (github.com/bird-bench/mini_dev "Data Link") and BIRD train (bird-bench.github.io)
get https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip data/raw/minidev.zip
get https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip data/raw/bird_train.zip

ls -l data/raw
sha256sum data/raw/*.zip | tee data/raw/SHA256SUMS

unzip -q -o data/raw/spider.zip -d data/spider
unzip -q -o data/raw/minidev.zip -d data/bird_minidev
unzip -q -o data/raw/bird_train.zip -d data/bird_train
# BIRD train ships its databases as a nested zip
find data/bird_train -name 'train_databases.zip' -exec unzip -q -o {} -d data/bird_train \;
find data/bird_minidev -name '*.zip' -path '*dev_databases*' -exec unzip -q -o {} -d data/bird_minidev \;

echo "--- tree"
find data/spider data/bird_minidev data/bird_train -maxdepth 3 -not -name '*.sqlite' | grep -vE '/database/[^/]+/' | head -60
echo "--- sqlite counts"
for d in data/spider data/bird_minidev data/bird_train; do echo "$d: $(find $d -name '*.sqlite' | wc -l) sqlite files"; done
echo DATA_DONE
