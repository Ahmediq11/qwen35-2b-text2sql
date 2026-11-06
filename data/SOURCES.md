# Data sources (verified 2026-11-06)

All files were downloaded on the Vast.ai box (`/workspace/wsp/data/raw`) by `scripts/remote_get_data.sh`. They are not in git.

| Dataset | Source page | Download | Size (bytes) | sha256 |
|---|---|---|---|---|
| Spider 1.0 | https://yale-lily.github.io/spider ("Spider Dataset") | Google Drive file id `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J` → `spider.zip` | 205,800,266 | `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b` |
| BIRD mini-dev | https://github.com/bird-bench/mini_dev ("Data Link") | https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip | 800,943,648 | `cc48ba16838204e4e214512030cb572eeb5f7bcdd999bae4b9b6ff12ec13b92f` |
| BIRD train | https://bird-bench.github.io/ ("Train Set") | https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip | 8,919,543,554 | `66e9e3115b59559554013aa3b124156249f30437a6b4e4f96de3d2dfb5ae8cbc` |
| GSM8K (Phase 0) | Hugging Face `openai/gsm8k`, config `main` | `main/test-00000-of-00001.parquet` | — | `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59` |

## Counts after unpacking (`python src/data.py summary data`)

| Split | Role | Items | Databases | Missing DBs |
|---|---|---|---|---|
| `spider_dev` (`dev.json`) | **locked test** | 1,034 | 20 | 0 |
| `spider_train` (`train_spider.json` + `train_others.json`) | practice | 8,659 (7,000 + 1,659) | 146 | 0 |
| `bird_minidev` (`MINIDEV/mini_dev_sqlite.json`) | **locked test** | 500 | 11 | 0 |
| `bird_train` (`train/train.json`) | practice | 9,428 | 69 | 0 |

These match the official sizes (Spider dev 1,034, BIRD mini-dev 500 SELECT-only SQLite questions over 11 databases). The
mini-dev zip is the original 500-question release (zip dated 2024-06-20), not the 780-question V2.

The zips contain `__MACOSX/` resource folders, which were deleted after unpacking.

## Notes from the gold self-check (`scripts/gold_selfcheck.py`, each gold query scored against itself)
- `spider_dev`: 1,034 / 1,034 correct.
- `bird_minidev`: 498 / 500 within 60 s. Gold queries `bird_minidev_518` and `bird_minidev_701` take longer than
  60 s, so the gold timeout is 300 s (predictions keep 30 s).
- `spider_train`: 8,656 / 8,659. Three gold queries fail in SQLite and are skipped as `gold_error`: 3153 (no such
  table `Ref_Company_Types`), 4513 and 4514 (ORDER BY before INTERSECT).

## Practice splits (`data/practice_splits.json`, `python src/data.py splits data data/practice_splits.json 0`)
These are split by database with a seeded hash, so no database is in both halves: `discover` has 8,960 items and
`confirm` has 9,127 items. 101 of the 215 practice databases are in `confirm`.
