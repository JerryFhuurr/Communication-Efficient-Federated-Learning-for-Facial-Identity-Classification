#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ROOT="$ROOT/outputs/adaptive-budget-confirmation-20261103"
OUT="$RUN_ROOT/seed-44-adaptive-retry1"
LOG="$RUN_ROOT/logs/seed-44-adaptive-retry1.log"
EXPECTED="$(tr -d '[:space:]' < docs/thesis/meeting-2026-11-10/confirmation-source-sha256.txt)"
ACTUAL="$(.venv/bin/python -c 'from flower_face.reproducibility import source_hash; print(source_hash())')"
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo "Frozen source mismatch" >&2; exit 2; }
mkdir -p "$OUT" "$RUN_ROOT/logs"
printf '[%s] START seed=44 method=adaptive-retry1 source=%s\n' "$(date -Is)" "$ACTUAL" | tee -a "$RUN_ROOT/progress.log"
.venv/bin/python -u -m flower_face.run \
  --rounds 15 \
  --manifest data/digiface-resnet18-layer3/manifest.json \
  --task-module flower_face.finetune_task \
  --num-classes 100 --num-clients 4 --image-size 224 \
  --seed 44 --lr 0.003 --weight-decay 0 \
  --augmentation horizontal-flip \
  --compression qsgd-llz-adaptive --qsgd-levels 127 \
  --llz-p 0 --llz-window 128 --llz-budget 5.9 \
  --skip-test --output-dir "$OUT" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if [[ "$rc" -ne 0 ]]; then
  printf '[%s] FAILED seed=44 method=adaptive-retry1 exit=%s\n' "$(date -Is)" "$rc" | tee -a "$RUN_ROOT/progress.log"
  exit "$rc"
fi
RESULT="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)/experiment.json"
if ! .venv/bin/python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text()); rows=[x for x in (p.parent/"history.jsonl").read_text().splitlines() if x.strip()]; raise SystemExit(0 if d.get("status")=="completed" and len(rows)==15 else 1)' "$RESULT"; then
  printf '[%s] FAILED seed=44 method=adaptive-retry1: experiment incomplete or missing rounds\n' "$(date -Is)" | tee -a "$RUN_ROOT/progress.log"
  exit 4
fi
AFTER="$(.venv/bin/python -c 'from flower_face.reproducibility import source_hash; print(source_hash())')"
[[ "$AFTER" == "$EXPECTED" ]] || { echo "Frozen source changed during retry" >&2; exit 3; }
printf '[%s] COMPLETE seed=44 method=adaptive-retry1 source=%s\n' "$(date -Is)" "$AFTER" | tee -a "$RUN_ROOT/progress.log"
