#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ROOT="$ROOT/outputs/adaptive-budget-confirmation-20261103"
FINGERPRINT_FILE="$ROOT/docs/thesis/meeting-2026-11-10/confirmation-source-sha256.txt"
EXPECTED="$(tr -d '[:space:]' < "$FINGERPRINT_FILE")"
ACTUAL="$(.venv/bin/python -c 'from flower_face.reproducibility import source_hash; print(source_hash())')"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  printf 'Frozen source mismatch: expected %s, found %s\n' "$EXPECTED" "$ACTUAL" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT/logs"
printf 'Frozen source sha256: %s\nGit HEAD: ' "$ACTUAL" | tee "$RUN_ROOT/source-record.txt"
git rev-parse HEAD | tee -a "$RUN_ROOT/source-record.txt"
printf 'Manifest sha256: ' | tee -a "$RUN_ROOT/source-record.txt"
sha256sum data/digiface-resnet18-layer3/manifest.json | awk '{print $1}' | tee -a "$RUN_ROOT/source-record.txt"

run_one() {
  local seed="$1"
  local method="$2"
  local compression="$3"
  local output="$RUN_ROOT/seed-$seed-$method"
  local log="$RUN_ROOT/logs/seed-$seed-$method.log"
  local now
  now="$(date -Is)"
  printf '\n[%s] START seed=%s method=%s source=%s\n' "$now" "$seed" "$method" "$ACTUAL" | tee -a "$RUN_ROOT/progress.log"
  local -a method_args=()
  if [[ "$compression" == "qsgd-llz-adaptive" ]]; then
    method_args+=(--llz-p 0 --llz-window 128 --llz-budget 5.9)
  fi
  .venv/bin/python -u -m flower_face.run \
    --rounds 15 \
    --manifest data/digiface-resnet18-layer3/manifest.json \
    --task-module flower_face.finetune_task \
    --num-classes 100 --num-clients 4 --image-size 224 \
    --seed "$seed" --lr 0.003 --weight-decay 0 \
    --augmentation horizontal-flip \
    --compression "$compression" --qsgd-levels 127 \
    "${method_args[@]}" --skip-test --output-dir "$output" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  if [[ "$rc" -ne 0 ]]; then
    printf '[%s] FAILED seed=%s method=%s exit=%s\n' "$(date -Is)" "$seed" "$method" "$rc" | tee -a "$RUN_ROOT/progress.log"
    return "$rc"
  fi
  local result_dir
  result_dir="$(find "$output" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [[ -z "$result_dir" || ! -f "$result_dir/experiment.json" ]]; then
    printf '[%s] FAILED seed=%s method=%s: missing experiment manifest\n' "$(date -Is)" "$seed" "$method" | tee -a "$RUN_ROOT/progress.log"
    return 4
  fi
  if ! .venv/bin/python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text()); rows=[x for x in (p.parent/"history.jsonl").read_text().splitlines() if x.strip()]; raise SystemExit(0 if d.get("status")=="completed" and len(rows)==15 else 1)' "$result_dir/experiment.json"; then
    printf '[%s] FAILED seed=%s method=%s: experiment incomplete or missing rounds\n' "$(date -Is)" "$seed" "$method" | tee -a "$RUN_ROOT/progress.log"
    return 4
  fi
  local after
  after="$(.venv/bin/python -c 'from flower_face.reproducibility import source_hash; print(source_hash())')"
  if [[ "$after" != "$EXPECTED" ]]; then
    printf 'Source changed during run: expected %s, found %s\n' "$EXPECTED" "$after" >&2
    exit 3
  fi
  printf '[%s] COMPLETE seed=%s method=%s source=%s\n' "$(date -Is)" "$seed" "$method" "$after" | tee -a "$RUN_ROOT/progress.log"
}

run_one 43 qsgd qsgd
run_one 43 adaptive qsgd-llz-adaptive
run_one 44 qsgd qsgd
run_one 44 adaptive qsgd-llz-adaptive
printf '\nConfirmation batch completed: %s\n' "$(date -Is)" | tee -a "$RUN_ROOT/progress.log"
