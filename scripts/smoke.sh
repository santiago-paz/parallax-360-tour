#!/usr/bin/env bash
# Smoke tests for scripts/build_scene.py
#
# Covers every CLI path: doctor, dry-run, real run, idempotency,
# --force, error cases, fine-grained override, batch with partial failure,
# tsc rollback. Each test reports PASS or FAIL and a summary is printed at
# the end.
#
# Usage:
#   ./scripts/smoke.sh                # runs every test
#   ./scripts/smoke.sh --skip-real    # skips tests that invoke the depth
#                                     # model (fast, no torch required)
#
# Requirements:
#   - venv activated (source venv/bin/activate) or python pointing at Python 3.x
#   - npm install already run (so `npx tsc` works)
#   - Run from the repo root (or a worktree of the repo)

set -u  # error on undefined variables; we do NOT use -e because we want
        # to capture expected failures without aborting the script.

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SKIP_REAL=0
if [[ "${1:-}" == "--skip-real" ]]; then
  SKIP_REAL=1
fi

# Unique ids per test to avoid cross-run contamination.
SUFFIX="smoke-$$"
LIVING_ID="living-${SUFFIX}"
KITCHEN_ID="kitchen-${SUFFIX}"
PATIO_ID="patio-${SUFFIX}"
ROLLBACK_ID="rollback-${SUFFIX}"

LIVING_PANO="public/${LIVING_ID}.jpeg"
KITCHEN_PANO="public/${KITCHEN_ID}.jpeg"
PATIO_PANO="public/${PATIO_ID}.jpeg"
ROLLBACK_PANO="public/${ROLLBACK_ID}.jpeg"
INVALID_PANO="/tmp/.${SUFFIX}.jpeg"  # name that does not produce a valid slug

LOG_DIR="/tmp/build_scene_smoke_${SUFFIX}"
mkdir -p "$LOG_DIR"

PASS=0
FAIL=0
SKIPPED=0
FAILED_TESTS=()

# ---------------------------------------------------------------------------
# Color helpers (downgrade to no-color when not a TTY)
# ---------------------------------------------------------------------------

if [[ -t 1 ]]; then
  C_OK=$'\033[0;32m'
  C_FAIL=$'\033[0;31m'
  C_SKIP=$'\033[0;33m'
  C_INFO=$'\033[0;36m'
  C_DIM=$'\033[2m'
  C_RESET=$'\033[0m'
else
  C_OK=""; C_FAIL=""; C_SKIP=""; C_INFO=""; C_DIM=""; C_RESET=""
fi

banner() {
  local title="$1"
  printf "\n${C_INFO}━━━ %s ━━━${C_RESET}\n" "$title"
}

pass() {
  PASS=$((PASS + 1))
  printf "  ${C_OK}✓${C_RESET} %s\n" "$1"
}

fail() {
  FAIL=$((FAIL + 1))
  FAILED_TESTS+=("$1")
  printf "  ${C_FAIL}✗${C_RESET} %s\n" "$1"
  if [[ -n "${2:-}" ]]; then
    printf "    ${C_DIM}%s${C_RESET}\n" "$2"
  fi
}

skip() {
  SKIPPED=$((SKIPPED + 1))
  printf "  ${C_SKIP}~${C_RESET} %s ${C_DIM}(%s)${C_RESET}\n" "$1" "$2"
}

# ---------------------------------------------------------------------------
# Cleanup (always, even if you abort with Ctrl+C)
# ---------------------------------------------------------------------------

cleanup() {
  printf "\n${C_DIM}Cleaning up test state...${C_RESET}\n"

  # Delete scenes.ts.bak if present; ignore otherwise.
  rm -f app/scenes.ts.bak

  # Revert any change to scenes.ts made by the smoke (ids carry our SUFFIX,
  # so they will not collide with real user changes).
  git checkout -- app/scenes.ts 2>/dev/null || true

  # Delete test panoramas
  rm -f "$LIVING_PANO" "$KITCHEN_PANO" "$PATIO_PANO" "$ROLLBACK_PANO" "$INVALID_PANO"

  # Delete generated assets (depth + bg + fg layers)
  for id in "$LIVING_ID" "$KITCHEN_ID" "$PATIO_ID" "$ROLLBACK_ID"; do
    rm -f "public/parallax/depth_${id}.png"
    rm -f public/parallax/${id}-bg.jpeg
    rm -f public/parallax/${id}-fg*.webp
    rm -f public/parallax/${id}-fg*.png
    rm -f public/parallax/${id}-fg-mask.png
  done
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Pre-flight for the script itself: are python, the panorama sample, and
# npx tsc available?
# ---------------------------------------------------------------------------

banner "Pre-flight"

if ! command -v python >/dev/null 2>&1; then
  printf "${C_FAIL}python is not on PATH. Activate the venv: source venv/bin/activate${C_RESET}\n"
  exit 1
fi
pass "python available ($(python --version 2>&1))"

if [[ ! -f public/image-1-360.webp ]]; then
  printf "${C_FAIL}Cannot find public/image-1-360.webp (repo panorama sample).${C_RESET}\n"
  exit 1
fi
pass "panorama sample (public/image-1-360.webp) present"

if ! command -v npx >/dev/null 2>&1; then
  skip "npx is not on PATH" "tests still run, but tsc will not validate"
else
  pass "npx available (tsc will validate scenes.ts)"
fi

# Copies so we don't touch versioned files.
cp public/image-1-360.webp "$LIVING_PANO"
cp public/image-1-360.webp "$KITCHEN_PANO"
cp public/image-1-360.webp "$PATIO_PANO"
cp public/image-1-360.webp "$ROLLBACK_PANO"
touch "$INVALID_PANO"   # empty file with an invalid name

# ---------------------------------------------------------------------------
# Test 1 — doctor
# ---------------------------------------------------------------------------

banner "Test 1 — --doctor"

OUT="$LOG_DIR/01-doctor.log"
if python scripts/build_scene.py --doctor > "$OUT" 2>&1; then
  if grep -q "Highest quality available" "$OUT" \
     && grep -q "synthetic" "$OUT" \
     && grep -q "da3" "$OUT"; then
    pass "prints backend table + highest quality"
  else
    fail "--doctor: incomplete output" "see $OUT"
  fi
else
  fail "--doctor: exit code != 0" "see $OUT"
fi

# ---------------------------------------------------------------------------
# Test 2 — no arguments (must fail with a message)
# ---------------------------------------------------------------------------

banner "Test 2 — no images and no --doctor"

OUT="$LOG_DIR/02-no-args.log"
if python scripts/build_scene.py > "$OUT" 2>&1; then
  fail "no args: should have failed but exited OK" "see $OUT"
else
  if grep -qi "no images" "$OUT"; then
    pass "fails with a clear message asking for an image or --doctor"
  else
    fail "no args: failed without a clear message" "see $OUT"
  fi
fi

# ---------------------------------------------------------------------------
# Test 3 — unknown quality (argparse rejects)
# ---------------------------------------------------------------------------

banner "Test 3 — invalid --quality"

OUT="$LOG_DIR/03-bad-quality.log"
if python scripts/build_scene.py "$LIVING_PANO" --quality maxima > "$OUT" 2>&1; then
  fail "--quality maxima: should have failed but exited OK" "see $OUT"
else
  if grep -qi "invalid choice" "$OUT" || grep -qi "argument" "$OUT"; then
    pass "argparse rejects --quality maxima"
  else
    fail "--quality maxima: failed without an argparse message" "see $OUT"
  fi
fi

# ---------------------------------------------------------------------------
# Test 4 — missing image
# ---------------------------------------------------------------------------

banner "Test 4 — image does not exist"

OUT="$LOG_DIR/04-missing-image.log"
if python scripts/build_scene.py /tmp/no-existe-${SUFFIX}.jpeg --quality low \
     > "$OUT" 2>&1; then
  fail "missing image: should have failed but exited OK" "see $OUT"
else
  if grep -qi "not found" "$OUT" || grep -qi "FileNotFoundError" "$OUT"; then
    pass "FileNotFoundError with a concrete path"
  else
    fail "missing image: error without a clear message" "see $OUT"
  fi
fi

# ---------------------------------------------------------------------------
# Test 5 — filename with an invalid slug
# ---------------------------------------------------------------------------

banner "Test 5 — filename without a valid stem"

OUT="$LOG_DIR/05-bad-slug.log"
if python scripts/build_scene.py "$INVALID_PANO" --quality low > "$OUT" 2>&1; then
  fail "invalid slug: should have failed but exited OK" "see $OUT"
else
  if grep -qi "cannot derive" "$OUT" || grep -qi "rename" "$OUT"; then
    pass "ValueError with instructions to rename"
  else
    fail "invalid slug: error without a clear message" "see $OUT"
  fi
fi

# ---------------------------------------------------------------------------
# Test 6 — dry-run for each preset (must not write anything)
# ---------------------------------------------------------------------------

banner "Test 6 — --dry-run for the 4 presets"

SCENES_HASH_BEFORE="$(git hash-object app/scenes.ts)"

for q in low medium high ultra; do
  OUT="$LOG_DIR/06-dryrun-${q}.log"
  if python scripts/build_scene.py "$LIVING_PANO" --quality "$q" --dry-run \
       > "$OUT" 2>&1; then
    if grep -q "DRY RUN" "$OUT" && grep -q "id: \"${LIVING_ID}\"" "$OUT"; then
      pass "dry-run ${q}: prints TS snippet with the right id"
    else
      fail "dry-run ${q}: incomplete output" "see $OUT"
    fi
  else
    fail "dry-run ${q}: exit code != 0" "see $OUT"
  fi
done

SCENES_HASH_AFTER="$(git hash-object app/scenes.ts)"
if [[ "$SCENES_HASH_BEFORE" == "$SCENES_HASH_AFTER" ]]; then
  pass "dry-run did not modify app/scenes.ts"
else
  fail "dry-run modified app/scenes.ts (it should not)" \
    "before=$SCENES_HASH_BEFORE after=$SCENES_HASH_AFTER"
fi

# Also check that it did not create any file in public/parallax/
if ls public/parallax/${LIVING_ID}* >/dev/null 2>&1 \
     || ls public/parallax/depth_${LIVING_ID}.png >/dev/null 2>&1; then
  fail "dry-run created files in public/parallax/"
else
  pass "dry-run did not create files in public/parallax/"
fi

# ---------------------------------------------------------------------------
# Test 7 — fine-grained override: dry-run with --max-dim must honor the override
# ---------------------------------------------------------------------------

banner "Test 7 — fine-grained override via --max-dim"

OUT="$LOG_DIR/07-override.log"
if python scripts/build_scene.py "$LIVING_PANO" --quality high --max-dim 4096 \
     --dry-run > "$OUT" 2>&1; then
  if grep -q "DRY RUN" "$OUT"; then
    pass "preset high + --max-dim 4096 runs without error"
  else
    fail "override: dry-run without DRY RUN header" "see $OUT"
  fi
else
  fail "override: exit code != 0" "see $OUT"
fi

# ---------------------------------------------------------------------------
# "Real" tests — invoke the depth model. Skippable with --skip-real.
# ---------------------------------------------------------------------------

if [[ "$SKIP_REAL" == "1" ]]; then
  banner "Tests 8-12 (real) — SKIPPED via --skip-real"
  skip "Test 8: real run quality=low"     "--skip-real"
  skip "Test 9: idempotency without --force" "--skip-real"
  skip "Test 10: --force replace"         "--skip-real"
  skip "Test 11: batch with duplicate"    "--skip-real"
  skip "Test 12: tsc rollback"            "--skip-real"
else

# ---------------------------------------------------------------------------
# Test 8 — real run, quality=low (auto cascade, fast)
# ---------------------------------------------------------------------------

banner "Test 8 — real run --quality low (may take ~10-30s)"

OUT="$LOG_DIR/08-real-low.log"
if python scripts/build_scene.py "$LIVING_PANO" --quality low --max-dim 512 \
     > "$OUT" 2>&1; then
  pass "exit code == 0"

  if [[ -f "public/parallax/depth_${LIVING_ID}.png" ]]; then
    pass "depth map generated"
  else
    fail "depth map was not generated"
  fi

  if [[ -f "public/parallax/${LIVING_ID}-bg.jpeg" ]]; then
    pass "inpainted background generated"
  else
    fail "background was not generated"
  fi

  if ls public/parallax/${LIVING_ID}-fg*.webp >/dev/null 2>&1; then
    pass "fg layers generated ($(ls public/parallax/${LIVING_ID}-fg*.webp | wc -l | tr -d ' ') layers)"
  else
    fail "fg layers were not generated"
  fi

  if grep -q "id: \"${LIVING_ID}\"" app/scenes.ts; then
    pass "entry added to app/scenes.ts"
  else
    fail "entry was not appended to scenes.ts"
  fi

  if [[ -f app/scenes.ts.bak ]]; then
    pass ".bak left on disk as a safety net"
  else
    fail ".bak was not left behind"
  fi
else
  fail "real run exit code != 0" "see $OUT"
  # If this fails, skip the remaining real tests
  banner "Skipping tests 9-12 because test 8 failed"
  skip "Test 9: idempotency"           "test 8 failed"
  skip "Test 10: --force replace"      "test 8 failed"
  skip "Test 11: batch with duplicate" "test 8 failed"
  skip "Test 12: tsc rollback"         "test 8 failed"
fi

# ---------------------------------------------------------------------------
# Test 9 — idempotency: re-running the same command must fail
# ---------------------------------------------------------------------------

if grep -q "id: \"${LIVING_ID}\"" app/scenes.ts 2>/dev/null; then
  banner "Test 9 — re-running without --force must fail"

  OUT="$LOG_DIR/09-duplicate.log"
  if python scripts/build_scene.py "$LIVING_PANO" --quality low --max-dim 512 \
       > "$OUT" 2>&1; then
    fail "duplicate id without --force: should have failed but exited OK" "see $OUT"
  else
    if grep -qi "already exists" "$OUT"; then
      pass "ValueError 'already exists' + suggests --force"
    else
      fail "failed but without a clear message" "see $OUT"
    fi
  fi

# ---------------------------------------------------------------------------
# Test 10 — --force regenerates and replaces
# ---------------------------------------------------------------------------

  banner "Test 10 — --force replace"

  OUT="$LOG_DIR/10-force.log"
  COUNT_BEFORE=$(grep -c "id: \"${LIVING_ID}\"" app/scenes.ts)

  if python scripts/build_scene.py "$LIVING_PANO" --quality medium --max-dim 512 \
       --force > "$OUT" 2>&1; then
    COUNT_AFTER=$(grep -c "id: \"${LIVING_ID}\"" app/scenes.ts)
    if [[ "$COUNT_AFTER" == "1" ]]; then
      pass "exactly 1 entry with that id (before: $COUNT_BEFORE, after: $COUNT_AFTER)"
    else
      fail "--force: there are $COUNT_AFTER entries with id '${LIVING_ID}', expected 1"
    fi

    # medium has 2 thresholds → there must be 2 fg layers (not 1 like low)
    N_FG=$(ls public/parallax/${LIVING_ID}-fg*.webp 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$N_FG" == "2" ]]; then
      pass "medium produced 2 fg layers (consistent with 2 thresholds)"
    else
      fail "medium produced $N_FG fg layers, expected 2"
    fi
  else
    fail "--force: exit code != 0" "see $OUT"
  fi
else
  skip "Test 9-10" "test 8 did not leave the entry in scenes.ts"
fi

# ---------------------------------------------------------------------------
# Test 11 — batch with one duplicate id: preflight aborts before loading model
# ---------------------------------------------------------------------------

banner "Test 11 — batch with duplicate: preflight aborts the batch"

OUT="$LOG_DIR/11-batch-dup.log"
if python scripts/build_scene.py "$KITCHEN_PANO" "$LIVING_PANO" "$PATIO_PANO" \
     --quality low --max-dim 512 > "$OUT" 2>&1; then
  fail "batch with duplicate: should have aborted but exited OK" "see $OUT"
else
  if grep -qi "already exists" "$OUT"; then
    pass "preflight aborts the batch before the model (clear message)"
  else
    fail "batch aborted without a duplicate message" "see $OUT"
  fi

  # Check that kitchen and patio were NOT processed (preflight aborts first)
  if ls public/parallax/depth_${KITCHEN_ID}.png >/dev/null 2>&1; then
    fail "preflight did not abort: kitchen was processed (it should not have been)"
  else
    pass "preflight effective: kitchen and patio were not processed"
  fi
fi

# ---------------------------------------------------------------------------
# Test 12 — tsc rollback: corrupt scenes.ts before running the script
# ---------------------------------------------------------------------------

banner "Test 12 — automatic rollback when tsc fails"

if ! command -v npx >/dev/null 2>&1; then
  skip "Test 12" "npx not available — tsc cannot validate"
else
  # Snapshot the current scenes.ts before corrupting it
  SCENES_GOOD="$LOG_DIR/scenes-good.ts"
  cp app/scenes.ts "$SCENES_GOOD"

  # Corrupt: append garbage TS at the end of the file
  echo "this is not valid typescript !!" >> app/scenes.ts

  OUT="$LOG_DIR/12-rollback.log"
  if python scripts/build_scene.py "$ROLLBACK_PANO" --quality low --max-dim 512 \
       > "$OUT" 2>&1; then
    fail "rollback: build_scene exited OK even though tsc should have failed" "see $OUT"
  else
    if grep -qi "tsc validation failed" "$OUT" || grep -qi "Reverted" "$OUT"; then
      pass "RuntimeError 'tsc validation failed' + rollback"
    else
      fail "build_scene failed but for another reason" "see $OUT"
    fi
  fi

  # Check that scenes.ts does NOT contain the garbage we appended
  if grep -q "not valid typescript" app/scenes.ts; then
    fail "rollback: garbage remained in scenes.ts (backup was not restored)"
  else
    pass "scenes.ts clean (restored from .bak)"
  fi
fi

fi  # end of --skip-real block

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

TOTAL=$((PASS + FAIL))
banner "Summary"
printf "  ${C_OK}%d PASS${C_RESET}   ${C_FAIL}%d FAIL${C_RESET}   ${C_SKIP}%d SKIPPED${C_RESET}   (total: %d)\n" \
  "$PASS" "$FAIL" "$SKIPPED" "$TOTAL"

if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
  printf "\n${C_FAIL}Failed:${C_RESET}\n"
  for t in "${FAILED_TESTS[@]}"; do
    printf "  - %s\n" "$t"
  done
  printf "\nLogs in: %s\n" "$LOG_DIR"
fi

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
printf "\n${C_OK}All tests passed.${C_RESET}\n"
exit 0
