#!/usr/bin/env bash
# Self-contained overnight pipeline: smoke test -> cost projection guard ->
# six generation arms on the 400-question test sample -> judges -> summary.
# Every stage is cached, so re-running after a failure resumes where it left off.
# Stops if the recorded spend exceeds $CAP at any checkpoint.
#
#   nohup caffeinate -i bash scripts/overnight.sh > reports/overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=./venv/bin/python
CAP=${CAP:-16}
SAMPLE=data/processed/test_sample_400.json
FT=paragraph-256-0_bge-small-ft_dense_norerank_doc
BASE=paragraph-256-0_bge-small_hybrid_norerank_doc
BIG=fixed-512-0_bge-small_hybrid_norerank_doc
DRY=${DRY:-0}

log() { echo "[$(date '+%H:%M:%S')] $*"; }
run() { log "\$ $*"; if [ "$DRY" = 1 ]; then return 0; fi; "$@" 2>&1 | grep -v -iE "warning|Loading weights|Batches:"; return "${PIPESTATUS[0]}"; }
guard() { log "spend check"; [ "$DRY" = 1 ] && return 0; $PY scripts/cost_ledger.py --cap "$CAP" || { log "STOPPED: over cap"; exit 1; }; }

log "=== overnight start (cap \$$CAP) ==="
[ -f .env ] || { log "no .env; run scripts/set_api_key.py first"; exit 1; }

# 0. smoke test (30 questions, synchronous, full price but tiny)
run $PY scripts/run_generation.py --split test --config $FT --model sonnet-5 --mode rag --top-k 5 \
    --effort low --sample $SAMPLE --limit 30 --sync --tag __smoke || exit 1
SMOKE=data/processed/generation/test/sonnet-5/rag__${FT}__k5__smoke.parquet
run $PY scripts/run_judge.py --split test --generation $SMOKE --no-judge || exit 1
[ "$DRY" = 1 ] || $PY scripts/project_cost.py --smoke $SMOKE --n 400 --cap "$CAP" || exit 1
guard

# 1. generation arms (Batches API, 50% off), sequential
run $PY scripts/run_generation.py --split test --config $FT   --model sonnet-5  --mode rag --top-k 5 --effort low --sample $SAMPLE || exit 1
guard
run $PY scripts/run_generation.py --split test --config $BASE --model sonnet-5  --mode rag --top-k 5 --effort low --sample $SAMPLE || exit 1
run $PY scripts/run_generation.py --split test --config $FT   --model haiku-4.5 --mode rag --top-k 5 --effort low --sample $SAMPLE || exit 1
guard
run $PY scripts/run_generation.py --split test --config none  --model sonnet-5  --mode closed-book --effort low --sample $SAMPLE || exit 1
run $PY scripts/run_generation.py --split test --config paragraph-256-0 --model sonnet-5 --mode oracle --effort low --sample $SAMPLE || exit 1
guard

# 2. judges (Opus, low effort): correctness on all arms; faithfulness judge on
#    the primary arm (others optional below); NLI faithfulness everywhere (free)
G=data/processed/generation/test
run $PY scripts/run_judge.py --split test --generation $G/sonnet-5/rag__${FT}__k5.parquet || exit 1
guard
run $PY scripts/run_judge.py --split test --generation $G/sonnet-5/rag__${BASE}__k5.parquet --no-faith-judge || exit 1
run $PY scripts/run_judge.py --split test --generation $G/haiku-4.5/rag__${FT}__k5.parquet --no-faith-judge || exit 1
run $PY scripts/run_judge.py --split test --generation $G/sonnet-5/closed-book__none__k5.parquet --no-faith-judge || exit 1
run $PY scripts/run_judge.py --split test --generation $G/sonnet-5/oracle__paragraph-256-0__k5.parquet --no-faith-judge || exit 1
guard

# 3. core summary (so results exist even if the optional stages stop)
run $PY scripts/summarize_generation.py --split test --primary "sonnet-5/rag__${FT}__k5" || exit 1
log "=== core done ==="

# 4. optional stages, each gated on the cap
run $PY scripts/run_judge.py --split test --generation $G/sonnet-5/rag__${BASE}__k5.parquet || exit 1
guard
run $PY scripts/run_judge.py --split test --generation $G/haiku-4.5/rag__${FT}__k5.parquet || exit 1
guard
run $PY scripts/run_generation.py --split test --config $BIG  --model sonnet-5  --mode rag --top-k 5 --effort low --sample $SAMPLE || exit 1
run $PY scripts/run_judge.py --split test --generation $G/sonnet-5/rag__${BIG}__k5.parquet --no-faith-judge || exit 1
run $PY scripts/summarize_generation.py --split test --primary "sonnet-5/rag__${FT}__k5" || exit 1
$PY scripts/cost_ledger.py
log "=== overnight done ==="
