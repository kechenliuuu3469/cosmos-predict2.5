#!/bin/bash
#
# Unified eval pipeline: vary (experiment, action_loader, dataset(s), iters, split).
#
# One-time:
#   python -m eval.prepare_gt --action-loader <lam|oxe_ee_7dim> \
#       --datasets "droid:20 bridge:10 fmb:15" --split val
#
# Per run:
#   bash eval/run_eval.sh \
#       --experiment oxe10_lam \
#       --action-loader lam \
#       --datasets "droid:20 bridge:10 fmb:15" \
#       --split val \
#       --iters "2000 4000 8000"
#
# Output layout:
#   eval_outputs/<experiment>/<action_loader>/<split>/
#     <dataset>/iter_XXXXXXXXX/
#       generations/<ep>_chunk.mp4
#       per_episode.csv  per_timestep.npz  summary.json
#       videos/<ep>_side_by_side.mp4
#
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

EVAL_DIR="$PROJECT_DIR/eval"
EVAL_CACHE_ROOT="${EVAL_CACHE_ROOT:-$PROJECT_DIR/eval_cache}"
EVAL_OUTPUTS_ROOT="${EVAL_OUTPUTS_ROOT:-$PROJECT_DIR/eval_outputs}"
PARAMS_TEMPLATE="$EVAL_DIR/inference_params.template.json"

# ---- defaults --------------------------------------------------------------
EXPERIMENT=""
ACTION_LOADER=""
DATASETS=""
ITERS=""
SPLIT="val"
GUIDANCE="0"
SAVE_VIDEOS="1"
NUM_EPISODES_DEFAULT="10"
RUN_TAG=""

usage() {
    cat <<EOF
Usage: $0 --experiment <name> [--action-loader <name>]
          --datasets "ds1:N1 ds2[:N2] ..." --iters "I1 I2 ..." [--split val]
          [--guidance 0] [--save-videos 1] [--num-episodes 10] [--run-tag <tag>]

--experiment         registry key in eval/experiments.py
--action-loader      registry key in eval/action_loaders.py
                     (default: experiment's default_action_loader)
--datasets           space-separated selectors; use ':N' to override the per-dataset episode count
--iters              space-separated checkpoint iters
--split              val / train / test (default: val)
--guidance           classifier-free-guidance strength (default: 0)
--save-videos        1 to write side-by-side GT|Gen mp4s (default: 1)
--num-episodes       default episode count per dataset when ':N' omitted
--run-tag            optional suffix appended to iter dirs (lets you e.g. sweep guidance
                     without clobbering earlier runs: --guidance 3 --run-tag g3)

Environment:
  IMAGINAIRE_OUTPUT_ROOT  required
  EVAL_CACHE_ROOT         default $PROJECT_DIR/eval_cache
  EVAL_OUTPUTS_ROOT       default $PROJECT_DIR/eval_outputs
EOF
}

# ---- arg parsing -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)       EXPERIMENT="$2"; shift 2 ;;
        --action-loader)    ACTION_LOADER="$2"; shift 2 ;;
        --datasets)         DATASETS="$2"; shift 2 ;;
        --iters)            ITERS="$2"; shift 2 ;;
        --split)            SPLIT="$2"; shift 2 ;;
        --guidance)         GUIDANCE="$2"; shift 2 ;;
        --save-videos)      SAVE_VIDEOS="$2"; shift 2 ;;
        --num-episodes)     NUM_EPISODES_DEFAULT="$2"; shift 2 ;;
        --run-tag)          RUN_TAG="$2"; shift 2 ;;
        -h|--help)          usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$EXPERIMENT" || -z "$DATASETS" || -z "$ITERS" ]]; then
    echo "Missing required args." >&2
    usage
    exit 1
fi

: "${IMAGINAIRE_OUTPUT_ROOT:?IMAGINAIRE_OUTPUT_ROOT must be set}"

# ---- resolve experiment + action loader -----------------------------------
eval "$(python - "$EXPERIMENT" "$ACTION_LOADER" <<'PY'
import os, sys
from eval.experiments import get_experiment
exp_name, loader_cli = sys.argv[1], sys.argv[2]
exp = get_experiment(exp_name)
loader = loader_cli or exp["default_action_loader"]
root = os.environ["IMAGINAIRE_OUTPUT_ROOT"]
print(f"CKPTS_ROOT='{exp['ckpts_root'].format(IMAGINAIRE_OUTPUT_ROOT=root)}'")
print(f"EXP_CFG='{exp['experiment']}'")
print(f"CFG_FILE='{exp['config_file']}'")
print(f"ACTION_LOADER='{loader}'")
PY
)"

# ---- resolve dataset selection -> stdin "ds N" pairs -----------------------
SELECTION=$(python - "$DATASETS" "$NUM_EPISODES_DEFAULT" <<'PY'
import sys
tokens = sys.argv[1].split()
default_n = int(sys.argv[2])
for t in tokens:
    if ":" in t:
        n, c = t.split(":", 1); print(f"{n} {int(c)}")
    else:
        print(f"{t} {default_n}")
PY
)
if [[ -z "$SELECTION" ]]; then
    echo "Empty dataset selection." >&2
    exit 1
fi

echo "==============================================================="
echo "  experiment:     $EXPERIMENT  ($EXP_CFG)"
echo "  action loader:  $ACTION_LOADER"
echo "  split:          $SPLIT"
echo "  guidance:       $GUIDANCE"
echo "  ckpts root:     $CKPTS_ROOT"
echo "  selection:"
echo "$SELECTION" | sed 's/^/    /'
echo "==============================================================="

# Output roots per-run.
OUT_ROOT="$EVAL_OUTPUTS_ROOT/$EXPERIMENT/$ACTION_LOADER/$SPLIT"
mkdir -p "$OUT_ROOT"

build_params() {
    local ds="$1" n="$2" save_root="$3" split="$4" guidance="$5" loader="$6" out="$7"
    python - "$PARAMS_TEMPLATE" "$ds" "$n" "$save_root" "$split" "$guidance" "$loader" "$out" "$EVAL_CACHE_ROOT" <<'PY'
import json, sys
from pathlib import Path
from eval.datasets import get_spec
from eval.action_loaders import get_loader, DEFAULT_SCALERS
tmpl, ds, n, save_root, split, guidance, loader, out, cache_root = sys.argv[1:]
d = json.load(open(tmpl))
spec = get_spec(ds)
d["name"] = f"{loader}_{ds}_{split}"
d["input_root"] = cache_root
d["input_json_sub_folder"] = f"annotations/{loader}/{ds}/{split}"
d["save_root"] = save_root
d["end"] = int(n)
d["guidance"] = int(guidance)
d["fps_downsample_ratio"] = spec.fps_downsample_ratio
d["save_fps"] = spec.save_fps
d["action_load_fn"] = get_loader(loader)["fn"]
# 7-dim loaders need per-dataset action / gripper scalers + state keys.
if loader == "oxe_ee_7dim":
    scalers = DEFAULT_SCALERS.get(ds)
    if scalers is None:
        raise SystemExit(f"No 7-dim scalers registered for dataset {ds!r}. "
                         f"Add an entry in eval/action_loaders.py:DEFAULT_SCALERS.")
    d.update(scalers)
Path(out).write_text(json.dumps(d, indent=2))
PY
}

# ---- main loop -------------------------------------------------------------
for ITER_NUM in $ITERS; do
    ITER=$(printf "iter_%09d" "$ITER_NUM")
    CKPT_DIR="$CKPTS_ROOT/$ITER"
    ITER_TAG="$ITER"
    [[ -n "$RUN_TAG" ]] && ITER_TAG="${ITER}_${RUN_TAG}"

    echo
    echo "========================================================"
    echo "== $ITER_TAG"
    echo "========================================================"

    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "[skip] missing checkpoint dir: $CKPT_DIR"
        continue
    fi

    # DCP -> .pt (idempotent, shared across datasets).
    if [[ ! -f "$CKPT_DIR/model_ema_bf16.pt" ]]; then
        echo "[convert] $CKPT_DIR/model -> $CKPT_DIR/model_ema_bf16.pt"
        python scripts/convert_distcp_to_pt.py "$CKPT_DIR/model" "$CKPT_DIR"
    else
        echo "[convert] already done"
    fi

    while read -r DS N; do
        [[ -z "$DS" ]] && continue
        ANN_DIR="$EVAL_CACHE_ROOT/annotations/$ACTION_LOADER/$DS/$SPLIT"
        GT_DIR="$EVAL_CACHE_ROOT/gt_composite/$DS/$SPLIT"
        GEN_DIR="$OUT_ROOT/$DS/$ITER_TAG/generations"
        RES_DIR="$OUT_ROOT/$DS/$ITER_TAG"

        echo "-------- dataset=$DS (first $N) --------"

        if [[ ! -d "$ANN_DIR" ]] || [[ -z "$(ls -A "$ANN_DIR" 2>/dev/null)" ]]; then
            echo "[skip $DS] no annotations under $ANN_DIR"
            echo "           run: python -m eval.prepare_gt --action-loader $ACTION_LOADER --datasets $DS --split $SPLIT"
            continue
        fi
        if [[ ! -d "$GT_DIR" ]] || [[ -z "$(ls -A "$GT_DIR" 2>/dev/null)" ]]; then
            echo "[skip $DS] no GT cache under $GT_DIR"
            continue
        fi

        mkdir -p "$GEN_DIR"
        N_EP=$(find "$ANN_DIR" -maxdepth 1 -name '*.json' | wc -l)
        TARGET=$(( N < N_EP ? N : N_EP ))
        N_GEN=$(find "$GEN_DIR" -maxdepth 1 -name '*_chunk.mp4' | wc -l)
        if [[ "$N_GEN" -lt "$TARGET" ]]; then
            echo "[infer $DS] $N_GEN/$TARGET done, generating missing"
            PARAMS_TMP=$(mktemp --suffix=.json)
            build_params "$DS" "$TARGET" "$GEN_DIR" "$SPLIT" "$GUIDANCE" "$ACTION_LOADER" "$PARAMS_TMP"
            python examples/action_conditioned.py \
                -i "$PARAMS_TMP" \
                --output-dir "$GEN_DIR" \
                --config-file "$CFG_FILE" \
                --checkpoint-path "$CKPT_DIR/model_ema_bf16.pt" \
                --experiment "$EXP_CFG"
            rm -f "$PARAMS_TMP"
        else
            echo "[infer $DS] already done ($N_GEN/$TARGET)"
        fi

        if [[ ! -f "$RES_DIR/summary.json" ]]; then
            echo "[eval $DS] computing per-view metrics"
            mkdir -p "$RES_DIR"
            EVAL_ARGS=(
                --gen-dir "$GEN_DIR"
                --gt-dir "$GT_DIR"
                --out-dir "$RES_DIR"
                --iter "$ITER_NUM"
                --dataset "$DS"
            )
            if [[ "$SAVE_VIDEOS" == "1" ]]; then
                EVAL_ARGS+=(--save-videos)
            fi
            python -m eval.evaluate "${EVAL_ARGS[@]}"
        else
            echo "[eval $DS] already done"
        fi
    done <<< "$SELECTION"
done

echo
echo "Done. Aggregate across checkpoints:"
echo "    python -m eval.aggregate --results-root $OUT_ROOT"
