#!/bin/bash
#
# Cross-dataset eval pipeline for one or more checkpoint iters.
#
#   (one-time)  python -m video_cross_eval.prepare_val_data --spec video_cross_eval/eval_spec.json
#   (per run)   bash video_cross_eval/run_eval.sh 2000 4000 ... 30000
#   (train)     SPLIT=train bash video_cross_eval/run_eval.sh 2000 ...
#
# For each ITER, for each (dataset, N_episodes) in the selection, it:
#   1. Converts DCP -> model_ema_bf16.pt          (once per iter)
#   2. Runs inference over the first N val episodes of <dataset>
#          -> <gen_root>/<split>/<dataset>/iter_XXXXXXXXX/<ep>_chunk.mp4
#   3. Runs evaluate.py with per-view metrics
#          -> <res_root>/<split>/<dataset>/iter_XXXXXXXXX/{per_episode.csv,
#               per_timestep.npz, summary.json, videos/<ep>_side_by_side.mp4}
#
# Selection is taken from:
#   - $DATASETS   (space-separated "name:N" or just "name"), e.g.
#         DATASETS="droid:20 bridge:10 fmb:15" bash video_cross_eval/run_eval.sh 2000
#   - $SPEC_FILE  (JSON, same schema as prepare_val_data --spec)
#   Default: $SPEC_FILE = video_cross_eval/eval_spec.json
#
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ---- Configure for your cluster -------------------------------------------
: "${IMAGINAIRE_OUTPUT_ROOT:?IMAGINAIRE_OUTPUT_ROOT must be set}"
EXPERIMENT="${EXPERIMENT:-ac_reason_embeddings_rectified_flow_2b_256_320_oxe_lam}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py}"
CKPTS_ROOT="${CKPTS_ROOT:-$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe10_lam_action_conditioned/checkpoints}"
SPLIT="${SPLIT:-val}"
if [[ "$SPLIT" != "train" && "$SPLIT" != "val" ]]; then
    echo "SPLIT must be 'train' or 'val' (got: $SPLIT)" >&2
    exit 1
fi

EVAL_DIR="$PROJECT_DIR/video_cross_eval"
GEN_ROOT="${GEN_ROOT:-$EVAL_DIR/generations/$SPLIT}"
RES_ROOT="${RES_ROOT:-$EVAL_DIR/results/$SPLIT}"
PARAMS_TEMPLATE="$EVAL_DIR/inference_params.json"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
SPEC_FILE="${SPEC_FILE:-$EVAL_DIR/eval_spec.json}"
DATASETS="${DATASETS:-}"
# ----------------------------------------------------------------------------

mkdir -p "$GEN_ROOT" "$RES_ROOT"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <iter> [<iter> ...]"
    exit 1
fi

# Resolve dataset selection -> list of "name N" lines on stdout.
resolve_selection() {
    python - "$SPEC_FILE" "$DATASETS" <<'PY'
import json, sys
spec_path, cli = sys.argv[1], sys.argv[2]
sel = {}
if spec_path and __import__("os").path.exists(spec_path):
    d = json.load(open(spec_path))
    default = int(d.get("default_num_episodes", 10))
    for k, v in d.get("datasets", {}).items():
        sel[k] = default if v is None else int(v)
if cli.strip():
    # tokens like "name:N" or "name" (default 10)
    for tok in cli.split():
        if ":" in tok:
            n, c = tok.split(":", 1); sel[n] = int(c)
        else:
            sel.setdefault(tok, 10)
for name, n in sel.items():
    print(f"{name} {n}")
PY
}

SELECTION=$(resolve_selection)
if [[ -z "$SELECTION" ]]; then
    echo "Empty dataset selection. Set \$DATASETS or populate \$SPEC_FILE ($SPEC_FILE)." >&2
    exit 1
fi

echo "Eval selection:"
echo "$SELECTION" | sed 's/^/  /'

# Build one inference_params tmp file for a (dataset, n, save_root, split).
build_params() {
    local ds="$1" n="$2" save_root="$3" split="$4" out="$5"
    python - "$PARAMS_TEMPLATE" "$ds" "$n" "$save_root" "$split" "$out" <<'PY'
import json, sys
from pathlib import Path
tmpl, ds, n, save_root, split, out = sys.argv[1:]
d = json.load(open(tmpl))
from video_cross_eval.datasets import get_spec
spec = get_spec(ds)
d["name"] = f"cross_eval_{ds}_{split}"
d["input_root"] = f"video_cross_eval/{ds}/{split}"
d["save_root"] = save_root
d["end"] = int(n)
d["fps_downsample_ratio"] = spec.fps_downsample_ratio
d["save_fps"] = spec.save_fps
d["action_load_fn"] = "video_cross_eval.action_loader.load_lam_action_fn"
Path(out).write_text(json.dumps(d, indent=2))
PY
}

for ITER_NUM in "$@"; do
    ITER=$(printf "iter_%09d" "$ITER_NUM")
    CKPT_DIR="$CKPTS_ROOT/$ITER"

    echo "========================================================"
    echo "== $ITER (split=$SPLIT)"
    echo "========================================================"

    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "[skip] missing checkpoint dir: $CKPT_DIR"
        continue
    fi

    # -- 1. Convert DCP -> .pt (idempotent) ----------------------------------
    if [[ ! -f "$CKPT_DIR/model_ema_bf16.pt" ]]; then
        echo "[convert] $CKPT_DIR/model -> $CKPT_DIR/model_ema_bf16.pt"
        python scripts/convert_distcp_to_pt.py "$CKPT_DIR/model" "$CKPT_DIR"
    else
        echo "[convert] already done"
    fi

    while read -r DS N; do
        [[ -z "$DS" ]] && continue
        ANN_DIR="$EVAL_DIR/$DS/$SPLIT/annotations"
        GT_DIR="$EVAL_DIR/$DS/$SPLIT/gt_composite"
        GEN_DIR="$GEN_ROOT/$DS/$ITER"
        RES_DIR="$RES_ROOT/$DS/$ITER"

        echo "-------- dataset=$DS (first $N) --------"

        if [[ ! -d "$ANN_DIR" ]] || [[ -z "$(ls -A "$ANN_DIR" 2>/dev/null)" ]]; then
            echo "[skip $DS] no annotations under $ANN_DIR — run prepare_val_data first"
            continue
        fi

        # -- 2. Inference (skip if all chunk mp4s already exist) ------------
        mkdir -p "$GEN_DIR"
        N_EP=$(find "$ANN_DIR" -maxdepth 1 -name '*.json' | wc -l)
        TARGET=$(( N < N_EP ? N : N_EP ))
        N_GEN=$(find "$GEN_DIR" -maxdepth 1 -name '*_chunk.mp4' | wc -l)
        if [[ "$N_GEN" -lt "$TARGET" ]]; then
            echo "[infer $DS] $N_GEN/$TARGET done, generating missing"
            PARAMS_TMP=$(mktemp --suffix=.json)
            build_params "$DS" "$TARGET" "$GEN_DIR" "$SPLIT" "$PARAMS_TMP"
            python examples/action_conditioned.py \
                -i "$PARAMS_TMP" \
                --output-dir "$GEN_DIR" \
                --config-file "$CONFIG_FILE" \
                --checkpoint-path "$CKPT_DIR/model_ema_bf16.pt" \
                --experiment "$EXPERIMENT"
            rm -f "$PARAMS_TMP"
        else
            echo "[infer $DS] already done ($N_GEN/$TARGET)"
        fi

        # -- 3. Metrics -----------------------------------------------------
        if [[ ! -f "$RES_DIR/summary.json" ]]; then
            echo "[eval $DS] computing metrics"
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
            python -m video_cross_eval.evaluate "${EVAL_ARGS[@]}"
        else
            echo "[eval $DS] already done"
        fi
    done <<< "$SELECTION"
done

echo
echo "All done. Aggregate across checkpoints:"
echo "    python -m video_cross_eval.aggregate --results-root $RES_ROOT"
