#!/bin/bash
#
# Full eval pipeline for a list of checkpoint iterations.
#
#   (one-time)  python -m video_model_eval.prepare_val_data
#   (per run)   bash  video_model_eval/run_eval.sh  2000 4000 6000 ... 30000
#
# For each ITER it:
#   1. Converts DCP -> model_ema_bf16.pt       (skipped if already there)
#   2. Runs inference on val -> generations/iter_XXXXXXXX/<ep>_chunk.mp4
#      (skipped if every annotation already has a matching _chunk.mp4)
#   3. Runs evaluate.py -> results/iter_XXXXXXXX/{per_episode.csv,per_timestep.npz,summary.json}
#
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ---- Configure for your cluster --------------------------------------------
: "${IMAGINAIRE_OUTPUT_ROOT:?IMAGINAIRE_OUTPUT_ROOT must be set}"
EXPERIMENT="ac_reason_embeddings_rectified_flow_2b_256_320_oxe_lam"
CONFIG_FILE="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
CKPTS_ROOT="$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_oxe10_lam_action_conditioned/checkpoints"
EVAL_DIR="$PROJECT_DIR/video_model_eval"
GEN_ROOT="$EVAL_DIR/generations"
RES_ROOT="$EVAL_DIR/results"
GT_DIR="$EVAL_DIR/val_inference_droid/gt_composite"
PARAMS_TEMPLATE="$EVAL_DIR/inference_params.json"
# ----------------------------------------------------------------------------

mkdir -p "$GEN_ROOT" "$RES_ROOT"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <iter> [<iter> ...]"
    exit 1
fi

for ITER_NUM in "$@"; do
    ITER=$(printf "iter_%09d" "$ITER_NUM")
    CKPT_DIR="$CKPTS_ROOT/$ITER"
    GEN_DIR="$GEN_ROOT/$ITER"
    RES_DIR="$RES_ROOT/$ITER"

    echo "========================================================"
    echo "== $ITER"
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

    # -- 2. Inference (skip if all chunk mp4s already exist) -----------------
    ANN_DIR="$EVAL_DIR/val_inference_droid/annotations"
    mkdir -p "$GEN_DIR"
    N_EP=$(find "$ANN_DIR" -name '*.json' | wc -l)
    N_GEN=$(find "$GEN_DIR" -name '*_chunk.mp4' | wc -l)
    # Also cap by --end in inference_params.json so "target" matches what we actually generate
    TARGET_END=$(python -c "import json; print(json.load(open('$PARAMS_TEMPLATE'))['end'])")
    TARGET=$(( TARGET_END < N_EP ? TARGET_END : N_EP ))
    if [[ "$N_GEN" -lt "$TARGET" ]]; then
        echo "[infer] $N_GEN/$TARGET done, generating missing"
        PARAMS_TMP=$(mktemp --suffix=.json)
        sed "s|__SAVE_ROOT__|$GEN_DIR|g" "$PARAMS_TEMPLATE" > "$PARAMS_TMP"
        python examples/action_conditioned.py \
            -i "$PARAMS_TMP" \
            --output-dir "$GEN_DIR" \
            --config-file "$CONFIG_FILE" \
            --checkpoint-path "$CKPT_DIR/model_ema_bf16.pt" \
            --experiment "$EXPERIMENT"
        rm -f "$PARAMS_TMP"
    else
        echo "[infer] already done ($N_GEN/$N_EP)"
    fi

    # -- 3. Metrics ----------------------------------------------------------
    if [[ ! -f "$RES_DIR/summary.json" ]]; then
        echo "[eval] computing metrics"
        mkdir -p "$RES_DIR"
        python -m video_model_eval.evaluate \
            --gen-dir "$GEN_DIR" \
            --gt-dir "$GT_DIR" \
            --out-dir "$RES_DIR" \
            --iter "$ITER_NUM"
    else
        echo "[eval] already done"
    fi
done

echo
echo "All done. To build the cross-checkpoint summary + plots:"
echo "    python -m video_model_eval.aggregate"
