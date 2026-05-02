#!/bin/bash
#
# Bridge baseline eval with guidance sweep.
#
#   (one-time)  python -m bridge_baseline_eval.prepare_gt
#   (per run)   bash bridge_baseline_eval/run_eval.sh 2000 10000 18000 24000 32000 40000
#
# By default sweeps GUIDANCES="0 1 3 7". Override via:
#   GUIDANCES="0 3" bash bridge_baseline_eval/run_eval.sh 2000 10000
#
# Output layout (one tree per guidance):
#   generations/g<G>/iter_XXXXXXXX/<ep>_chunk.mp4
#   results/g<G>/iter_XXXXXXXX/{per_episode.csv, per_timestep.npz,
#                               summary.json, videos/<ep>_side_by_side.mp4}
#
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ---- Configure ------------------------------------------------------------
: "${IMAGINAIRE_OUTPUT_ROOT:?IMAGINAIRE_OUTPUT_ROOT must be set}"
EXPERIMENT="${EXPERIMENT:-cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py}"
CKPTS_ROOT="${CKPTS_ROOT:-$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict2_action_conditioned/official_runs_vid2vid/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320/checkpoints}"

EVAL_DIR="$PROJECT_DIR/bridge_baseline_eval"
GEN_ROOT="${GEN_ROOT:-$EVAL_DIR/generations}"
RES_ROOT="${RES_ROOT:-$EVAL_DIR/results}"
GT_DIR="$EVAL_DIR/gt_composite"
PARAMS_TEMPLATE="$EVAL_DIR/inference_params.json"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
GUIDANCES="${GUIDANCES:-0 1 3 7}"
VIDEO_FPS="${VIDEO_FPS:-4}"
# ----------------------------------------------------------------------------

mkdir -p "$GEN_ROOT" "$RES_ROOT"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <iter> [<iter> ...]"
    echo "Example (representative ckpts): $0 2000 10000 18000 24000 32000 40000"
    echo "Override sweep:  GUIDANCES=\"0 3\" $0 2000"
    exit 1
fi

if [[ ! -d "$GT_DIR" ]] || [[ -z "$(ls -A "$GT_DIR" 2>/dev/null)" ]]; then
    echo "[error] GT videos not found under $GT_DIR" >&2
    echo "        Run: python -m bridge_baseline_eval.prepare_gt" >&2
    exit 1
fi

TARGET_END=$(python -c "import json; print(json.load(open('$PARAMS_TEMPLATE'))['end'])")

for ITER_NUM in "$@"; do
    ITER=$(printf "iter_%09d" "$ITER_NUM")
    CKPT_DIR="$CKPTS_ROOT/$ITER"

    echo "========================================================"
    echo "== $ITER"
    echo "========================================================"

    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "[skip] missing checkpoint dir: $CKPT_DIR"
        continue
    fi

    # -- 1. Convert DCP -> .pt once per iter (shared across guidances) -------
    if [[ ! -f "$CKPT_DIR/model_ema_bf16.pt" ]]; then
        echo "[convert] $CKPT_DIR/model -> $CKPT_DIR/model_ema_bf16.pt"
        python scripts/convert_distcp_to_pt.py "$CKPT_DIR/model" "$CKPT_DIR"
    else
        echo "[convert] already done"
    fi

    for G in $GUIDANCES; do
        GEN_DIR="$GEN_ROOT/g$G/$ITER"
        RES_DIR="$RES_ROOT/g$G/$ITER"

        echo "-------- guidance=$G --------"

        # -- 2. Inference (skip if all chunk mp4s already exist) -------------
        mkdir -p "$GEN_DIR"
        N_GEN=$(find "$GEN_DIR" -name '*_chunk.mp4' | wc -l)
        if [[ "$N_GEN" -lt "$TARGET_END" ]]; then
            echo "[infer g=$G] $N_GEN/$TARGET_END done, generating missing"
            PARAMS_TMP=$(mktemp --suffix=.json)
            python -c "
import json, sys
d = json.load(open(sys.argv[1]))
d['save_root'] = sys.argv[2]
d['guidance'] = int(sys.argv[3])
json.dump(d, open(sys.argv[4], 'w'))
" "$PARAMS_TEMPLATE" "$GEN_DIR" "$G" "$PARAMS_TMP"
            python examples/action_conditioned.py \
                -i "$PARAMS_TMP" \
                --output-dir "$GEN_DIR" \
                --config-file "$CONFIG_FILE" \
                --checkpoint-path "$CKPT_DIR/model_ema_bf16.pt" \
                --experiment "$EXPERIMENT"
            rm -f "$PARAMS_TMP"
        else
            echo "[infer g=$G] already done ($N_GEN/$TARGET_END)"
        fi

        # -- 3. Metrics --------------------------------------------------
        if [[ ! -f "$RES_DIR/summary.json" ]]; then
            echo "[eval g=$G] computing metrics"
            mkdir -p "$RES_DIR"
            EVAL_ARGS=(
                --gen-dir "$GEN_DIR"
                --gt-dir "$GT_DIR"
                --out-dir "$RES_DIR"
                --iter "$ITER_NUM"
                --video-fps "$VIDEO_FPS"
            )
            if [[ "$SAVE_VIDEOS" == "1" ]]; then
                EVAL_ARGS+=(--save-videos)
            fi
            python -m video_model_eval.evaluate "${EVAL_ARGS[@]}"
        else
            echo "[eval g=$G] already done"
        fi
    done
done

echo
echo "All done. Aggregate per guidance:"
for G in $GUIDANCES; do
    echo "    python -m video_model_eval.aggregate --results-root $RES_ROOT/g$G --out-dir $EVAL_DIR/g$G"
done
