#!/bin/bash
export OPENAI_API_KEY=""
export OPENAI_BASE_URL=""

set -e

PAPER_ID=""
WORKSPACE_DIR="results/test_run"
MODEL_NAME="gpt-4o-mini"
# REPLACE_FLAG="--replace"

# --- Main Orchestration Logic ---
echo "================================================================="
echo "🚀 STARTING RePro for Paper: ${PAPER_ID}"
echo "   (All outputs will be saved under ./${WORKSPACE_DIR})"
echo "================================================================="

echo -e "\nPHASE 1 & 2: Launching Code Generation and Signal Design IN PARALLEL..."

python -m scripts.generate_initial_code \
    --paper_id "${PAPER_ID}" \
    --workspace_dir "${WORKSPACE_DIR}" \
    --model "${MODEL_NAME}" &
CODE_GEN_PID=$!
echo "  -> Launched Code Generation (PID: ${CODE_GEN_PID})"

python -m scripts.design_signals \
    --paper_id "${PAPER_ID}" \
    --workspace_dir "${WORKSPACE_DIR}" \
    --model "${MODEL_NAME}" &
SIGNAL_GEN_PID=$!
echo "  -> Launched Signal Design (PID: ${SIGNAL_GEN_PID})"

echo -e "\nWaiting for background jobs to finish..."
wait $CODE_GEN_PID
wait $SIGNAL_GEN_PID
echo "✅ Code Generation and Signal Design completed successfully."

echo -e "\nPHASE 3: Launching Code Reflection Pipeline..."

python -m scripts.reflect_code \
    --paper_id "${PAPER_ID}" \
    --workspace_dir "${WORKSPACE_DIR}" \
    --model_eval "${MODEL_NAME}" \
    --model_plan "${MODEL_NAME}" \
    --model_revise "${MODEL_NAME}"

echo -e "\n================================================================="
echo "🎉🎉🎉 COMPLETED SUCCESSFULLY! 🎉🎉🎉"
echo "================================================================="