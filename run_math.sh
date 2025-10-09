#!/bin/bash

set -e
set -x

# --- Configuration ---
LLM_NAME="gpt-4o-mini"
DOMAIN="math"
AGENT_NAMES="MathSolver"
AGENT_NUMS=4
BATCH_SIZE=2
DATASET_JSON="datasets/MATH/test500.jsonl"
OUTPUT_DIR="trained_models"

mkdir -p $OUTPUT_DIR

BASE_CMD="python -m experiments.run_math --llm_name $LLM_NAME --domain $DOMAIN --agent_names $AGENT_NAMES --agent_nums $AGENT_NUMS --dataset_json $DATASET_JSON"

# --- GTD Mode ---

# == Phase 1: Generate initial dataset ==
echo "--- Running GTD Phase 1: Dataset Generation for MATH ---"
$BASE_CMD \
     --gtd-generate-data \
     --gtd-datagen-limit 10 \
     --gtd-dataset-path "$OUTPUT_DIR/gtd_math_dataset.jsonl"


# == Phase 2: Train Proxy and Diffusion models ==
echo "--- Running GTD Phase 2: Model Training for MATH ---"
$BASE_CMD \
    --gtd-train-models \
    --gtd-epochs 10 \
    --gtd-dataset-path "$OUTPUT_DIR/gtd_math_dataset.jsonl" \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_math.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_math.pth"


# == Phase 3: Run inference ==
echo "--- Running GTD Phase 3: Inference for MATH ---"
$BASE_CMD \
    --batch_size $BATCH_SIZE \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_math.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_math.pth"


echo "--- Script finished ---"
