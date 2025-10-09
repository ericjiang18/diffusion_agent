#!/bin/bash

set -e
set -x

# --- Configuration ---
LLM_NAME="gpt-4o-mini"
DOMAIN="multiarith"
AGENT_NAMES="MathSolver"
AGENT_NUMS=4
BATCH_SIZE=2
DATASET_JSON="datasets/MultiArith/MultiArith_test.json"
OUTPUT_DIR="trained_models"

mkdir -p $OUTPUT_DIR

BASE_CMD="python -m experiments.run_multiarith --llm_name $LLM_NAME --domain $DOMAIN --agent_names $AGENT_NAMES --agent_nums $AGENT_NUMS --dataset_json $DATASET_JSON"

# --- GTD Mode ---

# == Phase 1: Generate initial dataset ==
echo "--- Running GTD Phase 1: Dataset Generation for MultiArith ---"
$BASE_CMD \
     --gtd-generate-data \
     --gtd-datagen-limit 10 \
     --gtd-dataset-path "$OUTPUT_DIR/gtd_multiarith_dataset.jsonl"


# == Phase 2: Train Proxy and Diffusion models ==
echo "--- Running GTD Phase 2: Model Training for MultiArith ---"
$BASE_CMD \
    --gtd-train-models \
    --gtd-epochs 10 \
    --gtd-dataset-path "$OUTPUT_DIR/gtd_multiarith_dataset.jsonl" \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_multiarith.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_multiarith.pth"


# == Phase 3: Run inference ==
echo "--- Running GTD Phase 3: Inference for MultiArith ---"
$BASE_CMD \
    --batch_size $BATCH_SIZE \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_multiarith.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_multiarith.pth"


echo "--- Script finished ---"
