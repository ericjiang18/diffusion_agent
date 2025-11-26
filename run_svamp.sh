#!/bin/bash

set -e
set -x

# --- Configuration ---
LLM_NAME="gpt-4o-mini"
DOMAIN="svamp"
AGENT_NAMES="MathSolver"
AGENT_NUMS=4
BATCH_SIZE=2
OUTPUT_DIR="trained_models"

# IMPORTANT: Separate training and testing datasets to avoid data leakage
TRAIN_DATASET_JSON="datasets/SVAMP/SVAMP_train.jsonl"  # For Phase 1: data generation
TEST_DATASET_JSON="datasets/SVAMP/SVAMP_test.jsonl"    # For Phase 3: evaluation

mkdir -p $OUTPUT_DIR

# --- GTD Mode ---

# == Phase 1: Generate initial dataset for GTD models ==
echo "--- Running GTD Phase 1: Dataset Generation for SVAMP (using TRAINING set) ---"
python3 -m experiments.run_svamp \
     --llm_name $LLM_NAME \
     --domain $DOMAIN \
     --agent_names $AGENT_NAMES \
     --agent_nums $AGENT_NUMS \
     --dataset_json $TRAIN_DATASET_JSON \
     --mode GTD \
     --gtd-generate-data \
     --gtd-datagen-limit 1 \
     --gtd-dataset-path "$OUTPUT_DIR/gtd_svamp_dataset.jsonl"


# == Phase 2: Train Proxy and Diffusion models ==
echo "--- Running GTD Phase 2: Model Training for SVAMP (using Phase 1 generated data) ---"
python3 -m experiments.run_svamp \
    --llm_name $LLM_NAME \
    --domain $DOMAIN \
    --agent_names $AGENT_NAMES \
    --agent_nums $AGENT_NUMS \
    --dataset_json $TRAIN_DATASET_JSON \
    --mode GTD \
    --gtd-train-models \
    --gtd-epochs 10 \
    --gtd-dataset-path "$OUTPUT_DIR/gtd_svamp_dataset.jsonl" \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_svamp.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_svamp.pth"


# == Phase 3: Run inference with a pre-trained GTD Framework ==
echo "--- Running GTD Phase 3: Inference for SVAMP (using TEST set - no data leakage) ---"
python3 -m experiments.run_svamp \
    --llm_name $LLM_NAME \
    --domain $DOMAIN \
    --agent_names $AGENT_NAMES \
    --agent_nums $AGENT_NUMS \
    --dataset_json $TEST_DATASET_JSON \
    --mode GTD \
    --batch_size $BATCH_SIZE \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_svamp.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_svamp.pth"


echo "--- Script finished ---"
