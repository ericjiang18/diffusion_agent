#!/bin/bash

set -e
set -x

# --- Configuration ---
LLM_NAME="gpt-4o-mini"
DOMAIN="math"
AGENT_NAMES="MathSolver"
AGENT_NUMS=4
BATCH_SIZE=2
OUTPUT_DIR="trained_models"

# Separate training and testing datasets to avoid data leakage
TRAIN_DATASET_JSON="datasets/MATH/MATH_train.jsonl"  # For Phase 1: data generation
TEST_DATASET_JSON="datasets/MATH/MATH_test500.jsonl"    # For Phase 3: evaluation

mkdir -p $OUTPUT_DIR

# --- GTD Mode ---

# == Phase 1: Generate initial dataset for GTD models ==
echo "--- Running GTD Phase 1: Dataset Generation for MATH (using TRAINING set) ---"
python3 -m experiments.run_math \
     --llm_name $LLM_NAME \
     --domain $DOMAIN \
     --agent_names $AGENT_NAMES \
     --agent_nums $AGENT_NUMS \
     --dataset_json $TRAIN_DATASET_JSON \
     --mode GTD \
     --gtd-generate-data \
     --gtd-datagen-limit 10 \
     --gtd-dataset-path "$OUTPUT_DIR/gtd_math_dataset.jsonl"


# == Phase 2: Train Proxy and Diffusion models ==
echo "--- Running GTD Phase 2: Model Training for MATH (using Phase 1 generated data) ---"
python3 -m experiments.run_math \
    --llm_name $LLM_NAME \
    --domain $DOMAIN \
    --agent_names $AGENT_NAMES \
    --agent_nums $AGENT_NUMS \
    --dataset_json $TRAIN_DATASET_JSON \
    --mode GTD \
    --gtd-train-models \
    --gtd-epochs 10 \
    --gtd-dataset-path "$OUTPUT_DIR/gtd_math_dataset.jsonl" \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_math.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_math.pth"


# == Phase 3: Run inference with a pre-trained GTD Framework ==
echo "--- Running GTD Phase 3: Inference for MATH (using TEST set - no data leakage) ---"
python3 -m experiments.run_math \
    --llm_name $LLM_NAME \
    --domain $DOMAIN \
    --agent_names $AGENT_NAMES \
    --agent_nums $AGENT_NUMS \
    --dataset_json $TEST_DATASET_JSON \
    --mode GTD \
    --batch_size $BATCH_SIZE \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_math.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_math.pth"


echo "--- Script finished ---"
