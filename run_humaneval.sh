#!/bin/bash

# This script provides example commands to run the humaneval experiments.
set -e
set -x

# --- Configuration ---
LLM_NAME="gpt-4o-mini"
DOMAIN="humaneval"
AGENT_NAMES="Algorithm_Designer Bug_Fixer Programming_Expert Test_Analyst"
AGENT_NUMS="1 1 1 1" # Corresponds to the number of agents for each name
BATCH_SIZE=2
DATASET_JSON="datasets/humaneval/humaneval-py.jsonl"
OUTPUT_DIR="trained_models" # All outputs will go here

# Ensure the output directory exists
mkdir -p $OUTPUT_DIR

BASE_CMD="python3 -m experiments.run_humaneval --llm_name $LLM_NAME --domain $DOMAIN --agent_names $AGENT_NAMES --agent_nums $AGENT_NUMS --dataset_json $DATASET_JSON"

# --- GTD Mode ---
# This mode uses a Graph Topology Diffusion model to generate agent communication graphs.
# It consists of three phases. Uncomment the phase you want to run.

# == Phase 1: Generate initial dataset for GTD models ==
echo "--- Running GTD Phase 1: Dataset Generation for Humaneval ---"
$BASE_CMD \
     --mode GTD \
     --gtd-generate-data \
     --gtd-datagen-limit 10 \
     --gtd-dataset-path "$OUTPUT_DIR/gtd_humaneval_dataset.jsonl"


# == Phase 2: Train Proxy and Diffusion models ==
echo "--- Running GTD Phase 2: Model Training for Humaneval ---"
$BASE_CMD \
    --mode GTD \
    --gtd-train-models \
    --gtd-epochs 10 \
    --gtd-dataset-path "$OUTPUT_DIR/gtd_humaneval_dataset.jsonl" \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_humaneval.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_humaneval.pth"


# == Phase 3: Run inference with a pre-trained GTD Framework ==
echo "--- Running GTD Phase 3: Inference for Humaneval ---"
$BASE_CMD \
    --mode GTD \
    --batch_size $BATCH_SIZE \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_humaneval.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_humaneval.pth"


echo "--- Script finished ---"
