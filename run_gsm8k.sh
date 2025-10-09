#!/bin/bash

# This script provides example commands to run the gsm8k experiments.
# This script should be run from within the 'agent_diffusion' directory.
#
# Before running, ensure you have set up your environment, for example by creating
# a .env file from template.env and installing dependencies:
#
# cp template.env .env
# pip install -r requirements.txt

set -e
set -x

# --- Configuration ---
# Adjust these variables as needed
LLM_NAME=gpt-4o-mini
DOMAIN="gsm8k"
AGENT_NAMES="MathSolver"
AGENT_NUMS=4
BATCH_SIZE=2
DATASET_JSON="datasets/gsm8k/gsm8k.jsonl"
BASE_CMD="python -m experiments.run_gsm8k --llm_name $LLM_NAME --domain $DOMAIN --agent_names $AGENT_NAMES --agent_nums $AGENT_NUMS --dataset_json $DATASET_JSON"


#--- Test Mode ----

#echo "--- Running FullConnected Mode ---"
#$BASE_CMD \
#    --mode FullConnected \
#    --batch_size $BATCH_SIZE \
#    --num_rounds 1 \
#    --num_iterations 5


# --- GTD Mode ---
# This mode uses a Graph Topology Diffusion model to generate agent communication graphs.
# It consists of three phases. Uncomment the phase you want to run.

# == Phase 1: Generate initial dataset for GTD models ==
# This runs baseline topologies (like 'fully_connected', 'chain', etc.)
# on a subset of the data to collect performance metrics.
echo "--- Running GTD Phase 1: Dataset Generation ---"
$BASE_CMD \
     --mode GTD \
     --gtd-generate-data \
     --gtd-datagen-limit 50 \
     --gtd-dataset-path "gtd_gsm8k_dataset.jsonl"


# == Phase 2: Train Proxy and Diffusion models ==
# This uses the dataset from Phase 1 to train the necessary models for GTD.
echo "--- Running GTD Phase 2: Model Training ---"
$BASE_CMD \
    --mode GTD \
    --gtd-train-models \
    --gtd-epochs 10 \
    --gtd-dataset-path "gtd_gsm8k_dataset.jsonl" \
    --gtd-proxy-model-path "proxy_model_gsm8k.pth" \
    --gtd-diffusion-model-path "diffusion_model_gsm8k.pth"


# == Phase 3: Run inference with a pre-trained GTD Framework ==
# This uses the trained models to generate a new graph for each task and run the experiment.
echo "--- Running GTD Phase 3: Inference ---"
$BASE_CMD \
    --mode GTD \
    --batch_size $BATCH_SIZE \
    --gtd-proxy-model-path "proxy_model_gsm8k.pth" \
    --gtd-diffusion-model-path "diffusion_model_gsm8k.pth"


# --- Non-GTD Baseline Modes ---
# These modes use fixed or optimizable graph structures.

# == Example: FullConnected mode (no optimization) ==



# == Example: Chain mode with spatial graph optimization ==
# This will try to learn the best chain-like structure.
# echo "--- Running Chain Mode with Spatial Optimization ---"
# $BASE_CMD \
#     --mode Chain \
#     --batch_size $BATCH_SIZE \
#     --optimized_spatial \
#     --lr 0.01 \
#     --num_iterations 10

echo "--- Script finished ---" 