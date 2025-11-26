
set -e
set -x

# --- Configuration ---
# Adjust these variables as needed
LLM_NAME=gpt-4o-mini
DOMAIN="gsm8k"
AGENT_NAMES="MathSolver"
AGENT_NUMS=4
BATCH_SIZE=2

# IMPORTANT: Separate training and testing datasets to avoid data leakage
TRAIN_DATASET_JSON="datasets/gsm8k/gsm8k_train.jsonl"  # For Phase 1: data generation
TEST_DATASET_JSON="datasets/gsm8k/gsm8k_test.jsonl"    # For Phase 3: evaluation


# == Phase 1: Generate initial dataset for GTD models ==
echo "--- Running GTD Phase 1: Dataset Generation (using TRAINING set) ---"
python -m experiments.run_gsm8k \
     --llm_name $LLM_NAME \
     --domain $DOMAIN \
     --agent_names $AGENT_NAMES \
     --agent_nums $AGENT_NUMS \
     --dataset_json $TRAIN_DATASET_JSON \
     --mode GTD \
     --gtd-generate-data \
     --gtd-datagen-limit 50 \
     --gtd-dataset-path "gtd_gsm8k_dataset_training.jsonl"


# == Phase 2: Train Proxy and Diffusion models ==
echo "--- Running GTD Phase 2: Model Training (using Phase 1 generated data) ---"
python -m experiments.run_gsm8k \
    --llm_name $LLM_NAME \
    --domain $DOMAIN \
    --agent_names $AGENT_NAMES \
    --agent_nums $AGENT_NUMS \
    --dataset_json $TRAIN_DATASET_JSON \
    --mode GTD \
    --gtd-train-models \
    --gtd-epochs 10 \
    --gtd-dataset-path "gtd_gsm8k_dataset_training.jsonl" \
    --gtd-proxy-model-path "proxy_model_gsm8k.pth" \
    --gtd-diffusion-model-path "diffusion_model_gsm8k.pth"


# == Phase 3: Run inference with a pre-trained GTD Framework ==
echo "--- Running GTD Phase 3: Inference (using TEST set - no data leakage) ---"
python -m experiments.run_gsm8k \
    --llm_name $LLM_NAME \
    --domain $DOMAIN \
    --agent_names $AGENT_NAMES \
    --agent_nums $AGENT_NUMS \
    --dataset_json $TEST_DATASET_JSON \
    --mode GTD \
    --batch_size $BATCH_SIZE \
    --gtd-proxy-model-path "proxy_model_gsm8k.pth" \
    --gtd-diffusion-model-path "diffusion_model_gsm8k.pth"


echo "--- Script finished ---" 
