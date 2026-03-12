#!/bin/bash
# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR="datasets/math_sdpo"
DOMAIN="math"
CONFIG_NAME="baseline_grpo"
TIME="05:00:00"

# Sweep parameters
MODEL_PATHS=(
    # swiss-ai/Apertus-8B-Instruct-2509
    # Qwen/Qwen3-4B
    # Qwen/Qwen2.5-7B-Instruct
    # meta-llama/Llama-3.1-8B-Instruct
    Qwen/Qwen3-8B
)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=2e-6
LOG_VAL_GENERATIONS=20

# =============================================================================
# Paths
ROOT_DIR=/iopsstor/scratch/cscs/msantelmo/SDPO
DATA_DIR_ABS="${ROOT_DIR}/${DATA_DIR}"
# Fixed Slurm resources
ACCOUNT="infra01"
NODES=1
PARTITION="normal"
ENV="sdpo"
NTASKS_PER_NODE=1
GPUS_PER_NODE=4
MEM=460000
CPUS_PER_TASK=288
# =============================================================================

submit_job() {
    local exp_name="$1"
    local script_args="$2"

    local setup_cmds="rm -rf /app/verl; \
    cd $ROOT_DIR; \
    pip install word2number latex2sympy2 math-verify[antlr4_9_3]==0.8.0; \
    pip install -e $ROOT_DIR; \
    pip install --upgrade wandb; \
    export PYTHONPATH=$ROOT_DIR:\$PYTHONPATH"

    local run_cmd="bash $ROOT_DIR/training/verl_training.sh $exp_name $CONFIG_NAME $DATA_DIR $script_args"
    local wrapped_cmd="srun --environment=\"$ENV\" bash -c '$setup_cmds; $run_cmd'"

    local sbatch_cmd=(
        sbatch
        --job-name="$exp_name"
        --account="$ACCOUNT"
        --nodes="$NODES"
        --partition="$PARTITION"
        --time="$TIME"
        --ntasks-per-node="$NTASKS_PER_NODE"
        --gpus-per-node="$GPUS_PER_NODE"
        --mem="$MEM"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="$ROOT_DIR/outputs/${exp_name}/%j.out"
        --error="$ROOT_DIR/outputs/${exp_name}/%j.err"
        --wrap="$wrapped_cmd"
    )

    echo "Submitting job ${exp_name}"
    "${sbatch_cmd[@]}"
}

# =============================================================================
# Sweep with feedback generator
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    SANITIZED_MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
    EXP_NAME="GRPO-${DOMAIN}-${SANITIZED_MODEL_NAME}"

    ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
    trainer.group_name=SDPO-${DOMAIN} \
    actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    algorithm.rollout_correction.rollout_is=token \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    trainer.log_val_generations=${LOG_VAL_GENERATIONS} \
    trainer.validation_data_dir=${ROOT_DIR}/outputs/${EXP_NAME}/val_generations \
    trainer.rollout_data_dir=${ROOT_DIR}/outputs/${EXP_NAME}/train_generations"

    submit_job "$EXP_NAME" "$ARGS"
done
