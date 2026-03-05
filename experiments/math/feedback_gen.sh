#!/bin/bash

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT_DIR=/iopsstor/scratch/cscs/msantelmo/SDPO
DATA_DIR="datasets/math_sdpo"          # relative to ROOT_DIR (used as TASK env var)
DATA_DIR_ABS="${ROOT_DIR}/${DATA_DIR}"  # absolute path for preprocessing
CONFIG_NAME="sdpo"

# Fixed Slurm resources
ACCOUNT="infra01"
NODES=1
PARTITION="normal"
TIME="12:00:00"
ENV="sdpo"
NTASKS_PER_NODE=1
GPUS_PER_NODE=4
MEM=460000
CPUS_PER_TASK=288

# Sweep parameters
MODEL_PATHS=(
    # swiss-ai/Apertus-8B-Instruct-2509
    meta-llama/Llama-3.1-8B-Instruct
    Qwen/Qwen3-4B
    # Qwen/Qwen3-8B
)
FG_PROMPT_TEMPLATES=(
    default_feedback_generator_prompt
    intervention_prompt
)
USE_REFERENCE_IN_FG=true
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
LOG_VAL_GENERATIONS=20


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
    local wrapped_cmd="srun bash -c '$setup_cmds; $run_cmd'"

    local sbatch_cmd=(
        sbatch
        --job-name="$exp_name"
        --account="$ACCOUNT"
        --nodes="$NODES"
        --partition="$PARTITION"
        --time="$TIME"
        --environment="$ENV"
        --ntasks-per-node="$NTASKS_PER_NODE"
        --gpus-per-node="$GPUS_PER_NODE"
        --mem="$MEM"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="$ROOT_DIR/outputs/%j.out"
        --error="$ROOT_DIR/outputs/%j.err"
        --wrap="$wrapped_cmd"
    )

    echo "Submitting job ${exp_name}"
    "${sbatch_cmd[@]}"
}

# =============================================================================
# Sweep with feedback generator
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for FG_PROMPT in "${FG_PROMPT_TEMPLATES[@]}"; do
        SANITIZED_MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
        EXP_NAME="SDPO-MATH-${SANITIZED_MODEL_NAME}-FG-${FG_PROMPT}"

        ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
        trainer.group_name=SDPO-math \
        actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=$LR \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
        algorithm.rollout_correction.rollout_is=token \
        actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
        actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
        actor_rollout_ref.actor.self_distillation.feedback_generator.enable=true \
        actor_rollout_ref.actor.self_distillation.feedback_generator.backend=openai \
        actor_rollout_ref.actor.self_distillation.feedback_generator.include_ground_truth=${USE_REFERENCE_IN_FG} \
        actor_rollout_ref.actor.self_distillation.feedback_generator.fail_on_error=false \
        actor_rollout_ref.actor.self_distillation.feedback_generator.max_concurrent_requests=64 \
        +actor_rollout_ref.actor.self_distillation.feedback_generator.generation_kwargs.max_completion_tokens=4096 \
        actor_rollout_ref.actor.self_distillation.feedback_generator.model=Qwen/Qwen3-Next-80B-A3B-Thinking \
        actor/feedback_generator_prompt@actor_rollout_ref.actor.self_distillation.feedback_generator=${FG_PROMPT} \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.rollout.val_kwargs.n=8 \
        trainer.log_val_generations=${LOG_VAL_GENERATIONS} \
        trainer.validation_data_dir=${ROOT_DIR}/outputs/${EXP_NAME}/val_generations \
        trainer.rollout_data_dir=${ROOT_DIR}/outputs/${EXP_NAME}/train_generations"

        submit_job "$EXP_NAME" "$ARGS"
    done
done

# Sweep with reference solution as feedback (no generated feedback)
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    SANITIZED_MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
    EXP_NAME="SDPO-MATH-${SANITIZED_MODEL_NAME}-REFERENCE"

    ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
    trainer.group_name=SDPO-math \
    actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    algorithm.rollout_correction.rollout_is=token \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
    actor_rollout_ref.actor.self_distillation.feedback_generator.enable=true \
    actor_rollout_ref.actor.self_distillation.feedback_generator.backend=reference_based \
    actor_rollout_ref.actor.self_distillation.feedback_generator.include_ground_truth=${USE_REFERENCE_IN_FG} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    trainer.log_val_generations=${LOG_VAL_GENERATIONS} \
    trainer.validation_data_dir=${ROOT_DIR}/outputs/${EXP_NAME}/val_generations \
    trainer.rollout_data_dir=${ROOT_DIR}/outputs/${EXP_NAME}/train_generations"

    submit_job "$EXP_NAME" "$ARGS"
done