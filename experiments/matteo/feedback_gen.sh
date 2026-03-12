#!/bin/bash
# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR="datasets/math_sdpo"
DOMAIN="math"
CONFIG_NAME="sdpo"
TIME="08:00:00"

# Sweep parameters
MODEL_PATHS=(
    # swiss-ai/Apertus-8B-Instruct-2509
    # meta-llama/Llama-3.1-8B-Instruct
    # Qwen/Qwen3-4B
    # Qwen/Qwen2.5-7B-Instruct
    Qwen/Qwen3-8B
)
IS_THINKING_ON=true
USE_SUCC_ATTEMPTS_AS_FEEDBACKS=(false)
FG_PROMPT_TEMPLATES=(
    # default_feedback_generator_prompt
    # intervention_prompt
)
USE_REFERENCE_IN_FG=true
TRAIN_BATCH_SIZE=32
N_ROLLOUTS=(1)
LRS=(1e-5)  # 2e-6)
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
LOG_VAL_GENERATIONS=20
TEACHER_UPDATES=(
    0.0
    0.01
    # 0.01 0.05
)

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
for N_ROLLOUT in "${N_ROLLOUTS[@]}"; do
    for LR in "${LRS[@]}"; do
        for TEACHER_UPDATE in "${TEACHER_UPDATES[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for FG_PROMPT in "${FG_PROMPT_TEMPLATES[@]}"; do
                    DATETIME=$(date +"%Y%m%d-%H%M%S")
                    SANITIZED_MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                    TEACHER_UPDATE_STR=$( [ "$TEACHER_UPDATE" = 0.0 ] && echo "-FROZEN_TEACHER" || echo "" )
                    EXP_NAME="SDPO-${DOMAIN}-${SANITIZED_MODEL_NAME}-FG-${FG_PROMPT}${TEACHER_UPDATE_STR}-LR${LR}-N${N_ROLLOUT}-${DATETIME}"

                    ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
                    trainer.group_name=SDPO-${DOMAIN} \
                    actor_rollout_ref.rollout.n=$N_ROLLOUT \
                    data.apply_chat_template_kwargs.enable_thinking=$IS_THINKING_ON \
                    actor_rollout_ref.model.path=$MODEL_PATH \
                    actor_rollout_ref.actor.optim.lr=$LR \
                    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
                    actor_rollout_ref.actor.self_distillation.distillation_topk=500 \
                    algorithm.rollout_correction.rollout_is=token \
                    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
                    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
                    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE \
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
                    trainer.rollout_data_dir=${ROOT_DIR}/outputs/${EXP_NAME}/train_generations \
                    hydra.run.dir=${ROOT_DIR}/outputs/${EXP_NAME} \
                    hydra.output_subdir=.hydra"

                    submit_job "$EXP_NAME" "$ARGS"
                    sleep 2
                done
            done
        done
    done
done

# Sweep with reference solution as feedback (no generated feedback)
for TEACHER_UPDATE in "${TEACHER_UPDATES[@]}"; do
    for N_ROLLOUT in "${N_ROLLOUTS[@]}"; do
        for LR in "${LRS[@]}"; do
            for USE_SUCC_ATTEMPTS_AS_FEEDBACK in "${USE_SUCC_ATTEMPTS_AS_FEEDBACKS[@]}"; do
                for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                    DATETIME=$(date +"%Y%m%d-%H%M%S")
                    SANITIZED_MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                    TEACHER_UPDATE_STR=$( [ "$TEACHER_UPDATE" = 0.0 ] && echo "-FROZEN_TEACHER" || echo "" )
                    EXP_NAME="SDPO-${DOMAIN}-${SANITIZED_MODEL_NAME}-REFERENCE${TEACHER_UPDATE_STR}-LR${LR}-N${N_ROLLOUT}-${DATETIME}"

                    ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
                    trainer.group_name=SDPO-${DOMAIN} \
                    actor_rollout_ref.rollout.n=$N_ROLLOUT \
                    actor_rollout_ref.model.path=$MODEL_PATH \
                    actor_rollout_ref.actor.optim.lr=$LR \
                    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
                    actor_rollout_ref.actor.self_distillation.distillation_topk=500 \
                    algorithm.rollout_correction.rollout_is=token \
                    data.apply_chat_template_kwargs.enable_thinking=$IS_THINKING_ON \
                    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
                    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
                    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE \
                    actor_rollout_ref.actor.self_distillation.use_successful_previous_attempt=${USE_SUCC_ATTEMPTS_AS_FEEDBACK} \
                    actor_rollout_ref.actor.self_distillation.feedback_generator.enable=true \
                    actor_rollout_ref.actor.self_distillation.feedback_generator.backend=reference_based \
                    actor_rollout_ref.actor.self_distillation.feedback_generator.include_ground_truth=${USE_REFERENCE_IN_FG} \
                    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
                    actor_rollout_ref.rollout.val_kwargs.n=8 \
                    trainer.log_val_generations=${LOG_VAL_GENERATIONS} \
                    trainer.validation_data_dir=${ROOT_DIR}/outputs/${EXP_NAME}/val_generations \
                    trainer.rollout_data_dir=${ROOT_DIR}/outputs/${EXP_NAME}/train_generations \
                    hydra.run.dir=${ROOT_DIR}/outputs/${EXP_NAME} \
                    hydra.output_subdir=.hydra"

                    submit_job "$EXP_NAME" "$ARGS"
                    sleep 2
                done
            done
        done
    done
done