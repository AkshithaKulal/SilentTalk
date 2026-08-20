#!/bin/bash
# DEGRADED Stage 2: synthetic EN -> Tulu (sat_Olck). Non-competitive.
set -euo pipefail

PAIR_DIR="./stage2_data/train/eng_Latn-sat_Olck"
if [ ! -d "$PAIR_DIR" ]; then
  echo "Missing $PAIR_DIR" >&2
  echo "Run: python3 prepare_stage2_data.py --mode synthetic_degraded" >&2
  exit 1
fi

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="python3"
if [ -x "../../.venv/Scripts/python.exe" ]; then
  PYTHON_BIN="../../.venv/Scripts/python.exe"
  echo "Using venv Python: $PYTHON_BIN"
fi

"$PYTHON_BIN" train_lora.py \
    --data_dir ./stage2_data \
    --model ./model_cache/indictrans2-en-indic-1B \
    --resume_from_checkpoint ./stage1_output/checkpoint-1500 \
    --output_dir ./stage2_synthetic_sat_output \
    --src_lang_list eng_Latn \
    --tgt_lang_list sat_Olck \
    --tulu_tag_alias sat_Olck \
    --save_steps 500 \
    --eval_steps 500 \
    --max_steps 3000 \
    --batch_size 2 \
    --grad_accum_steps 16 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    --learning_rate 2e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --optimizer adamw_torch \
    --lr_scheduler inverse_sqrt \
    --num_workers 0 \
    --metric_for_best_model eval_BLEU \
    --greater_is_better \
    --seed 42 \
    --save_total_limit 3 \
    --patience 4 \
    --weight_decay 0.01 \
    --lora_target_modules "q_proj,k_proj" \
    --lora_dropout 0.1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --print_samples \
    --report_to none
