#!/bin/bash
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT_NUM=1000  # change to your latest checkpoint number

# IMPORTANT: --model stays the BASE model. Checkpoint via --resume_from_checkpoint.
python3 train_lora.py \
    --data_dir ./stage1_data \
    --model ./model_cache/indictrans2-en-indic-1B \
    --resume_from_checkpoint ./stage1_output/checkpoint-$CHECKPOINT_NUM \
    --output_dir ./stage1_output \
    --src_lang_list eng_Latn \
    --tgt_lang_list kan_Knda \
    --save_steps 500 \
    --eval_steps 500 \
    --max_steps 8000 \
    --batch_size 2 \
    --grad_accum_steps 16 \
    --warmup_steps 500 \
    --learning_rate 2e-4 \
    --lora_target_modules "q_proj,k_proj" \
    --lora_r 16 \
    --lora_alpha 32 \
    --metric_for_best_model eval_BLEU \
    --greater_is_better \
    --patience 6 \
    --print_samples \
    --report_to none
