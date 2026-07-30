# Stage 2: KN->Tulu LoRA continuing from Stage 1 checkpoint-1500
# Requires: kn_tcy_raw.csv prepared via prepare_stage2_data.py
$ErrorActionPreference = "Stop"

$env:WANDB_DISABLED = "true"
$env:WANDB_MODE = "disabled"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

# IMPORTANT: --model = BASE model. Stage-1 LoRA via --resume_from_checkpoint.
python -u train_lora.py `
    --data_dir ./stage2_data `
    --model ./model_cache/indictrans2-en-indic-1B `
    --resume_from_checkpoint ./stage1_output/checkpoint-1500 `
    --output_dir ./stage2_output `
    --src_lang_list kan_Knda `
    --tgt_lang_list tcy_Knda `
    --save_steps 500 `
    --eval_steps 500 `
    --max_steps 8000 `
    --batch_size 2 `
    --grad_accum_steps 16 `
    --warmup_steps 500 `
    --max_grad_norm 1.0 `
    --learning_rate 2e-4 `
    --adam_beta1 0.9 `
    --adam_beta2 0.98 `
    --optimizer adamw_torch `
    --lr_scheduler inverse_sqrt `
    --num_workers 2 `
    --metric_for_best_model eval_BLEU `
    --greater_is_better `
    --patience 6 `
    --weight_decay 0.01 `
    --lora_target_modules "q_proj,k_proj" `
    --lora_dropout 0.1 `
    --lora_r 16 `
    --lora_alpha 32 `
    --print_samples `
    --report_to none
