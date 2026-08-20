# DEGRADED Stage 2: synthetic EN -> Tulu (sat_Olck alias). Non-competitive.
# Requires: python prepare_stage2_data.py --mode synthetic_degraded
# Prefer run_stage2_train.ps1 once kn_tcy_raw.csv is available.
$ErrorActionPreference = "Stop"

$pairDir = ".\stage2_data\train\eng_Latn-sat_Olck"
if (-not (Test-Path $pairDir)) {
    Write-Error @"
Missing $pairDir
Run first:  python prepare_stage2_data.py --mode synthetic_degraded
This path is degraded / non-competitive. Prefer real KN-TCY when available.
"@
}

$env:WANDB_DISABLED = "true"
$env:WANDB_MODE = "disabled"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$pythonExe = "python"
$venvPython = Resolve-Path "..\..\.venv\Scripts\python.exe" -ErrorAction SilentlyContinue
if ($venvPython) {
    $pythonExe = $venvPython.Path
    Write-Host "Using venv Python: $pythonExe"
}

# Fresh LoRA from Stage 1 EN-KN adapter; separate output dir from real KN-TCY.
& $pythonExe -u train_lora.py `
    --data_dir ./stage2_data `
    --model ./model_cache/indictrans2-en-indic-1B `
    --resume_from_checkpoint ./stage1_output/checkpoint-1500 `
    --output_dir ./stage2_synthetic_sat_output `
    --src_lang_list eng_Latn `
    --tgt_lang_list sat_Olck `
    --tulu_tag_alias sat_Olck `
    --save_steps 500 `
    --eval_steps 500 `
    --max_steps 3000 `
    --batch_size 2 `
    --grad_accum_steps 16 `
    --warmup_steps 500 `
    --max_grad_norm 1.0 `
    --learning_rate 2e-4 `
    --adam_beta1 0.9 `
    --adam_beta2 0.98 `
    --optimizer adamw_torch `
    --lr_scheduler inverse_sqrt `
    --num_workers 0 `
    --metric_for_best_model eval_BLEU `
    --greater_is_better `
    --seed 42 `
    --save_total_limit 3 `
    --patience 4 `
    --weight_decay 0.01 `
    --lora_target_modules "q_proj,k_proj" `
    --lora_dropout 0.1 `
    --lora_r 16 `
    --lora_alpha 32 `
    --print_samples `
    --report_to none
