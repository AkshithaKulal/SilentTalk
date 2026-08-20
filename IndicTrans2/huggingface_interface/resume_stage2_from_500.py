import os
import subprocess

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = r"C:\Sarastra\SilentTalk\IndicTrans2\huggingface_interface"
os.chdir(BASE_DIR)

# Resume Stage 2 from checkpoint-500 with shorter horizon (max_steps=3000, patience=4).
# Use BASE model + --resume_from_checkpoint to avoid stacking a second LoRA adapter.
cmd = [
    "python",
    "train_lora.py",
    "--data_dir",
    os.path.join(BASE_DIR, "stage2_data"),
    "--model",
    os.path.join(BASE_DIR, "model_cache", "indictrans2-en-indic-1B"),
    "--resume_from_checkpoint",
    os.path.join(BASE_DIR, "stage2_output", "checkpoint-500"),
    "--output_dir",
    os.path.join(BASE_DIR, "stage2_output"),
    "--src_lang_list",
    "kan_Knda",
    "--tgt_lang_list",
    "sat_Olck",
    "--tulu_tag_alias",
    "sat_Olck",
    "--save_steps",
    "500",
    "--eval_steps",
    "500",
    "--max_steps",
    "3000",
    "--batch_size",
    "2",
    "--grad_accum_steps",
    "16",
    "--warmup_steps",
    "500",
    "--learning_rate",
    "2e-4",
    "--optimizer",
    "adamw_torch",
    "--lr_scheduler",
    "inverse_sqrt",
    "--num_workers",
    "0",
    "--metric_for_best_model",
    "eval_BLEU",
    "--greater_is_better",
    "--patience",
    "4",
    "--lora_target_modules",
    "q_proj,k_proj",
    "--lora_r",
    "16",
    "--lora_alpha",
    "32",
    "--report_to",
    "none",
]

print("Launching:", " ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd))
