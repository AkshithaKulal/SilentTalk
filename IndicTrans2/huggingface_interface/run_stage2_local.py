import os
import subprocess

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = r"C:\Sarastra\SilentTalk\IndicTrans2\huggingface_interface"
os.chdir(BASE_DIR)

# Use "python" on Windows (not python3)
cmd = [
    "python", "train_lora.py",
    "--data_dir", os.path.join(BASE_DIR, "stage2_data"),
    "--model", os.path.join(BASE_DIR, "model_cache", "indictrans2-en-indic-1B"),
    "--resume_from_checkpoint", os.path.join(BASE_DIR, "stage1_output", "checkpoint-1500"),
    "--output_dir", os.path.join(BASE_DIR, "stage2_output"),
    "--src_lang_list", "kan_Knda",
    "--tgt_lang_list", "sat_Olck",
    "--tulu_tag_alias", "sat_Olck",
    "--save_steps", "500",
    "--eval_steps", "500",
    "--max_steps", "8000",
    "--batch_size", "2",          # smaller than Colab's T4 setting — 8GB card
    "--grad_accum_steps", "16",   # keeps same effective batch size (2x16=32)
    "--warmup_steps", "500",
    "--learning_rate", "2e-4",
    "--optimizer", "adamw_torch",
    "--lr_scheduler", "inverse_sqrt",
    "--num_workers", "2",
    "--metric_for_best_model", "eval_BLEU",
    "--greater_is_better",
    "--patience", "6",
    "--lora_target_modules", "q_proj,k_proj",
    "--lora_r", "16",
    "--lora_alpha", "32",
    "--print_samples",
    "--report_to", "none",
]

subprocess.run(cmd)
