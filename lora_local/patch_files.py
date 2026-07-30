# Patch 1: train_lora.py
with open("train_lora.py") as f:
    content = f.read()
content = content.replace("evaluation_strategy", "eval_strategy")
with open("train_lora.py", "w") as f:
    f.write(content)

# Patch 2: collator.py - imports must be MODULE-level, not inside the class
collator_path = "IndicTransToolkit_repo/IndicTransToolkit/collator.py"
with open(collator_path) as f:
    content = f.read()

# Remove any indented/class-body copies of these imports
bad_block = (
    "    from transformers.utils import PaddingStrategy\n"
    "    from transformers.tokenization_utils import PreTrainedTokenizerBase\n"
    "    from transformers.data.data_collator import pad_without_fast_tokenizer_warning\n"
)
content = content.replace(bad_block, "")

good_imports = (
    "from transformers.utils import PaddingStrategy\n"
    "from transformers.tokenization_utils import PreTrainedTokenizerBase\n"
    "from transformers.data.data_collator import pad_without_fast_tokenizer_warning\n"
)

if "from transformers.data.data_collator import pad_without_fast_tokenizer_warning\n" not in content:
    anchor = "from dataclasses import dataclass\n"
    if anchor not in content:
        raise SystemExit("Could not find insert point in collator.py")
    # Insert after typing imports block if present, else after dataclass import
    if "from typing import" in content:
        # insert after the typing import line
        lines = content.splitlines(keepends=True)
        out = []
        inserted = False
        for line in lines:
            out.append(line)
            if (not inserted) and line.startswith("from typing import"):
                out.append("\n")
                out.append(good_imports)
                inserted = True
        content = "".join(out)
    else:
        content = content.replace(anchor, anchor + "\n" + good_imports)

with open(collator_path, "w") as f:
    f.write(content)

print("collator.py patched (module-level imports)")
print("All patches applied")
