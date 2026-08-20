"""
Tulu language-tag alias for IndicTrans2 Stage 2 (KN->Tulu).

IndicTrans2 has no tcy_Knda token. train_lora.py skips pairs where
src_lang == tgt_lang, so Tulu cannot reuse kan_Knda on both sides.

We map Tulu onto an existing unused FLORES tag: sat_Olck (Santali).
That tag is only an alias - not real Santali text.

Default sat_Olck -> ISO "or" in IndicProcessor would Odia-normalize
Kannada-script Tulu. apply_tulu_processor_override() forces sat_Olck -> "kn".
"""

SRC_KN = "kan_Knda"
TULU_LANG_TAG = "sat_Olck"
PAIR_DIR = f"{SRC_KN}-{TULU_LANG_TAG}"


def apply_tulu_processor_override(processor, alias: str = TULU_LANG_TAG) -> None:
    """Force the Tulu stand-in tag to use Kannada (kn) tokenization/normalization."""
    if not hasattr(processor, "_flores_codes"):
        raise AttributeError(
            "IndicProcessor has no _flores_codes; cannot apply Tulu alias override"
        )
    processor._flores_codes[alias] = "kn"


def should_apply_tulu_override(src_lang_list, tgt_lang_list, alias: str = TULU_LANG_TAG) -> bool:
    """True when the Tulu stand-in tag appears as src or tgt."""
    langs = set(src_lang_list or []) | set(tgt_lang_list or [])
    return alias in langs
