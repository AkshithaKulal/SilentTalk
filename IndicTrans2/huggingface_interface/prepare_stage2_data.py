"""
Prepare Stage 2 data for Kannada->Tulu (Tulu stored under sat_Olck alias).

Modes:
    real                - kn_tcy_raw.csv (kannada, tulu) -> kan_Knda-sat_Olck
    synthetic_degraded  - synthetic_en_tulu_fixed.csv -> eng_Latn-sat_Olck
                                                (EN->Tulu continuity only; NOT competitive KN->Tulu)
    auto                - try real first, fallback to synthetic_degraded

No public GitHub mirror of DravidianLangTech-2022 KN–TCY exists; drop
kn_tcy_raw.csv from organizers (e.g. Dr. Asha) for real mode.

Usage (from IndicTrans2/huggingface_interface/):
    python prepare_stage2_data.py --mode auto
    python prepare_stage2_data.py --mode real
    python prepare_stage2_data.py --mode synthetic_degraded
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from tulu_lang_alias import PAIR_DIR, SRC_KN, TULU_LANG_TAG

ROOT = Path(__file__).resolve().parent
KN_TCY_CSV = ROOT / "kn_tcy_raw.csv"
SYNTHETIC_CSV = ROOT / "synthetic_en_tulu_fixed.csv"
STAGE2_DATA = ROOT / "stage2_data"


def _clean_lines(series) -> list[str]:
    return [
        " ".join(str(x).replace("\r", " ").replace("\n", " ").split())
        for x in series.tolist()
    ]


def _write_pair(
    train_src,
    train_tgt,
    dev_src,
    dev_tgt,
    pair_name: str,
    src_ext: str,
    tgt_ext: str,
) -> None:
    for split, src_lines, tgt_lines in (
        ("train", train_src, train_tgt),
        ("dev", dev_src, dev_tgt),
    ):
        if len(src_lines) != len(tgt_lines):
            raise ValueError(
                f"{split}: src/tgt length mismatch ({len(src_lines)} vs {len(tgt_lines)})"
            )
        out_dir = STAGE2_DATA / split / pair_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{split}.{src_ext}").write_text(
            "\n".join(src_lines) + ("\n" if src_lines else ""),
            encoding="utf-8",
        )
        (out_dir / f"{split}.{tgt_ext}").write_text(
            "\n".join(tgt_lines) + ("\n" if tgt_lines else ""),
            encoding="utf-8",
        )


def _split_df(df: pd.DataFrame, seed: int = 42, frac: float = 0.9):
    if "split" in df.columns:
        split_col = df["split"].astype(str).str.lower()
        train_df = df[split_col.isin(["train", "training"])]
        dev_df = df[split_col.isin(["dev", "valid", "validation", "test"])]
        if len(train_df) == 0 or len(dev_df) == 0:
            train_df = df.sample(frac=frac, random_state=seed)
            dev_df = df.drop(train_df.index)
        return train_df, dev_df
    train_df = df.sample(frac=frac, random_state=seed)
    dev_df = df.drop(train_df.index)
    return train_df, dev_df


def prepare_real() -> None:
    if not KN_TCY_CSV.exists():
        raise FileNotFoundError(
            f"{KN_TCY_CSV.name} not found.\n"
            "Drop DravidianLangTech KN-TCY here with columns: kannada, tulu\n"
            "(No public GitHub mirror - obtain from organizers / Dr. Asha.)\n"
            "Or run: python prepare_stage2_data.py --mode synthetic_degraded"
        )

    df = pd.read_csv(KN_TCY_CSV).dropna()
    for col in ("kannada", "tulu"):
        if col not in df.columns:
            raise ValueError(f"{KN_TCY_CSV.name} must have columns: kannada, tulu")

    df["kannada"] = df["kannada"].astype(str).str.strip()
    df["tulu"] = df["tulu"].astype(str).str.strip()
    df = df[(df["kannada"] != "") & (df["tulu"] != "")]
    df = df.drop_duplicates(subset=["kannada", "tulu"])

    train_df, dev_df = _split_df(df)
    _write_pair(
        _clean_lines(train_df["kannada"]),
        _clean_lines(train_df["tulu"]),
        _clean_lines(dev_df["kannada"]),
        _clean_lines(dev_df["tulu"]),
        pair_name=PAIR_DIR,
        src_ext=SRC_KN,
        tgt_ext=TULU_LANG_TAG,
    )
    print("MODE: real KN-TCY (Kannada->Tulu via sat_Olck alias)")
    print(f"Source CSV: {KN_TCY_CSV.name}")
    print(f"Pair dir:   {PAIR_DIR}")
    print(f"Train: {len(train_df)} | Dev: {len(dev_df)}")
    print(
        f"Wrote {STAGE2_DATA / 'train' / PAIR_DIR} and {STAGE2_DATA / 'dev' / PAIR_DIR}"
    )


def prepare_synthetic_degraded() -> None:
    """EN->Tulu under eng_Latn-sat_Olck. Documented as non-competitive."""
    if not SYNTHETIC_CSV.exists():
        raise FileNotFoundError(
            f"{SYNTHETIC_CSV.name} not found — cannot run synthetic_degraded mode."
        )

    df = pd.read_csv(SYNTHETIC_CSV).dropna()
    # tolerate english/tulu or eng/tulu
    eng_col = "english" if "english" in df.columns else ("eng" if "eng" in df.columns else None)
    tulu_col = "tulu" if "tulu" in df.columns else None
    if eng_col is None or tulu_col is None:
        raise ValueError(
            f"{SYNTHETIC_CSV.name} must have columns english,tulu (got {list(df.columns)})"
        )

    df[eng_col] = df[eng_col].astype(str).str.strip()
    df[tulu_col] = df[tulu_col].astype(str).str.strip()
    df = df[(df[eng_col] != "") & (df[tulu_col] != "")]
    df = df.drop_duplicates(subset=[eng_col, tulu_col])

    train_df, dev_df = _split_df(df)
    pair = f"eng_Latn-{TULU_LANG_TAG}"
    _write_pair(
        _clean_lines(train_df[eng_col]),
        _clean_lines(train_df[tulu_col]),
        _clean_lines(dev_df[eng_col]),
        _clean_lines(dev_df[tulu_col]),
        pair_name=pair,
        src_ext="eng_Latn",
        tgt_ext=TULU_LANG_TAG,
    )
    print("MODE: synthetic_degraded fallback (EN->Tulu via sat_Olck alias)")
    print(
        "WARNING: Not real KN-TCY. Scores are non-competitive continuity experiments only."
    )
    print(f"Source CSV: {SYNTHETIC_CSV.name}")
    print(f"Pair dir:   {pair}")
    print(f"Train: {len(train_df)} | Dev: {len(dev_df)}")
    print(f"Wrote {STAGE2_DATA / 'train' / pair} and {STAGE2_DATA / 'dev' / pair}")


def main():
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Prepare Stage 2 KN-TCY / synthetic data")
    parser.add_argument(
        "--mode",
        choices=["auto", "real", "synthetic_degraded"],
        default="auto",
        help="auto = try real else synthetic_degraded; real = kn_tcy_raw.csv KN->Tulu; synthetic_degraded = EN->Tulu fallback",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    if args.mode == "real":
        prepare_real()
    elif args.mode == "synthetic_degraded":
        prepare_synthetic_degraded()
    else:
        if KN_TCY_CSV.exists():
            print("AUTO MODE: Found kn_tcy_raw.csv, preparing real KN-TCY data.")
            prepare_real()
        else:
            print("AUTO MODE: kn_tcy_raw.csv missing, using synthetic_degraded fallback.")
            prepare_synthetic_degraded()


if __name__ == "__main__":
    main()
