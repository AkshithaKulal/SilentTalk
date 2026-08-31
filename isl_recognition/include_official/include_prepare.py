#!/usr/bin/env python3
"""Download missing INCLUDE zips from Zenodo, then extract into extracted/.

Copy this file onto the office PC (or run it from the SilentTalk repo).
Stdlib only. Python 3.8+.

Office (dataset is already at F:\\include_dataset):

    cd F:\\include_dataset
    python include_prepare.py

That will:
  1. Delete junk leftover names like Adjectives_4of8.zip@download=1
  2. Ask Zenodo for the official file list + sizes
  3. Re-download ONLY zips that are missing, 0 bytes, or the wrong size
     (today that should be Seasons_1of1.zip and Society_3of3.zip)
  4. Extract zips that are not already in extracted/

Do NOT pass --force-download. That would re-get ~57 GB you already have.

After it finishes, re-run audit_include_folder.py inside extracted/.
Expect ~4292 videos, 263 words, 15 categories. Then we train a NEW
sequence model (not the old 84-class MLP).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records/4010759"
ZENODO_FILE = "https://zenodo.org/records/4010759/files/{name}?download=1"
USER_AGENT = "SilentTalk-INCLUDE-prepare/1.0"
MIN_BYTES_IF_NO_API = 10 * 1024 * 1024  # 10 MB — catches 0-byte failed downloads
CHUNK = 1024 * 1024

# Official 44 category zips (fallback if the API is down).
FALLBACK_ZIPS = [
    "Adjectives_1of8.zip",
    "Adjectives_2of8.zip",
    "Adjectives_3of8.zip",
    "Adjectives_4of8.zip",
    "Adjectives_5of8.zip",
    "Adjectives_6of8.zip",
    "Adjectives_7of8.zip",
    "Adjectives_8of8.zip",
    "Animals_1of2.zip",
    "Animals_2of2.zip",
    "Clothes_1of2.zip",
    "Clothes_2of2.zip",
    "Colours_1of2.zip",
    "Colours_2of2.zip",
    "Days_and_Time_1of3.zip",
    "Days_and_Time_2of3.zip",
    "Days_and_Time_3of3.zip",
    "Electronics_1of2.zip",
    "Electronics_2of2.zip",
    "Greetings_1of2.zip",
    "Greetings_2of2.zip",
    "Home_1of4.zip",
    "Home_2of4.zip",
    "Home_3of4.zip",
    "Home_4of4.zip",
    "Jobs_1of2.zip",
    "Jobs_2of2.zip",
    "Means_of_Transportation_1of2.zip",
    "Means_of_Transportation_2of2.zip",
    "People_1of5.zip",
    "People_2of5.zip",
    "People_3of5.zip",
    "People_4of5.zip",
    "People_5of5.zip",
    "Places_1of4.zip",
    "Places_2of4.zip",
    "Places_3of4.zip",
    "Places_4of4.zip",
    "Pronouns_1of2.zip",
    "Pronouns_2of2.zip",
    "Seasons_1of1.zip",
    "Society_1of3.zip",
    "Society_2of3.zip",
    "Society_3of3.zip",
]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}


def gb(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB"


def find_root(start: Path) -> Path:
    if (start / "extracted").is_dir() or list(start.glob("*_1of*.zip")):
        return start
    if start.name.lower() == "extracted":
        return start.parent
    return start


def cleanup_junk(root: Path) -> None:
    for p in root.iterdir():
        if not p.is_file():
            continue
        if "@download" in p.name or p.name.endswith(".crdownload") or p.name.endswith(".tmp"):
            print(f"  delete junk: {p.name}")
            p.unlink()


def fetch_zenodo_files() -> list[dict]:
    req = urllib.request.Request(ZENODO_API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = []
    for f in data.get("files") or []:
        name = f.get("key") or f.get("filename") or ""
        if not name.lower().endswith(".zip"):
            continue
        links = f.get("links") or {}
        out.append(
            {
                "name": name,
                "size": int(f.get("size") or 0),
                "checksum": f.get("checksum") or "",
                "url": links.get("self") or links.get("download") or ZENODO_FILE.format(name=name),
            }
        )
    return out


def catalog() -> list[dict]:
    try:
        files = fetch_zenodo_files()
        if files:
            print(f"Zenodo API: {len(files)} zip files")
            return files
    except Exception as exc:  # noqa: BLE001
        print(f"Zenodo API unavailable ({exc}). Using name list only.")
    return [
        {"name": n, "size": 0, "checksum": "", "url": ZENODO_FILE.format(name=n)}
        for n in FALLBACK_ZIPS
    ]


def needs_download(path: Path, expected: int) -> str | None:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size == 0:
        return "empty (0 bytes)"
    if expected and abs(size - expected) > max(1024 * 1024, int(expected * 0.002)):
        return f"size mismatch local={size} zenodo={expected}"
    if not expected and size < MIN_BYTES_IF_NO_API:
        return f"too small ({size} bytes)"
    return None


def download_one(url: str, dest: Path, expected: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    already = tmp.stat().st_size if tmp.exists() else 0

    headers = {"User-Agent": USER_AGENT}
    if already > 0:
        headers["Range"] = f"bytes={already}-"
        print(f"  resume {dest.name} from {gb(already)}")

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as err:
        if err.code in (416, 501) and already:
            tmp.unlink(missing_ok=True)
            already = 0
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            resp = urllib.request.urlopen(req, timeout=120)
        else:
            raise

    mode = "ab" if already and resp.status == 206 else "wb"
    if mode == "wb" and already:
        already = 0
        tmp.unlink(missing_ok=True)

    total = expected or already
    try:
        length = resp.headers.get("Content-Length")
        if length and resp.status != 206:
            total = int(length)
        elif length and resp.status == 206:
            total = already + int(length)
    except ValueError:
        pass

    last = time.time()
    with open(tmp, mode) as fh:
        got = already
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            now = time.time()
            if now - last >= 2:
                last = now
                if total:
                    print(f"\r  {dest.name}: {gb(got)} / {gb(total)}", end="", flush=True)
                else:
                    print(f"\r  {dest.name}: {gb(got)}", end="", flush=True)
    print()
    tmp.replace(dest)
    final = dest.stat().st_size
    if expected and abs(final - expected) > max(1024 * 1024, int(expected * 0.002)):
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name} finished at {final} bytes, expected {expected}")
    print(f"  OK {dest.name} ({gb(final)})")


def zip_already_extracted(extracted: Path, zip_path: Path) -> bool:
    marker = extracted / zip_path.stem
    if not marker.is_dir():
        return False
    n = 0
    for p in marker.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            n += 1
            if n >= 3:
                return True
    return n > 0


def extract_one(zip_path: Path, extracted: Path) -> None:
    extracted.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extracted)
    print(f"  extracted {zip_path.name}")


def cmd_download(root: Path, force: bool) -> list[str]:
    print("=== INCLUDE zip check / download ===")
    print(f"folder: {root}")
    cleanup_junk(root)
    files = catalog()
    bad: list[str] = []
    ok_n = 0
    for info in files:
        name = info["name"]
        dest = root / name
        reason = needs_download(dest, info["size"])
        if reason is None and not force:
            print(f"  keep {name} ({gb(dest.stat().st_size)})")
            ok_n += 1
            continue
        print(f"  DOWNLOAD {name}  ({reason or 'forced'})")
        try:
            download_one(info["url"], dest, info["size"])
            ok_n += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name}: {exc}", file=sys.stderr)
            bad.append(name)
    print(f"zips OK: {ok_n}   failed: {len(bad)}")
    return bad


def cmd_extract(root: Path, force: bool) -> list[str]:
    extracted = root / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    print("=== extract into", extracted, "===")
    failed: list[str] = []
    zips = sorted(p for p in root.glob("*.zip") if "@download" not in p.name)
    for zp in zips:
        if zp.stat().st_size < MIN_BYTES_IF_NO_API:
            print(f"  skip empty/small {zp.name}")
            failed.append(zp.name)
            continue
        if zip_already_extracted(extracted, zp) and not force:
            print(f"  already extracted {zp.stem}")
            continue
        try:
            extract_one(zp, extracted)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL extract {zp.name}: {exc}", file=sys.stderr)
            failed.append(zp.name)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Download + extract INCLUDE zips")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Folder that contains the .zip files (default: this script's folder, or F:\\include_dataset)",
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download every zip. Do not use this — you already have ~42 good files.",
    )
    parser.add_argument("--force-extract", action="store_true", help="Re-extract zips even if folders exist")
    args = parser.parse_args()

    root = args.root
    if root is None:
        here = Path(__file__).resolve().parent
        if list(here.glob("*_1of*.zip")):
            root = here
        elif Path(r"F:\include_dataset").exists():
            root = Path(r"F:\include_dataset")
        else:
            root = Path.cwd()
    root = find_root(root.resolve())
    if not root.is_dir():
        print(f"ERROR: {root} is not a folder", file=sys.stderr)
        return 1

    do_dl = not args.extract_only
    do_ex = not args.download_only
    failed: list[str] = []
    if do_dl:
        failed.extend(cmd_download(root, args.force_download))
    if do_ex:
        failed.extend(cmd_extract(root, args.force_extract))

    print()
    print("Next: copy audit_include_folder.py into extracted\\ and run:")
    print("  python audit_include_folder.py")
    print("Need ~4292 videos, 263 words, 15 categories before training.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
