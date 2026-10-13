#!/usr/bin/env python3
"""Download raw sources into content/data/raw/ and record their sha256 in sources.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
from bible_content.paths import RAW_DIR

WEB_USFM_URL = "https://ebible.org/Scriptures/engwebp_usfm.zip"
XREF_URL = "https://a.openbible.info/data/cross-references.zip"


def download(url: str, name: str, force: bool) -> dict[str, object]:
    dest = RAW_DIR / name
    if force or not dest.exists():
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "bible-chat-content/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            while block := resp.read(1 << 20):
                f.write(block)
    data = dest.read_bytes()
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip file")
    sha = hashlib.sha256(data).hexdigest()
    print(f"  {dest} {len(data):,} bytes sha256={sha}")
    return {"url": url, "file": name, "bytes": len(data), "sha256": sha,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-references", action="store_true",
                        help="also download OpenBible.info cross references (CC-BY)")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    record_path = RAW_DIR / "sources.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    record["web_usfm"] = download(WEB_USFM_URL, "engwebp_usfm.zip", args.force)
    if args.cross_references:
        xref_zip = download(XREF_URL, "cross-references.zip", args.force)
        with zipfile.ZipFile(RAW_DIR / "cross-references.zip") as zf:
            zf.extract("cross_references.txt", RAW_DIR)
        record["openbible_cross_references"] = xref_zip
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {record_path}")


if __name__ == "__main__":
    main()
