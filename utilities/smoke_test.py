#!/usr/bin/env python3
"""
Lightweight smoke test for figures:
- asserts at least one PDF and one PNG in figures/output
- checks magic headers and non-zero sizes
- writes manifest.json / manifest.txt with sha256 checksums
"""
import argparse, hashlib, json, sys
from pathlib import Path

PDF_MAGIC = b"%PDF"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

def sha256_of(p: Path, buf=1<<16) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(buf)
            if not b: break
            h.update(b)
    return h.hexdigest()

def ok_header(p: Path) -> bool:
    head = p.read_bytes()[:8]
    if p.suffix.lower()==".pdf": return head.startswith(PDF_MAGIC)
    if p.suffix.lower()==".png": return head.startswith(PNG_MAGIC)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="figures/output")
    args = ap.parse_args()
    out = Path(args.dir)
    if not out.exists(): print(f"ERROR: {out} not found", file=sys.stderr); sys.exit(1)
    pdfs = sorted(out.glob("*.pdf"))
    pngs = sorted(out.glob("*.png"))
    if not pdfs: print("ERROR: no PDF files", file=sys.stderr); sys.exit(1)
    if not pngs: print("ERROR: no PNG files", file=sys.stderr); sys.exit(1)
    manifest, bad = [], 0
    for p in pdfs+pngs:
        size = p.stat().st_size
        hdr  = ok_header(p)
        if not (hdr and size>0):
            print(f"FAIL: {p.name}: header_ok={hdr} size={size}", file=sys.stderr)
            bad += 1
        manifest.append({"file": p.name, "size": size, "sha256": sha256_of(p), "header_ok": hdr})
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2))
    (out/"manifest.txt").write_text("\n".join(f"{m['file']}\t{m['size']} B\t{m['sha256']}" for m in manifest))
    if bad: sys.exit(1)
    print(f"Smoke test OK: {len(manifest)} files validated.")

if __name__ == "__main__":
    main()
