"""Download one pinned official model snapshot, with no duplicate weight cache."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
from huggingface_hub import HfApi, snapshot_download

# Retained as historical failed-route code. Do not silently reuse the Mac path.
raise RuntimeError("Disabled after user restricted downloads to server-side mirrors; use mirror_download.py")

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
manifest = out / "source.json"
api = HfApi()
revision = json.loads(manifest.read_text())["revision"] if manifest.exists() else "main"
official = json.loads((Path(__file__).parent / "official_source.json").read_text())
revision = official["revision"]
info = api.model_info("Qwen/Qwen3-8B", revision=revision, files_metadata=True)
if info.sha != revision:
    raise RuntimeError("Mirror revision does not match independently fetched official metadata")
files = [f for f in info.siblings if f.rfilename.endswith((".json", ".safetensors", ".txt", ".jinja"))]
required = sum(f.size or 0 for f in files)
source = dict(repo="Qwen/Qwen3-8B", revision=info.sha, expected_bytes=required,
              files={f.rfilename: dict(size=f.size, sha256=f.lfs.sha256 if f.lfs else None)
                     for f in files}, pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat())
for name, expected_hash in official["sha256"].items():
    if source["files"][name]["sha256"] != expected_hash:
        raise RuntimeError(f"Mirror metadata differs from official metadata: {name}")
if shutil.disk_usage(out).free < required + 15 * 2**30 and not manifest.exists():
    raise RuntimeError("Insufficient model-download space including 15GiB reserve")
manifest.write_text(json.dumps(source, indent=2))
print(json.dumps(dict(event="download_start", **source)), flush=True)
snapshot_download(repo_id=source["repo"], revision=info.sha,
                  local_dir=str(out / "original"), allow_patterns=[f.rfilename for f in files],
                  max_workers=3)
for name, meta in source["files"].items():
    path = out / "original" / name
    if path.stat().st_size != meta["size"]:
        raise RuntimeError(f"Size mismatch: {name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if meta["sha256"] and actual != meta["sha256"]:
        raise RuntimeError(f"SHA mismatch: {name}")
    meta["verified_sha256"] = actual
    print(json.dumps(dict(event="file_verified", name=name, sha256=actual)), flush=True)
source["complete"] = True
(out / "source_verified.json").write_text(json.dumps(source, indent=2))
print(json.dumps(dict(event="download_verified", revision=info.sha, bytes=required)), flush=True)
