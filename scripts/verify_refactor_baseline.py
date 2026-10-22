"""Read-only verification of the initial refactor handoff. Standard library only."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def verify(root: Path, manifest: dict) -> dict:
    checked = 0
    errors = []
    seen = set()
    for entry in manifest["files"]:
        if not entry.get("protected"):
            continue
        rel = PurePosixPath(entry["path"])
        if rel.is_absolute() or ".." in rel.parts or ":" in str(rel) or "\\" in str(rel):
            errors.append({"path":str(rel), "reason":"unsafe_manifest_path"})
            continue
        if str(rel) in seen:
            errors.append({"path":str(rel), "reason":"duplicate_manifest_path"})
            continue
        seen.add(str(rel))
        path = root.joinpath(*rel.parts)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            errors.append({"path":str(rel), "reason":"symlink_or_path_escape"})
            continue
        if not path.is_file():
            errors.append({"path":str(rel), "reason":"missing"})
            continue
        data = path.read_bytes()
        expected = entry["git_blob_sha1"]
        raw_sha = git_blob(data)
        actual = raw_sha
        # Windows checkouts may apply CRLF via .gitattributes. Git's clean filter
        # is authoritative; hash-object without -w never writes an object.
        if raw_sha != expected:
            try:
                result = subprocess.run(
                    ["git", "hash-object", "--path=" + str(rel), "--", str(path)],
                    cwd=root, capture_output=True, text=True, check=True,
                )
                actual = result.stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                pass
        checked += 1
        if actual != expected:
            errors.append({"path":str(rel), "reason":"changed", "expected":expected,
                           "actual":actual, "raw_sha":raw_sha})
    if checked == 0:
        errors.append({"path":"", "reason":"no_protected_files_checked"})
    return {"status":"PASS" if not errors else "FAIL", "checked":checked,
            "expected_parent":manifest.get("parent_commit"), "errors":errors,
            "scope":"Initial protected files only; not a functional or security test."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest or root / "docs/refactor/baseline-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = verify(root, manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status":"ERROR", "reason":str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
