"""Download the exact wheels in a validated pip report; verify every digest.

Useful when the host has better registry connectivity than Docker. No installed
package or guessed PyPI hash is used. The same directory is safe to resume.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
import urllib.request
import re
import time
from generate_lock import render_lock


def download(report, directory, *, nvidia_mirror=False):
    render_lock(report, platform="linux", python_version="3.12")
    directory.mkdir(parents=True, exist_ok=True)
    def fetch(item):
        url = item["download_info"]["url"]
        digest = item["download_info"]["archive_info"]["hashes"]["sha256"]
        filename = Path(unquote(urlsplit(url).path)).name
        if nvidia_mirror and item['metadata']['name'].startswith('nvidia-'):
            # NVIDIA's official index publishes the identical CUDA artifacts.
            # The original resolver's SHA256 remains mandatory, including for
            # resumed partial ranges; the lock keeps its original provenance.
            url = 'https://pypi.nvidia.com/' + item['metadata']['name'] + '/' + filename
        target = directory / filename
        def hashed(path):
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()
        if target.exists() and hashed(target) == digest:
            return filename
        part = target.with_suffix(target.suffix + ".part")
        if part.exists() and hashed(part) == digest:
            part.replace(target)
            return filename
        # Bounded ranges avoid losing a multi-GB CUDA download to one broken
        # stream. Keep only completed ranges; the final full SHA remains decisive.
        offset = part.stat().st_size if part.exists() else 0
        total = None
        chunk_size = 4 * 1024 * 1024
        last_report = offset
        while total is None or offset < total:
            if total is not None and offset > total:
                raise ValueError(f"partial file larger than artifact: {filename}")
            end = offset + chunk_size - 1
            if total is not None: end = min(end, total - 1)
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}", "Cache-Control": "no-cache"})
                    with urllib.request.urlopen(req, timeout=30) as response:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if response.status != 206 or not match or int(match[1]) != offset:
                            raise ValueError(f"invalid range response: {filename}")
                        actual_end, size = int(match[2]), int(match[3])
                        if actual_end > end or (total is not None and size != total):
                            raise ValueError(f"inconsistent range response: {filename}")
                        data = response.read(actual_end-offset+2)
                        if len(data) != actual_end-offset+1:
                            raise ValueError(f"incomplete range: {filename}")
                    total = size
                    with part.open("ab") as stream:
                        stream.write(data)
                    offset += len(data)
                    if offset-last_report >= 64*1024*1024:
                        print(f"{filename}: {offset}/{total} bytes", flush=True);last_report=offset
                    break
                except Exception:
                    if attempt == 3: raise
                    time.sleep(2**attempt)
        if hashed(part) != digest:
            raise ValueError(f"wheel hash mismatch: {filename}; retained partial for inspection")
        part.replace(target)
        return filename
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, item) for item in report["install"]]
        failures = []
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                print(failures[-1], flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} artifact downloads failed; valid files and resumable ranges retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--nvidia-mirror", action="store_true", help="Use NVIDIA's official mirror for NVIDIA wheels, retaining the resolver SHA256")
    args = parser.parse_args()
    download(json.loads(args.report.read_text(encoding="utf-8")), args.directory, nvidia_mirror=args.nvidia_mirror)
