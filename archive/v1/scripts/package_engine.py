#!/usr/bin/env python3
"""Package the release engine binaries as soma-engine-<platform>.tar.gz.

The archive holds exactly soma-mixer-serve, soma-mixer, and SHA256SUMS,
with fixed ownership and mtime so identical binaries give an identical
archive. Sign it with `python3 -m soma.service.main sign <archive> --key
<secret key>`; users install it with `engine-install`.
"""

import argparse
import gzip
import hashlib
import io
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from soma.service.distribution import engine_platform  # noqa: E402

BINARIES = ("soma-mixer-serve", "soma-mixer")


def package(binary_dir, out_dir, platform_name=None):
    platform_name = platform_name or engine_platform()
    files = {}
    for name in BINARIES:
        with open(os.path.join(binary_dir, name), "rb") as handle:
            files[name] = handle.read()
    sums = "".join("%s  %s\n" % (hashlib.sha256(files[name]).hexdigest(), name)
                   for name in BINARIES)
    files["SHA256SUMS"] = sums.encode("ascii")
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size = len(files[name])
            info.mode = 0o644 if name == "SHA256SUMS" else 0o755
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(files[name]))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "soma-engine-%s.tar.gz" % platform_name)
    with open(path, "wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as compressed:
            compressed.write(raw.getvalue())
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", default=os.path.join(
        ROOT, "engine", "soma-engine", "target", "release"))
    parser.add_argument("--out", default="dist")
    parser.add_argument("--platform", help="override, e.g. darwin-arm64")
    args = parser.parse_args()
    print(package(args.binaries, args.out, args.platform))
    return 0


if __name__ == "__main__":
    sys.exit(main())
