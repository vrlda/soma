"""Signed distribution: verified import and download of `.soma` brains.

Policy:

- ``import_signed``: if ``<file>.sig`` exists, it must verify against a
  trusted key or the import is refused. An unsigned file imports only when
  ``require_signature`` is false, and the result says ``"unsigned"``.
- ``install_engine``: the same for a prebuilt engine package
  (``soma-engine-<platform>.tar.gz``), extracted to ``$SOMA_HOME/engine``.
- ``download``: fetches the artifact and ``<url>.sig`` over HTTPS (or
  ``file://``), stops at ``max_bytes``, verifies before anything is imported,
  and always requires a trusted signature.

Trusted public keys come from ``configs/trusted_keys.json`` and
``$SOMA_HOME/trusted_keys.json`` (default ``~/.soma``); see
``soma.persistence.signing``.
"""

import json
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request

from ..persistence import signing

DEFAULT_MAX_BYTES = 8 * 1024 ** 3
USER_AGENT = "soma-downloader/1"


def keyring(extra_paths=()):
    return signing.load_keyring(list(signing.default_keyring_paths()) + list(extra_paths))


def user_keyring_path():
    return signing.default_keyring_paths()[1]


def trust_key(public_hex, label, path=None):
    """Add a public key to the user keyring and return its entry."""
    public = bytes.fromhex(public_hex)
    if len(public) != 32:
        raise ValueError("public key must be 32 bytes (64 hex digits)")
    path = path or user_keyring_path()
    payload = {"keys": []}
    if os.path.exists(path):
        with open(path) as handle:
            payload = json.load(handle)
    entry = {"key_id": signing.key_id(public), "public_key": public.hex(), "label": label}
    payload["keys"] = [key for key in payload.get("keys", []) if key["key_id"] != entry["key_id"]]
    payload["keys"].append(entry)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return entry


def check_signature(path, keys, require_signature):
    """Return a provenance record, or raise ValueError per the policy above."""
    if os.path.exists(path + ".sig"):
        envelope = signing.verify_file(path, keys)
        return {"signature": "verified", "key_id": envelope["key_id"],
                "artifact_sha256": envelope["artifact_sha256"],
                "signed_utc": envelope["signed_utc"]}
    if require_signature:
        raise ValueError("artifact is unsigned and a signature is required")
    return {"signature": "unsigned"}


def import_signed(store, path, name, require_signature=False, extra_keyrings=()):
    provenance = check_signature(path, keyring(extra_keyrings), require_signature)
    manifest = store.import_soma(path, name)
    manifest["provenance"] = provenance
    return manifest


def _fetch(url, target, max_bytes):
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("https", "file"):
        raise ValueError("refusing %s URL; use https" % (scheme or "relative"))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    total = 0
    with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise ValueError("download exceeds %d bytes" % max_bytes)
            handle.write(block)
    return total


def download(store, url, name, max_bytes=DEFAULT_MAX_BYTES, keep=None, extra_keyrings=()):
    """Download, verify, then import. Nothing is imported unless it verifies."""
    keys = keyring(extra_keyrings)
    if not keys:
        raise ValueError("no trusted keys; add one with `key-trust` first")
    directory = tempfile.mkdtemp(prefix="soma-download-")
    try:
        artifact = os.path.join(directory, "artifact.soma")
        _fetch(url + ".sig", artifact + ".sig", 1 << 16)
        size = _fetch(url, artifact, max_bytes)
        provenance = check_signature(artifact, keys, require_signature=True)
        manifest = store.import_soma(artifact, name)
        manifest["provenance"] = dict(provenance, source_url=url, bytes=size)
        if keep:
            shutil.copyfile(artifact, keep)
            shutil.copyfile(artifact + ".sig", keep + ".sig")
        return manifest
    finally:
        shutil.rmtree(directory, ignore_errors=True)


ENGINE_MEMBERS = ("soma-mixer-serve", "soma-mixer", "SHA256SUMS")


def engine_platform():
    """Package suffix for this machine, e.g. ``linux-x86_64`` or ``darwin-arm64``."""
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return "%s-%s" % (platform.system().lower(), machine)


def _extract_engine(archive_path, destination):
    """Extract only the expected regular files; refuse anything else."""
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = sorted(member.name for member in members)
        if names != sorted(ENGINE_MEMBERS) or not all(member.isfile() for member in members):
            raise ValueError("unexpected engine package contents: %s" % names)
        staging = tempfile.mkdtemp(prefix=".engine-", dir=os.path.dirname(destination))
        try:
            for member in members:
                source = archive.extractfile(member)
                target = os.path.join(staging, member.name)
                with open(target, "wb") as handle:
                    shutil.copyfileobj(source, handle)
                if member.name != "SHA256SUMS":
                    os.chmod(target, 0o755)
            _check_sums(staging)
            if os.path.isdir(destination):
                shutil.rmtree(destination)
            os.rename(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _check_sums(directory):
    import hashlib
    with open(os.path.join(directory, "SHA256SUMS")) as handle:
        expected = dict(reversed(line.split()) for line in handle if line.strip())
    for name in ("soma-mixer-serve", "soma-mixer"):
        with open(os.path.join(directory, name), "rb") as handle:
            if hashlib.sha256(handle.read()).hexdigest() != expected.get(name):
                raise ValueError("engine package checksum mismatch: %s" % name)


def install_engine(url, max_bytes=256 * 1024 ** 2, extra_keyrings=()):
    """Download a signed engine package, verify it, and install it."""
    from ..memory.engine import installed_binary
    keys = keyring(extra_keyrings)
    if not keys:
        raise ValueError("no trusted keys; add one with `key-trust` first")
    destination = os.path.dirname(installed_binary())
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    directory = tempfile.mkdtemp(prefix="soma-engine-download-")
    try:
        package = os.path.join(directory, "engine.tar.gz")
        _fetch(url + ".sig", package + ".sig", 1 << 16)
        _fetch(url, package, max_bytes)
        provenance = check_signature(package, keys, require_signature=True)
        _extract_engine(package, destination)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    return dict(provenance, source_url=url, installed=destination)
