"""Binary `.soma` container, format v1.

Layout (big-endian):
  magic      8 bytes  b"SOMA0001"
  nchunks    uint32
  per chunk: name_len uint16, name bytes, payload_len uint64,
             payload bytes, sha256(payload) 32 bytes
  footer_sha uint32-length-prefixed sha256 over everything before it

Chunks: manifest.json, brain.json, dialogue.json, episodic.json.
Every hash is verified on read; any mismatch or truncation raises.
Atomic writes via temp file + rename + fsync.
"""

import hashlib
import json
import os
import struct
import tempfile

SOMAFORMAT_VERSION = 1
MAGIC = b"SOMA0001"
CHUNK_NAMES = ("manifest.json", "brain.json", "dialogue.json", "episodic.json")


def _sha256(data):
    return hashlib.sha256(data).digest()


def write_soma(path, chunks):
    """Write chunks {name: bytes} atomically with full integrity hashes."""
    if set(chunks.keys()) != set(CHUNK_NAMES):
        raise ValueError("soma v1 requires exactly chunks %s" % (sorted(CHUNK_NAMES),))
    for name, payload in chunks.items():
        if not isinstance(payload, (bytes, bytearray)):
            raise ValueError("chunk %s must be bytes" % name)
    body = bytearray(MAGIC)
    body += struct.pack(">I", len(CHUNK_NAMES))
    for name in CHUNK_NAMES:
        payload = bytes(chunks[name])
        name_bytes = name.encode("ascii")
        body += struct.pack(">H", len(name_bytes))
        body += name_bytes
        body += struct.pack(">Q", len(payload))
        body += payload
        body += _sha256(payload)
    body += _sha256(bytes(body))
    directory = os.path.dirname(os.path.abspath(path)) or "."
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".soma-tmp-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(bytes(body))
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return {"path": os.path.abspath(path), "bytes": len(body),
            "sha256": hashlib.sha256(bytes(body)).hexdigest()}


def read_soma(path):
    """Read and fully verify a `.soma` container. Returns {name: bytes}."""
    with open(path, "rb") as handle:
        data = handle.read()
    offset = 0
    if len(data) < 8 + 4 + 32:
        raise ValueError("soma file too small")
    magic = data[0:8]
    offset = 8
    if magic != MAGIC:
        raise ValueError("bad soma magic: %r" % (magic,))
    (nchunks,) = struct.unpack_from(">I", data, offset)
    offset += 4
    if nchunks != len(CHUNK_NAMES):
        raise ValueError("unexpected soma chunk count: %d" % nchunks)
    chunks = {}
    for _ in range(nchunks):
        (name_len,) = struct.unpack_from(">H", data, offset)
        offset += 2
        if offset + name_len + 8 + 32 > len(data):
            raise ValueError("soma file truncated in chunk header")
        name = data[offset:offset + name_len].decode("ascii")
        offset += name_len
        (payload_len,) = struct.unpack_from(">Q", data, offset)
        offset += 8
        if offset + payload_len + 32 > len(data):
            raise ValueError("soma file truncated in chunk payload")
        payload = data[offset:offset + payload_len]
        offset += payload_len
        digest = data[offset:offset + 32]
        offset += 32
        if digest != _sha256(payload):
            raise ValueError("soma chunk hash mismatch: %s" % name)
        if name in chunks:
            raise ValueError("duplicate soma chunk: %s" % name)
        chunks[name] = payload
    if set(chunks.keys()) != set(CHUNK_NAMES):
        raise ValueError("soma chunk set mismatch: %s" % sorted(chunks.keys()))
    footer = data[offset:]
    if _sha256(data[:offset]) != footer:
        raise ValueError("soma footer hash mismatch")
    for name in CHUNK_NAMES:
        json.loads(chunks[name].decode("utf-8"))
    return chunks
