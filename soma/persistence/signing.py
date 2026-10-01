"""Ed25519 signatures for released artifacts (brains, engine binaries).

Standard library only: Ed25519 follows the RFC 8032 reference
(section 6), checked against its test vectors and, where installed,
against the ``cryptography`` package (tests/test_r8_signing.py). Speed
does not matter here because only one hash is signed per artifact.

A signature is a JSON envelope stored next to the artifact as
``<artifact>.sig``::

    {"format": "soma-signature-v1", "algorithm": "ed25519",
     "key_id": "<16 hex>", "artifact_name": "...", "artifact_bytes": n,
     "artifact_sha256": "<hex>", "signed_utc": "...", "signature": "<hex>"}

The signed message is ``b"SOMA-SIGNATURE-V1\\n"`` followed by the
canonical JSON of every field except ``signature``. ``key_id`` is the
first 16 hex digits of SHA-256(public key). Verification needs a keyring
of trusted public keys. Secret keys never leave the signer's machine, and
none is stored in this repository.
"""

import hashlib
import json
import os
import time

FORMAT = "soma-signature-v1"
DOMAIN = b"SOMA-SIGNATURE-V1\n"
SIGNED_FIELDS = ("format", "algorithm", "key_id", "artifact_name", "artifact_bytes",
                 "artifact_sha256", "signed_utc")

# --- Ed25519 (RFC 8032, section 6) -------------------------------------------

_P = 2 ** 255 - 19
_L = 2 ** 252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _sha512(data):
    return hashlib.sha512(data).digest()


def _sha512_mod_l(data):
    return int.from_bytes(_sha512(data), "little") % _L


def _point_add(a, b):
    x1, y1, z1, t1 = a
    x2, y2, z2, t2 = b
    first = (y1 - x1) * (y2 - x2) % _P
    second = (y1 + x1) * (y2 + x2) % _P
    third = t1 * 2 * _D * t2 % _P
    fourth = z1 * 2 * z2 % _P
    e, f, g, h = second - first, fourth - third, fourth + third, second + first
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _point_mul(scalar, point):
    result = (0, 1, 1, 0)
    while scalar > 0:
        if scalar & 1:
            result = _point_add(result, point)
        point = _point_add(point, point)
        scalar >>= 1
    return result


def _point_equal(a, b):
    x1, y1, z1, _ = a
    x2, y2, z2, _ = b
    return (x1 * z2 - x2 * z1) % _P == 0 and (y1 * z2 - y2 * z1) % _P == 0


def _recover_x(y, sign):
    if y >= _P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P)
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_GY = 4 * pow(5, _P - 2, _P) % _P
_GX = _recover_x(_GY, 0)
_G = (_GX, _GY, 1, _GX * _GY % _P)


def _compress(point):
    x, y, z, _ = point
    inverse = pow(z, _P - 2, _P)
    x, y = x * inverse % _P, y * inverse % _P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(data):
    if len(data) != 32:
        return None
    y = int.from_bytes(data, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def _secret_expand(secret):
    if len(secret) != 32:
        raise ValueError("Ed25519 secret key must be 32 bytes")
    digest = _sha512(secret)
    scalar = int.from_bytes(digest[:32], "little")
    scalar &= (1 << 254) - 8
    scalar |= 1 << 254
    return scalar, digest[32:]


def public_key(secret):
    """32-byte public key for a 32-byte secret seed."""
    scalar, _ = _secret_expand(secret)
    return _compress(_point_mul(scalar, _G))


def sign_bytes(secret, message):
    scalar, prefix = _secret_expand(secret)
    public = _compress(_point_mul(scalar, _G))
    nonce = _sha512_mod_l(prefix + message)
    commitment = _compress(_point_mul(nonce, _G))
    challenge = _sha512_mod_l(commitment + public + message)
    return commitment + int.to_bytes((nonce + challenge * scalar) % _L, 32, "little")


def verify_bytes(public, message, signature):
    if len(public) != 32 or len(signature) != 64:
        return False
    point = _decompress(public)
    commitment = _decompress(signature[:32])
    if point is None or commitment is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    challenge = _sha512_mod_l(signature[:32] + public + message)
    return _point_equal(_point_mul(s, _G),
                        _point_add(commitment, _point_mul(challenge, point)))


# --- keys, keyrings, envelopes -------------------------------------------------

def key_id(public):
    return hashlib.sha256(public).hexdigest()[:16]


def generate_secret():
    return os.urandom(32)


def write_secret_key(path, secret=None):
    """Create a new secret key file (mode 0600); refuse to overwrite one."""
    secret = secret or generate_secret()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"format": "soma-secret-key-v1", "algorithm": "ed25519",
                   "secret": secret.hex()}, handle)
        handle.write("\n")
    public = public_key(secret)
    return {"key_id": key_id(public), "public_key": public.hex()}


def read_secret_key(path):
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("format") != "soma-secret-key-v1" or payload.get("algorithm") != "ed25519":
        raise ValueError("not a SOMA Ed25519 secret key")
    secret = bytes.fromhex(payload["secret"])
    if len(secret) != 32:
        raise ValueError("secret key must be 32 bytes")
    return secret


def load_keyring(paths):
    """Trusted public keys from keyring files: {key_id: public bytes}.

    A keyring file is {"keys": [{"key_id", "public_key", "label"}, ...]}.
    Missing files are skipped. A key whose id does not match its bytes is
    rejected.
    """
    keys = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path) as handle:
            for entry in json.load(handle).get("keys", []):
                public = bytes.fromhex(entry["public_key"])
                if len(public) != 32 or key_id(public) != entry["key_id"]:
                    raise ValueError("keyring entry %r is inconsistent" % entry.get("key_id"))
                keys[entry["key_id"]] = public
    return keys


def default_keyring_paths():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    home = os.environ.get("SOMA_HOME", os.path.join(os.path.expanduser("~"), ".soma"))
    return [os.path.join(root, "configs", "trusted_keys.json"),
            os.path.join(home, "trusted_keys.json")]


def _file_digest(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _message(envelope):
    return DOMAIN + json.dumps({name: envelope[name] for name in SIGNED_FIELDS},
                               sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_file(path, secret, signature_path=None):
    """Write ``<path>.sig`` and return the envelope."""
    sha256, size = _file_digest(path)
    envelope = {
        "format": FORMAT,
        "algorithm": "ed25519",
        "key_id": key_id(public_key(secret)),
        "artifact_name": os.path.basename(path),
        "artifact_bytes": size,
        "artifact_sha256": sha256,
        "signed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    envelope["signature"] = sign_bytes(secret, _message(envelope)).hex()
    with open(signature_path or path + ".sig", "w") as handle:
        json.dump(envelope, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return envelope


def verify_file(path, keyring, signature_path=None):
    """Verify ``path`` against its envelope and a keyring.

    Returns the envelope. Raises ValueError naming the reason otherwise:
    missing or malformed signature, unknown key, bad signature, or an
    artifact whose size or hash differs from what was signed.
    """
    signature_path = signature_path or path + ".sig"
    if not os.path.exists(signature_path):
        raise ValueError("no signature at %s" % signature_path)
    with open(signature_path) as handle:
        try:
            envelope = json.load(handle)
        except ValueError:
            raise ValueError("signature is not valid JSON")
    if not isinstance(envelope, dict) or envelope.get("format") != FORMAT \
            or envelope.get("algorithm") != "ed25519" \
            or any(name not in envelope for name in SIGNED_FIELDS + ("signature",)):
        raise ValueError("malformed signature envelope")
    public = keyring.get(envelope["key_id"])
    if public is None:
        raise ValueError("signed by untrusted key %s" % envelope["key_id"])
    try:
        signature = bytes.fromhex(envelope["signature"])
    except ValueError:
        raise ValueError("malformed signature bytes")
    if not verify_bytes(public, _message(envelope), signature):
        raise ValueError("signature does not verify")
    sha256, size = _file_digest(path)
    if size != envelope["artifact_bytes"] or sha256 != envelope["artifact_sha256"]:
        raise ValueError("artifact does not match its signature")
    return envelope
