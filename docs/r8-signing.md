# R8 signatures, downloader, and prebuilt engine

Master plan §22 step 12 (first part). Code:
`soma/persistence/signing.py`, `soma/service/distribution.py`,
`scripts/package_engine.py`, `.github/workflows/release-engine.yml`.
Tests: `tests/test_r8_signing.py`.

## What exists

| Piece | Behavior |
|---|---|
| Ed25519 | Standard library only, following RFC 8032 section 6. Passes the RFC 8032 test vectors 1–3, and 20 random keys and messages match OpenSSL 3.0 byte for byte. Rejects non-canonical `S` and malformed keys. |
| Signature envelope | `<artifact>.sig`, JSON `soma-signature-v1`. It signs a domain-separated canonical JSON of the key id, artifact name, size, SHA-256, and time. |
| Keys | `key-generate PATH` writes a secret key (mode 0600, never overwrites an existing file). Trusted public keys come from `configs/trusted_keys.json` (committed, public only) and `$SOMA_HOME/trusted_keys.json` (`key-trust HEX`). A keyring entry whose id does not match its key is rejected. |
| `sign` / `verify` | Write and check `<file>.sig`. |
| `brain-import-soma` | If a `.sig` exists, it must verify, or nothing is imported. Unsigned files still import and are marked `"signature": "unsigned"`. `--require-signature` refuses them. |
| `brain-download URL NAME` | HTTPS or `file://` only, with a size cap (default 8 GiB). It always needs a trusted signature, and verifies before importing. The result records provenance (key, hash, URL). |
| `engine-install URL` | The same for a prebuilt engine package (`soma-engine-<platform>.tar.gz`). It extracts only the three expected regular files, checks their `SHA256SUMS`, and swaps them into `$SOMA_HOME/engine` atomically. A URL ending in `/` picks this machine's platform. |
| Engine lookup | `$SOMA_MIXER_SERVE`, then a source build in the checkout, then the installed engine. |
| `install.sh` | Builds the engine with `cargo` when present. Otherwise it runs `engine-install "$SOMA_ENGINE_URL"` if that is set. |
| `doctor` | Reports trusted keys, with a hint when there are none. |
| Release workflow | On an `engine-v*` tag (or a manual run), it builds and tests the engine on Linux x86_64 and macOS arm64 and x86_64. It packages the binaries deterministically, signs them if the `SOMA_RELEASE_KEY` secret exists, and attaches them to the release. |

## Who holds the key

No secret key is in this repository, and none was generated for it. The
release key belongs to the project owner:

```bash
python3 -m soma.service.main key-generate ~/soma-release.key   # keep offline, back it up
# the command prints the key id and public key: commit the public key in
# configs/trusted_keys.json as {"keys": [{"key_id", "public_key", "label"}]}
# and store the file's contents as the SOMA_RELEASE_KEY repository secret
```

Until a public key is in `configs/trusted_keys.json`, downloads and
engine installs fail with "no trusted keys". That failure is intended.

## Boundaries

- Ed25519 here is not constant-time Python. Signing happens on the
  owner's machine, where timing side channels are not a concern; verifying
  uses public data only.
- There is no key rotation or revocation list yet. A compromised key is
  handled by removing it from `configs/trusted_keys.json` and releasing
  again.
- macOS builds come from CI and have not been run on a real Mac in this
  project. Windows is not packaged.
- There is no auto-updater. Updating is running `engine-install` (or
  `brain-download`) again with a newer release.
