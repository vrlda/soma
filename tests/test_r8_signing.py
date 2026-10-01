import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from unittest import mock

from soma.persistence import signing
from soma.service import distribution
from soma.service.main import main
from soma.service.store import BrainStore

# RFC 8032 section 7.1, tests 1-3: (secret, public, message, signature).
RFC8032 = (
    ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
     "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
     "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
    ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
     "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
     "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
    ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
     "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
     "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
)

SECRET = bytes(range(32))
OTHER = bytes(range(1, 33))


class Ed25519Tests(unittest.TestCase):
    def test_rfc8032_vectors(self):
        for secret, public, message, signature in RFC8032:
            secret, public = bytes.fromhex(secret), bytes.fromhex(public)
            message, signature = bytes.fromhex(message), bytes.fromhex(signature)
            self.assertEqual(signing.public_key(secret), public)
            self.assertEqual(signing.sign_bytes(secret, message), signature)
            self.assertTrue(signing.verify_bytes(public, message, signature))
            self.assertFalse(signing.verify_bytes(public, message + b"!", signature))
            flipped = bytes([signature[0] ^ 1]) + signature[1:]
            self.assertFalse(signing.verify_bytes(public, message, flipped))

    def test_rejects_noncanonical_and_malformed_input(self):
        public = signing.public_key(SECRET)
        signature = signing.sign_bytes(SECRET, b"m")
        too_big_s = signature[:32] + (int.from_bytes(signature[32:], "little")
                                      + signing._L).to_bytes(32, "little")
        self.assertFalse(signing.verify_bytes(public, b"m", too_big_s))
        self.assertFalse(signing.verify_bytes(public[:31], b"m", signature))
        self.assertFalse(signing.verify_bytes(public, b"m", signature[:63]))


class Workspace(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = self.directory.name
        self.home = os.path.join(self.root, "home")
        patcher = mock.patch.dict(os.environ, {"SOMA_HOME": self.home})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.directory.cleanup)
        self.store = BrainStore(os.path.join(self.root, "brains"))
        self.store.create("source", max_circuits=1024)
        self.artifact = os.path.join(self.root, "source.soma")
        self.store.export_soma("source", self.artifact)

    def trust(self, secret=SECRET):
        return distribution.trust_key(signing.public_key(secret).hex(), "test")


class SignedFileTests(Workspace):
    def test_sign_and_verify_round_trip(self):
        self.trust()
        envelope = signing.sign_file(self.artifact, SECRET)
        verified = signing.verify_file(self.artifact, distribution.keyring())
        self.assertEqual(verified, envelope)
        self.assertEqual(envelope["key_id"], signing.key_id(signing.public_key(SECRET)))

    def test_tampering_and_untrusted_keys_fail(self):
        signing.sign_file(self.artifact, SECRET)
        with self.assertRaisesRegex(ValueError, "untrusted key"):
            signing.verify_file(self.artifact, distribution.keyring())
        self.trust()
        with open(self.artifact, "ab") as handle:
            handle.write(b"x")
        with self.assertRaisesRegex(ValueError, "does not match"):
            signing.verify_file(self.artifact, distribution.keyring())
        signing.sign_file(self.artifact, SECRET)
        with open(self.artifact + ".sig") as handle:
            envelope = json.load(handle)
        envelope["artifact_bytes"] += 1
        with open(self.artifact + ".sig", "w") as handle:
            json.dump(envelope, handle)
        with self.assertRaisesRegex(ValueError, "does not verify"):
            signing.verify_file(self.artifact, distribution.keyring())
        with open(self.artifact + ".sig", "w") as handle:
            handle.write("{not json")
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            signing.verify_file(self.artifact, distribution.keyring())

    def test_inconsistent_keyring_entry_is_rejected(self):
        path = os.path.join(self.root, "bad.json")
        with open(path, "w") as handle:
            json.dump({"keys": [{"key_id": "0" * 16,
                                 "public_key": signing.public_key(SECRET).hex()}]}, handle)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            signing.load_keyring([path])

    def test_secret_key_file_is_private_and_never_overwritten(self):
        path = os.path.join(self.root, "release.key")
        entry = signing.write_secret_key(path)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(signing.key_id(bytes.fromhex(entry["public_key"])), entry["key_id"])
        with self.assertRaises(FileExistsError):
            signing.write_secret_key(path)
        self.assertEqual(signing.public_key(signing.read_secret_key(path)).hex(),
                         entry["public_key"])


class SignedImportTests(Workspace):
    def test_unsigned_import_is_marked_or_refused(self):
        result = distribution.import_signed(self.store, self.artifact, "copy")
        self.assertEqual(result["provenance"], {"signature": "unsigned"})
        with self.assertRaisesRegex(ValueError, "unsigned"):
            distribution.import_signed(self.store, self.artifact, "strict",
                                       require_signature=True)
        self.assertNotIn("strict", os.listdir(self.store.root))

    def test_signed_import_verifies_and_bad_signature_blocks_import(self):
        self.trust()
        signing.sign_file(self.artifact, SECRET)
        result = distribution.import_signed(self.store, self.artifact, "copy",
                                            require_signature=True)
        self.assertEqual(result["provenance"]["signature"], "verified")
        signing.sign_file(self.artifact, OTHER)
        with self.assertRaisesRegex(ValueError, "untrusted"):
            distribution.import_signed(self.store, self.artifact, "other")
        self.assertNotIn("other", os.listdir(self.store.root))

    def test_download_requires_trusted_signature_and_limits_size(self):
        url = "file://" + os.path.abspath(self.artifact)
        with self.assertRaisesRegex(ValueError, "no trusted keys"):
            distribution.download(self.store, url, "fetched")
        self.trust()
        with self.assertRaises(OSError):  # no .sig next to the artifact
            distribution.download(self.store, url, "fetched")
        signing.sign_file(self.artifact, SECRET)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            distribution.download(self.store, url, "fetched", max_bytes=100)
        keep = os.path.join(self.root, "kept.soma")
        result = distribution.download(self.store, url, "fetched", keep=keep)
        self.assertEqual(result["provenance"]["signature"], "verified")
        self.assertEqual(result["provenance"]["source_url"], url)
        signing.verify_file(keep, distribution.keyring())
        with self.assertRaisesRegex(ValueError, "https"):
            distribution.download(self.store, "http://example.invalid/x.soma", "plain")

    def test_cli_key_sign_verify_import(self):
        key = os.path.join(self.root, "release.key")
        brains = ["--root", self.store.root]
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        errors = contextlib.redirect_stderr(io.StringIO())
        errors.__enter__()
        self.addCleanup(errors.__exit__, None, None, None)
        self.assertEqual(main(brains + ["key-generate", key]), 0)
        public = signing.public_key(signing.read_secret_key(key)).hex()
        self.assertEqual(main(brains + ["key-trust", public, "--label", "release"]), 0)
        self.assertEqual(main(brains + ["sign", self.artifact, "--key", key]), 0)
        self.assertEqual(main(brains + ["verify", self.artifact]), 0)
        self.assertEqual(main(brains + ["brain-import-soma", self.artifact, "cli",
                                        "--require-signature"]), 0)
        self.assertEqual(main(brains + ["key-generate", key]), 1)


ENGINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "engine", "soma-engine", "target", "release")


@unittest.skipUnless(os.access(os.path.join(ENGINE_DIR, "soma-mixer-serve"), os.X_OK),
                     "engine not built")
class EngineInstallTests(Workspace):
    def package(self):
        from scripts.package_engine import package
        return package(ENGINE_DIR, os.path.join(self.root, "dist"), "test-platform")

    def test_package_is_deterministic_and_installs_after_verification(self):
        from soma.memory import engine
        first = self.package()
        with open(first, "rb") as handle:
            content = handle.read()
        with open(self.package(), "rb") as handle:
            self.assertEqual(handle.read(), content)
        url = "file://" + os.path.abspath(first)
        self.trust()
        with self.assertRaises(OSError):  # unsigned: no .sig to fetch
            distribution.install_engine(url)
        self.assertFalse(os.path.exists(engine.installed_binary()))
        signing.sign_file(first, SECRET)
        result = distribution.install_engine(url)
        self.assertEqual(result["signature"], "verified")
        installed = engine.installed_binary()
        self.assertTrue(os.access(installed, os.X_OK))
        with mock.patch.object(engine, "DEFAULT_BINARY", os.path.join(self.root, "absent")):
            self.assertEqual(engine.engine_binary(), installed)
            with engine.EngineMixingMemory(config={"max_circuits": 4096}) as memory:
                memory.observe_bytes(b"abc")
                self.assertGreater(memory.summary()["events_seen"], 0)

    def test_unexpected_package_members_are_refused(self):
        import tarfile
        bad = os.path.join(self.root, "bad.tar.gz")
        with tarfile.open(bad, "w:gz") as archive:
            archive.add(os.path.join(ENGINE_DIR, "soma-mixer-serve"), arcname="../escape")
        self.trust()
        signing.sign_file(bad, SECRET)
        with self.assertRaisesRegex(ValueError, "unexpected engine package"):
            distribution.install_engine("file://" + bad)
