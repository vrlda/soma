"""Brain store: named persistent brains on disk.

Layout per brain `<root>/<name>/`:
  brain.json      organism checkpoint (owns background sequence memory)
  dialogue.json   turn-scoped dialogue memory state
  episodic.json   episodic buffer state
  manifest.json   identity, ancestor, created, description, hashes
"""

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time

from ..memory import EpisodicBuffer, SequenceCircuitMemory
from ..organism import Organism

MANIFEST_VERSION = 1


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BrainStore(object):
    def __init__(self, root, state_budget_bytes=None):
        self.root = os.path.abspath(root)
        if state_budget_bytes is not None and (
                isinstance(state_budget_bytes, bool) or int(state_budget_bytes) < 1):
            raise ValueError("state budget must be a positive integer")
        self.state_budget_bytes = None if state_budget_bytes is None else int(state_budget_bytes)
        if not os.path.isdir(self.root):
            os.makedirs(self.root)

    def _dir(self, name):
        self._check_name(name)
        return os.path.join(self.root, name)

    @staticmethod
    def _check_name(name):
        if not name or not isinstance(name, str) or "/" in name or name.startswith("."):
            raise ValueError("invalid brain name: %r" % (name,))

    def _paths(self, name):
        directory = self._dir(name)
        return {
            "dir": directory,
            "brain": os.path.join(directory, "brain.json"),
            "dialogue": os.path.join(directory, "dialogue.json"),
            "episodic": os.path.join(directory, "episodic.json"),
            "manifest": os.path.join(directory, "manifest.json"),
        }

    def list(self):
        names = []
        for entry in sorted(os.listdir(self.root)):
            directory = os.path.join(self.root, entry)
            if os.path.isdir(directory) and os.path.exists(os.path.join(directory, "manifest.json")):
                names.append(entry)
        return names

    def create(self, name, description="", symbols=(0, 1), max_order=16,
               max_circuits=131072, seed=0):
        paths = self._paths(name)
        if os.path.exists(paths["dir"]):
            raise ValueError("brain already exists: %s" % name)
        os.makedirs(paths["dir"])
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=seed)
        organism.enable_sequence_memory(tuple(symbols), max_order=max_order,
                                        max_circuits=max_circuits)
        organism.save(paths["brain"])
        episodic = EpisodicBuffer()
        with open(paths["episodic"], "w") as handle:
            json.dump(episodic.state_dict(), handle, sort_keys=True)
        from ..memory import SequenceCircuitMemory
        dialogue = SequenceCircuitMemory((0, 1), max_order=256, max_circuits=16384)
        with open(paths["dialogue"], "w") as handle:
            json.dump(dialogue.state_dict(), handle, sort_keys=True)
        manifest = {
            "version": MANIFEST_VERSION,
            "name": name,
            "identity": _sha256_file(paths["brain"])[:16],
            "ancestor": None,
            "created_utc": int(time.time()),
            "description": str(description),
            "symbols": list(symbols),
            "max_order": max_order,
            "max_circuits": max_circuits,
        }
        with open(paths["manifest"], "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest

    def load(self, name):
        from ..memory import SequenceCircuitMemory
        from ..persistence import recover_if_needed
        paths = self._paths(name)
        if not os.path.exists(paths["manifest"]) and not os.path.exists(paths["brain"] + ".prev"):
            raise ValueError("unknown brain: %s" % name)
        recover_if_needed(paths)
        if not os.path.exists(paths["manifest"]):
            raise ValueError("unknown brain: %s" % name)
        organism = Organism.load(paths["brain"])
        with open(paths["episodic"]) as handle:
            episodic = EpisodicBuffer.from_state_dict(json.load(handle))
        if os.path.exists(paths["dialogue"]):
            with open(paths["dialogue"]) as handle:
                dialogue = SequenceCircuitMemory.from_state_dict(json.load(handle))
        else:
            dialogue = SequenceCircuitMemory((0, 1), max_order=256, max_circuits=16384)
        with open(paths["manifest"]) as handle:
            manifest = json.load(handle)
        return organism, episodic, manifest, dialogue

    def save(self, name, organism, episodic, description=None, dialogue=None):
        from ..persistence import begin_save, end_save, rotate_previous
        paths = self._paths(name)
        if not os.path.exists(paths["manifest"]):
            raise ValueError("unknown brain: %s" % name)
        organism.validate()
        episodic.validate()
        with open(paths["manifest"]) as handle:
            manifest = json.load(handle)
        begin_save(paths["dir"])
        rotate_previous(paths)
        organism.save(paths["brain"])
        with open(paths["episodic"], "w") as handle:
            json.dump(episodic.state_dict(), handle, sort_keys=True)
        if dialogue is None:
            from ..memory import SequenceCircuitMemory
            dialogue = SequenceCircuitMemory((0, 1), max_order=256, max_circuits=16384)
        dialogue.validate()
        with open(paths["dialogue"], "w") as handle:
            json.dump(dialogue.state_dict(), handle, sort_keys=True)
        manifest["identity"] = _sha256_file(paths["brain"])[:16]
        if description is not None:
            manifest["description"] = str(description)
        with open(paths["manifest"], "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        end_save(paths["dir"])
        if self.state_budget_bytes is not None:
            total = sum(os.path.getsize(paths[key]) for key in
                        ("brain", "dialogue", "episodic", "manifest")
                        if paths.get(key) is not None and os.path.exists(paths[key]))
            if total > self.state_budget_bytes:
                raise ValueError("brain state %d bytes exceeds budget %d" % (
                    total, self.state_budget_bytes))
        return manifest

    def clone(self, source, destination, reason):
        if not reason:
            raise ValueError("clone requires a reason")
        source_paths = self._paths(source)
        destination_paths = self._paths(destination)
        if not os.path.exists(source_paths["manifest"]):
            raise ValueError("unknown brain: %s" % source)
        if os.path.exists(destination_paths["dir"]):
            raise ValueError("brain already exists: %s" % destination)
        shutil.copytree(source_paths["dir"], destination_paths["dir"],
                        ignore=shutil.ignore_patterns("journal.log", "*.prev"))
        with open(destination_paths["manifest"]) as handle:
            manifest = json.load(handle)
        manifest["name"] = destination
        manifest["ancestor"] = manifest.get("identity")
        manifest["identity"] = _sha256_file(destination_paths["brain"])[:16]
        manifest["created_utc"] = int(time.time())
        manifest["clone_reason"] = str(reason)
        with open(destination_paths["manifest"], "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest

    def inspect(self, name):
        organism, episodic, manifest, dialogue = self.load(name)
        memory = organism.sequence_memory
        return {
            "name": name,
            "identity": manifest.get("identity"),
            "ancestor": manifest.get("ancestor"),
            "description": manifest.get("description", ""),
            "sequence_circuits": len(memory.circuits) if memory is not None else 0,
            "sequence_events": memory.events_seen if memory is not None else 0,
            "dialogue_circuits": len(dialogue.circuits),
            "episodic_entries": len(episodic.entries),
            "episodic_hits": episodic.hits,
        }

    def backup(self, name, destination_dir):
        paths = self._paths(name)
        if not os.path.exists(paths["manifest"]):
            raise ValueError("unknown brain: %s" % name)
        target = os.path.join(os.path.abspath(destination_dir), name)
        if os.path.exists(target):
            raise ValueError("backup target exists: %s" % target)
        shutil.copytree(paths["dir"], target)
        return target

    def restore(self, name, source_dir):
        destination_paths = self._paths(name)
        if os.path.exists(destination_paths["dir"]):
            raise ValueError("brain already exists: %s" % name)
        source = os.path.join(os.path.abspath(source_dir), name)
        if not os.path.exists(os.path.join(source, "manifest.json")):
            raise ValueError("no backup found: %s" % source)
        shutil.copytree(source, destination_paths["dir"])
        organism, episodic, _, _ = self.load(name)
        organism.validate()
        episodic.validate()
        return self.inspect(name)

    def export(self, name, path):
        paths = self._paths(name)
        if not os.path.exists(paths["manifest"]):
            raise ValueError("unknown brain: %s" % name)
        with tempfile.TemporaryDirectory() as scratch:
            staged = os.path.join(scratch, name)
            shutil.copytree(paths["dir"], staged,
                            ignore=shutil.ignore_patterns("journal.log", "*.prev"))
            with tarfile.open(path, "w:gz") as archive:
                archive.add(staged, arcname=name)
        return {"path": os.path.abspath(path), "sha256": _sha256_file(path)}

    def export_soma(self, name, path):
        """Write the versioned binary `.soma` artifact for download."""
        from ..persistence import write_soma
        paths = self._paths(name)
        if not os.path.exists(paths["manifest"]):
            raise ValueError("unknown brain: %s" % name)
        chunks = {}
        for key in ("manifest", "brain", "dialogue", "episodic"):
            with open(paths[key], "rb") as handle:
                chunks[key + ".json"] = handle.read()
        return write_soma(path, chunks)

    def import_soma(self, path, name):
        """Import a `.soma` artifact with full hash verification."""
        from ..persistence import read_soma
        destination_paths = self._paths(name)
        if os.path.exists(destination_paths["dir"]):
            raise ValueError("brain already exists: %s" % name)
        chunks = read_soma(path)
        os.makedirs(destination_paths["dir"])
        mapping = {"manifest.json": "manifest", "brain.json": "brain",
                   "dialogue.json": "dialogue", "episodic.json": "episodic"}
        for chunk_name, key in mapping.items():
            with open(destination_paths[key], "wb") as handle:
                handle.write(chunks[chunk_name])
        organism, episodic, _, _ = self.load(name)
        organism.validate()
        episodic.validate()
        manifest = self.inspect(name)
        manifest["name"] = name
        return manifest

    def import_brain(self, path, name):
        destination_paths = self._paths(name)
        if os.path.exists(destination_paths["dir"]):
            raise ValueError("brain already exists: %s" % name)
        with tempfile.TemporaryDirectory() as scratch:
            with tarfile.open(path, "r:gz") as archive:
                archive.extractall(scratch)
            roots = [entry for entry in os.listdir(scratch)]
            if len(roots) != 1:
                raise ValueError("export bundle must hold exactly one brain")
            shutil.move(os.path.join(scratch, roots[0]), destination_paths["dir"])
        organism, episodic, _, _ = self.load(name)
        organism.validate()
        episodic.validate()
        return self.inspect(name)
