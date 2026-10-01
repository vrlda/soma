"""SOMA prosumer command line: brains, chat, teach, serve, doctor."""

import argparse
import json
import os
import sys

DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), ".soma", "brains")


def _store(args):
    from .store import BrainStore
    return BrainStore(args.root)


def cmd_brain_create(args):
    from .store import BrainStore
    store = BrainStore(args.root)
    memory = args.memory
    if memory == "auto":
        from ..memory.engine import engine_available
        memory = "mixing" if engine_available() else "suffix"
    manifest = store.create(args.name, description=args.description or "",
                            memory=memory, max_circuits=args.max_circuits)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_brain_list(args):
    print(json.dumps(_store(args).list(), indent=2))
    return 0


def cmd_brain_clone(args):
    manifest = _store(args).clone(args.source, args.destination, args.reason or "cli clone")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_brain_teach(args):
    from .chat import teach_text
    store = _store(args)
    total = {"bytes": 0, "bits": 0}
    for path in args.paths:
        with open(path, "rb") as handle:
            result = teach_text(store, args.name, handle.read(), provenance=path,
                                trust=args.trust)
        total["bytes"] += result["bytes"]
        total["bits"] += result["bits"]
    print(json.dumps(total, indent=2, sort_keys=True))
    return 0


def cmd_brain_approve(args):
    from .chat import approve_staged
    result = approve_staged(_store(args), args.name, weight=args.weight)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_brain_correct(args):
    from .chat import correct
    result = correct(_store(args), args.name, args.question, args.answer)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_brain_forget(args):
    from .chat import forget_fact
    result = forget_fact(_store(args), args.name, args.question)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_brain_inspect(args):
    print(json.dumps(_store(args).inspect(args.name), indent=2, sort_keys=True))
    return 0


def cmd_brain_backup(args):
    print(_store(args).backup(args.name, args.destination))
    return 0


def cmd_brain_restore(args):
    print(json.dumps(_store(args).restore(args.name, args.source), indent=2, sort_keys=True))
    return 0


def cmd_brain_export(args):
    print(json.dumps(_store(args).export(args.name, args.path), indent=2, sort_keys=True))
    return 0


def cmd_brain_import(args):
    print(json.dumps(_store(args).import_brain(args.path, args.name), indent=2, sort_keys=True))
    return 0


def cmd_brain_export_soma(args):
    print(json.dumps(_store(args).export_soma(args.name, args.path), indent=2, sort_keys=True))
    return 0


def cmd_brain_import_soma(args):
    from .distribution import import_signed
    result = import_signed(_store(args), args.path, args.name,
                           require_signature=args.require_signature,
                           extra_keyrings=args.keyring or ())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_brain_download(args):
    from .distribution import download
    result = download(_store(args), args.url, args.name, max_bytes=args.max_bytes,
                      keep=args.keep, extra_keyrings=args.keyring or ())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_engine_install(args):
    from .distribution import engine_platform, install_engine
    url = args.url
    if url.endswith("/"):
        url += "soma-engine-%s.tar.gz" % engine_platform()
    print(json.dumps(install_engine(url, extra_keyrings=args.keyring or ()), indent=2,
                     sort_keys=True))
    return 0


def cmd_key_generate(args):
    from ..persistence.signing import write_secret_key
    try:
        entry = write_secret_key(args.path)
    except FileExistsError:
        raise ValueError("refusing to overwrite existing key %s" % args.path)
    print(json.dumps(dict(entry, secret_key_file=os.path.abspath(args.path)), indent=2,
                     sort_keys=True))
    return 0


def cmd_key_trust(args):
    from .distribution import trust_key
    print(json.dumps(trust_key(args.public_key, args.label, path=args.keyring_file), indent=2,
                     sort_keys=True))
    return 0


def cmd_sign(args):
    from ..persistence.signing import read_secret_key, sign_file
    print(json.dumps(sign_file(args.path, read_secret_key(args.key)), indent=2, sort_keys=True))
    return 0


def cmd_verify(args):
    from ..persistence.signing import verify_file
    from .distribution import keyring
    envelope = verify_file(args.path, keyring(args.keyring or ()))
    print(json.dumps({"verified": True, "key_id": envelope["key_id"],
                      "artifact_sha256": envelope["artifact_sha256"]}, indent=2, sort_keys=True))
    return 0


def cmd_profile(args):
    from .profile import profile_brain
    print(json.dumps(profile_brain(_store(args), args.name, steps=args.steps),
                     indent=2, sort_keys=True))
    return 0


def cmd_chat(args):
    from .chat import repl
    repl(_store(args), args.name)
    return 0


def cmd_serve(args):
    from .serve import serve
    return serve(args.root, host=args.host, port=args.port)


def cmd_doctor(args):
    from .doctor import doctor
    print(json.dumps(doctor(args.root), indent=2, sort_keys=True))
    return 0


def cmd_telemetry(args):
    from .telemetry import set_consent, status
    if args.state == "status":
        print(json.dumps(status(args.root), indent=2, sort_keys=True))
    else:
        print(json.dumps(set_consent(args.root, args.state == "on",
                                     content=args.content), indent=2, sort_keys=True))
    return 0


def cmd_bundle(args):
    from .telemetry import bundle
    print(bundle(args.root, args.path))
    return 0


def cmd_transducers(args):
    from ..transducers import (
        BitScalarTransducer, ScalarStreamTransducer, SymbolBitsTransducer, TextBytesTransducer,
    )
    from ..transducers.history import BitHistoryTransducer, HistoryScalarTransducer
    adapters = [BitScalarTransducer(), ScalarStreamTransducer(), SymbolBitsTransducer(),
                TextBytesTransducer(), BitHistoryTransducer(), HistoryScalarTransducer()]
    print(json.dumps([adapter.spec.to_dict() for adapter in adapters], indent=2, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="soma-service", description="Prosumer SOMA brain service.")
    parser.add_argument("--root", default=os.environ.get("SOMA_BRAINS", DEFAULT_ROOT))
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("brain-create")
    create.add_argument("name")
    create.add_argument("--description", default="")
    create.add_argument("--memory", choices=("auto", "suffix", "mixing"), default="auto",
                        help="language memory: 'mixing' is the engine-served circuit-mixing "
                             "memory (needs the Rust engine built); 'auto' (default) uses it "
                             "when the engine is built, else the original suffix memory")
    create.add_argument("--max-circuits", type=int, default=131072,
                        help="circuit budget of the language memory")
    create.set_defaults(function=cmd_brain_create)

    listed = commands.add_parser("brain-list")
    listed.set_defaults(function=cmd_brain_list)

    clone = commands.add_parser("brain-clone")
    clone.add_argument("source")
    clone.add_argument("destination")
    clone.add_argument("--reason", default="")
    clone.set_defaults(function=cmd_brain_clone)

    teach = commands.add_parser("brain-teach")
    teach.add_argument("name")
    teach.add_argument("paths", nargs="+")
    teach.add_argument("--trust", default="trusted", choices=("trusted", "untrusted"))
    teach.set_defaults(function=cmd_brain_teach)

    approve = commands.add_parser("brain-approve")
    approve.add_argument("name")
    approve.add_argument("--weight", type=int, default=1)
    approve.set_defaults(function=cmd_brain_approve)

    correct = commands.add_parser("brain-correct")
    correct.add_argument("name")
    correct.add_argument("question")
    correct.add_argument("answer")
    correct.set_defaults(function=cmd_brain_correct)

    forget = commands.add_parser("brain-forget")
    forget.add_argument("name")
    forget.add_argument("question")
    forget.set_defaults(function=cmd_brain_forget)

    inspect = commands.add_parser("brain-inspect")
    inspect.add_argument("name")
    inspect.set_defaults(function=cmd_brain_inspect)

    backup = commands.add_parser("brain-backup")
    backup.add_argument("name")
    backup.add_argument("destination")
    backup.set_defaults(function=cmd_brain_backup)

    restore = commands.add_parser("brain-restore")
    restore.add_argument("name")
    restore.add_argument("source")
    restore.set_defaults(function=cmd_brain_restore)

    export = commands.add_parser("brain-export")
    export.add_argument("name")
    export.add_argument("path")
    export.set_defaults(function=cmd_brain_export)

    import_brain = commands.add_parser("brain-import")
    import_brain.add_argument("path")
    import_brain.add_argument("name")
    import_brain.set_defaults(function=cmd_brain_import)

    export_soma = commands.add_parser("brain-export-soma")
    export_soma.add_argument("name")
    export_soma.add_argument("path")
    export_soma.set_defaults(function=cmd_brain_export_soma)

    import_soma = commands.add_parser("brain-import-soma")
    import_soma.add_argument("path")
    import_soma.add_argument("name")
    import_soma.add_argument("--require-signature", action="store_true",
                             help="refuse artifacts without a trusted signature")
    import_soma.add_argument("--keyring", action="append", help="extra trusted-key file")
    import_soma.set_defaults(function=cmd_brain_import_soma)

    download = commands.add_parser("brain-download",
                                   help="download a signed .soma brain, verify, import")
    download.add_argument("url")
    download.add_argument("name")
    download.add_argument("--max-bytes", type=int, default=8 * 1024 ** 3)
    download.add_argument("--keep", help="also save the verified artifact here")
    download.add_argument("--keyring", action="append", help="extra trusted-key file")
    download.set_defaults(function=cmd_brain_download)

    engine_install = commands.add_parser(
        "engine-install", help="install a signed prebuilt engine (URL, or release dir ending /)")
    engine_install.add_argument("url")
    engine_install.add_argument("--keyring", action="append", help="extra trusted-key file")
    engine_install.set_defaults(function=cmd_engine_install)

    key_generate = commands.add_parser("key-generate", help="new Ed25519 signing key (secret)")
    key_generate.add_argument("path")
    key_generate.set_defaults(function=cmd_key_generate)

    key_trust = commands.add_parser("key-trust", help="trust a public key for verification")
    key_trust.add_argument("public_key")
    key_trust.add_argument("--label", default="")
    key_trust.add_argument("--keyring-file", help="default: $SOMA_HOME/trusted_keys.json")
    key_trust.set_defaults(function=cmd_key_trust)

    sign = commands.add_parser("sign", help="write <file>.sig with a secret key")
    sign.add_argument("path")
    sign.add_argument("--key", required=True)
    sign.set_defaults(function=cmd_sign)

    verify = commands.add_parser("verify", help="verify <file>.sig against trusted keys")
    verify.add_argument("path")
    verify.add_argument("--keyring", action="append", help="extra trusted-key file")
    verify.set_defaults(function=cmd_verify)

    profile = commands.add_parser("profile")
    profile.add_argument("name")
    profile.add_argument("--steps", type=int, default=2000)
    profile.set_defaults(function=cmd_profile)

    chat = commands.add_parser("chat")
    chat.add_argument("name")
    chat.set_defaults(function=cmd_chat)

    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(function=cmd_serve)

    doctor = commands.add_parser("doctor")
    doctor.set_defaults(function=cmd_doctor)

    telemetry = commands.add_parser("telemetry")
    telemetry.add_argument("state", choices=("on", "off", "status"), nargs="?", default="status")
    telemetry.add_argument("--content", action="store_true")
    telemetry.set_defaults(function=cmd_telemetry)

    bundle = commands.add_parser("bundle")
    bundle.add_argument("path")
    bundle.set_defaults(function=cmd_bundle)

    transducers = commands.add_parser("transducers")
    transducers.set_defaults(function=cmd_transducers)

    return parser


def main(argv=None):
    import time as _time
    from .telemetry import record as _record
    arguments = build_parser().parse_args(argv)
    started = _time.time()
    try:
        code = arguments.function(arguments)
    except ValueError as error:
        print("error: %s" % error, file=sys.stderr)
        code = 1
    except (OSError, AssertionError) as error:
        print("error: %s" % error, file=sys.stderr)
        code = 2
    _record(arguments.root, arguments.command, int(( _time.time() - started) * 1000), code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
