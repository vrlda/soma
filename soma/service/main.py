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
    manifest = store.create(args.name, description=args.description or "")
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
    print(json.dumps(_store(args).import_soma(args.path, args.name), indent=2, sort_keys=True))
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
    import_soma.set_defaults(function=cmd_brain_import_soma)

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

    transducers = commands.add_parser("transducers")
    transducers.set_defaults(function=cmd_transducers)

    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    try:
        return arguments.function(arguments)
    except ValueError as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    except (OSError, AssertionError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
