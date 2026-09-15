"""Prosumer chat, teach, and forget over a stored brain."""

from ..evaluation.dialogue import bits_to_bytes, respond, teach_fact, text_to_bits
from ..transducers.text_bytes import decode_bits


def _decode_response(raw):
    try:
        return bytes(raw).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return bytes(raw).decode("utf-8", errors="ignore")


def teach_text(store, name, text, provenance="approved-notes"):
    """Ingest approved text: background-rate exposure, recorded provenance."""
    if isinstance(text, str):
        text = text.encode("utf-8")
    organism, episodic, _, dialogue = store.load(name)
    memory = organism.sequence_memory
    if memory is None:
        raise ValueError("brain has no sequence memory")
    from ..evaluation.english import bit_stream
    bits, _ = bit_stream(bytes(text))
    for symbol in bits:
        memory.observe(symbol, learn=True)
    store.save(name, organism, episodic, dialogue=dialogue)
    return {"bytes": len(text), "bits": len(bits), "provenance": provenance}


def correct(store, name, question, answer, fact_text=None):
    """Teach a correction: episodic rule plus dialogue exposure."""
    organism, episodic, _, dialogue = store.load(name)
    fact = fact_text if fact_text is not None else "USER %s AGENT %s " % (question, answer)
    teach_fact(dialogue, episodic, fact, question, answer, provenance="correction")
    store.save(name, organism, episodic, dialogue=dialogue)
    return {"question": question, "answer": answer}


def chat_turn(store, name, user_text, max_bytes=24, seed=0):
    """One attributed turn: USER text in, AGENT text out."""
    organism, episodic, _, dialogue = store.load(name)
    memory = organism.sequence_memory
    if memory is None:
        raise ValueError("brain has no sequence memory")
    user_turn = "USER %s " % user_text
    for bit in text_to_bits(user_turn):
        dialogue.observe(bit, learn=True)
    store.save(name, organism, episodic, dialogue=dialogue)
    organism2, episodic2, _, dialogue2 = store.load(name)
    raw = respond(organism2.sequence_memory, dialogue2, episodic2,
                  user_turn, max_bytes=max_bytes, seed=seed)
    return _decode_response(raw)


def forget_fact(store, name, question):
    """Remove episodic rules for a question. Returns removed count."""
    organism, episodic, _, dialogue = store.load(name)
    from ..evaluation.dialogue import text_to_bits as _bits
    trigger = tuple(_bits(question))
    removed = [entry_id for entry_id, entry in episodic.entries.items()
               if entry["trigger"] == trigger]
    for entry_id in removed:
        episodic.remove(entry_id)
    store.save(name, organism, episodic, dialogue=dialogue)
    return {"removed": len(removed), "question": question}


def repl(store, name):
    """Interactive chat loop. /quit exits, /teach <path> ingests a file."""
    print("chatting with brain '%s' (/quit to exit, /teach <path> to ingest)" % name)
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line.startswith("/teach "):
            path = line[len("/teach "):].strip()
            with open(path, "rb") as handle:
                result = teach_text(store, name, handle.read(), provenance=path)
            print("learned %d bytes (%d bits) from %s" % (result["bytes"], result["bits"], path))
            continue
        print("soma> %s" % chat_turn(store, name, line))
