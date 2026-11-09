from liollm.tokenizer import BPETokenizer

TEXT = "Hallo Welt! Hello world, hello world. Größe, Übermut & Emojis 🚀🚀 123456\n" * 20


def test_encode_decode(tmp_path):
    tok = BPETokenizer()
    tok.train(TEXT, vocab_size=300)
    assert tok.vocab_size == 300

    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT
    # nach dem Training sollten es weniger Tokens als Bytes sein
    assert len(ids) < len(TEXT.encode("utf-8"))

    # speichern und laden testen
    tok.save(tmp_path / "tok.json")
    tok2 = BPETokenizer.load(tmp_path / "tok.json")
    assert tok2.encode(TEXT) == ids


def test_unbekannter_text():
    # auch Text, der beim Training nicht vorkam, muss funktionieren
    tok = BPETokenizer()
    tok.train(TEXT, vocab_size=280)
    text = "Völlig neuer Text mit 日本語 und \t Tabs."
    assert tok.decode(tok.encode(text)) == text
