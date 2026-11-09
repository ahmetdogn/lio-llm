# BPE Tokenizer (Byte Pair Encoding)
#
# Idee: Man startet mit den 256 moeglichen Bytes als Vokabular und fuegt dann immer
# das haeufigste benachbarte Paar zu einem neuen Token zusammen. Das macht man so oft,
# bis man die gewuenschte Vokabulargroesse erreicht hat.
#
# Vorteil von Bytes statt Zeichen: es gibt keine "unbekannten" Zeichen, weil jeder Text
# als UTF-8 Bytes dargestellt werden kann (auch Umlaute, Emojis usw.)
#
# Habe mich dabei an dem Tokenizer-Video von Andrej Karpathy orientiert.

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import regex as re

# Regex aus GPT-4 zum Aufteilen in "Woerter". Dadurch werden keine Merges ueber
# Wortgrenzen hinweg gelernt (sonst lernt er z.B. "e d" als ein Token)
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def merge(ids, pair, new_id):
    # ersetzt jedes Vorkommen von pair in ids durch new_id
    # z.B. ids=[1,2,3,1,2], pair=(1,2), new_id=99  ->  [99,3,99]
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self):
        self.merges = {}  # (id1, id2) -> neue id
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self.pattern = re.compile(SPLIT_PATTERN)
        self.cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256

        # Jedes Wort nur einmal speichern und merken wie oft es vorkommt.
        # Dann muss man nicht jedes mal ueber den ganzen Text gehen -> viel schneller
        word_counts = Counter(self.pattern.findall(text))
        words = {w: tuple(w.encode("utf-8")) for w in word_counts}

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for new_id in range(256, vocab_size):
            # Paare zaehlen
            counts = Counter()
            for w, ids in words.items():
                for p in pairwise(ids):
                    counts[p] += word_counts[w]
            if len(counts) == 0:
                break

            best = max(counts, key=counts.get)
            words = {w: merge(ids, best, new_id) for w, ids in words.items()}
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            if verbose and (new_id - 255) % 50 == 0:
                print(f"merge {new_id - 255}: {self.vocab[new_id]} kommt {counts[best]} mal vor")

        self.cache = {}

    def encode_word(self, word):
        if word in self.cache:
            return self.cache[word]
        ids = tuple(word.encode("utf-8"))
        while len(ids) >= 2:
            # wichtig: die Merges muessen in der gleichen Reihenfolge angewendet werden
            # wie beim Training, also immer das Paar mit der kleinsten merge-id zuerst
            pairs = set(pairwise(ids))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nichts mehr zum mergen
            ids = merge(ids, pair, self.merges[pair])
        self.cache[word] = list(ids)
        return list(ids)

    def encode(self, text):
        ids = []
        for word in self.pattern.findall(text):
            ids += self.encode_word(word)
        return ids

    def decode(self, ids):
        b = b"".join(self.vocab[i] for i in ids)
        # errors="replace", weil das Modell manchmal ungueltige Byte-Folgen generiert
        return b.decode("utf-8", errors="replace")

    def save(self, path):
        data = {"merges": [[a, b, i] for (a, b), i in self.merges.items()]}
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path):
        tok = cls()
        data = json.loads(Path(path).read_text())
        for a, b, i in data["merges"]:
            tok.merges[(a, b)] = i
            tok.vocab[i] = tok.vocab[a] + tok.vocab[b]
        return tok
