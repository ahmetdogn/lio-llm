# Bereitet die Daten vor:
#  1. Text laden (standardmaessig Tiny Shakespeare)
#  2. Tokenizer trainieren
#  3. Text in Token-IDs umwandeln und als train.bin / val.bin speichern
#
# Benutzung:
#   python data/prepare.py
#   python data/prepare.py --input mein_text.txt

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))  # damit liollm gefunden wird
from liollm import BPETokenizer

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--vocab-size", type=int, default=1024)
args = parser.parse_args()

if args.input:
    text = Path(args.input).read_text(encoding="utf-8")
else:
    path = DATA_DIR / "input.txt"
    if not path.exists():
        print("lade Tiny Shakespeare herunter...")
        urllib.request.urlretrieve(URL, path)
    text = path.read_text(encoding="utf-8")

print("Anzahl Zeichen:", len(text))

tok = BPETokenizer()
tok.train(text, args.vocab_size, verbose=True)
tok.save(DATA_DIR / "tokenizer.json")

# uint16 reicht, solange vocab_size < 65536
ids = np.array(tok.encode(text), dtype=np.uint16)
print("Anzahl Tokens:", len(ids))
print("Bytes pro Token:", round(len(text.encode()) / len(ids), 2))

# 90% Training, 10% Validierung
n = int(0.9 * len(ids))
ids[:n].tofile(DATA_DIR / "train.bin")
ids[n:].tofile(DATA_DIR / "val.bin")
print("fertig")
