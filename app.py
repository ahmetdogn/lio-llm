# Weboberflaeche zum Ausprobieren vom Modell
#
# Lokal starten mit:  python app.py  (dann http://localhost:5000 oeffnen)
# Laeuft auch auf Vercel. Dort wird die numpy-Version vom Modell benutzt, weil PyTorch
# zu gross fuer Vercel ist (siehe liollm/numpy_model.py)

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from liollm.numpy_model import NumpyGPT
from liollm.tokenizer import BPETokenizer

BASE = Path(__file__).parent

app = Flask(__name__)

model = NumpyGPT(BASE / "checkpoints" / "model.npz")
tok = BPETokenizer.load(BASE / "data" / "tokenizer.json")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if prompt == "":
        prompt = "\n"

    # Werte begrenzen, damit niemand 100000 Tokens anfragt
    max_tokens = min(int(data.get("max_tokens", 150)), 300)
    temperature = float(data.get("temperature", 0.8))
    temperature = max(0.1, min(temperature, 2.0))

    ids = model.generate(tok.encode(prompt)[-200:], max_tokens, temperature)
    return jsonify({"text": tok.decode(ids)})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
