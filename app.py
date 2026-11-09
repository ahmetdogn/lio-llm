# Kleine Weboberflaeche zum Ausprobieren vom Modell
#
# Starten mit:  python app.py
# dann im Browser http://localhost:5000 oeffnen

import torch
from flask import Flask, jsonify, render_template, request

from generate import generate_text, load_model

app = Flask(__name__)

print("lade Modell...")
model, tok = load_model()
print("Modell geladen")


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
    max_tokens = min(int(data.get("max_tokens", 150)), 500)
    temperature = float(data.get("temperature", 0.8))
    temperature = max(0.1, min(temperature, 2.0))

    with torch.no_grad():
        text = generate_text(model, tok, prompt, max_tokens, temperature)
    return jsonify({"text": text})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
