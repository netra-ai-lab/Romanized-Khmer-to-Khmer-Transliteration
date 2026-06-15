from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request


def create_app() -> Flask:
    """Flask application factory."""
    template_folder = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_folder))

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/transliterate", methods=["POST"])
    def transliterate_endpoint():
        from khmer_transliterator import transliterate_with_dict
        data = request.json or {}
        word = data.get("word", "").strip()
        if not word:
            return jsonify({"candidates": []})
        candidates = transliterate_with_dict(word, n=5)
        return jsonify({"candidates": candidates})

    return app
