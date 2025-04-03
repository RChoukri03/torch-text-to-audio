from flask import Flask, request, jsonify, send_file
import tempfile
import torchaudio
from utils.arabicTTSwrapper import ArabicTTSWrapper

app = Flask(__name__)
tts_system = ArabicTTSWrapper()

@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = data.get("text", "")
    model_key = data.get("model", "model1")

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    try:
        wav, phonemes = tts_system.synthesize(text, model_key=model_key)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        torchaudio.save(tmp_file.name, wav.unsqueeze(0), 22050)
        return send_file(tmp_file.name, mimetype="audio/wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/phonemes", methods=["POST"])
def phonemes():
    data = request.get_json()
    text = data.get("text", "")
    model_key = data.get("model", "model1")

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    try:
        _, phonemes = tts_system.synthesize(text, model_key=model_key)
        return jsonify({"phonemes": phonemes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
