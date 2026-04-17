import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from Flood.predict_flood import load_flood_model, predict_flood_risk

app = Flask(__name__)
CORS(app)

model = load_flood_model()


@app.route("/generate", methods=["GET"])
def flood_risk():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if lat is None or lon is None:
        return jsonify({"message": "lat and lon required"}), 400

    try:
        score = predict_flood_risk(float(lat), float(lon), model)

        return (
            jsonify({"flood_risk_score": score, "lat": float(lat), "lon": float(lon)}),
            200,
        )

    except Exception as e:
        return jsonify({"message": "Prediction failed", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
