from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route("/metrics")
def get_metrics():
    return jsonify({
        "f1_score": 0.83,
        "train_loss": [0.58, 0.41, 0.33, ...],  
        "val_loss": [0.61, 0.45, 0.37, ...]
    })

@app.route("/image/<name>")
def get_image(name):
    path = os.path.join("static/images")
    return send_from_directory(path, name)

if __name__ == "__main__":
    app.run(debug=True)
