from flask import Flask, request, jsonify, abort

from constants import FEATURES
from utils.logger import app_logger

# creating instance of the class
app = Flask(__name__)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    app_logger.info("Input - %s", data)

    # TODO - Payload type validation
    if not data:
        abort(400, "Empty payload.")

    if not set(FEATURES).issubset(set(data.keys())):
        abort(400, "Send all required fields")

    inp = [data[f] for f in FEATURES]
    pct_change = model.predict([inp])
    return jsonify(pct_change=pct_change[0])


if __name__ == '__main__':
    import pickle as p

    app_logger.info("Loading Model...")
    model_file = 'models/tree.pkl'
    model = p.load(open(model_file, 'rb'))
    app_logger.info("Loaded.")
    app.run(debug=True, host='0.0.0.0')
