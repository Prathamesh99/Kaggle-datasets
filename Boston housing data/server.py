import numpy as np
from flask import Flask, request, jsonify, json
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    #data = json.loads(data1.decode("utf-8"))
    predict_request = [data['lstat'], data['rm']]
    predict_request = np.array(predict_request).reshape(1,-1)
    prediction = model.predict(predict_request)
    output = [prediction[0]]
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
